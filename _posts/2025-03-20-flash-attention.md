---
layout: post
title: "From Scratch #3: Flash Attention"
date: 2025-03-20
description: "Implementing Flash Attention from scratch: the online softmax, fusing softmax with the value matmul, the O(N) memory argument, the FA2 backward pass, a readable pure-PyTorch version, a real Triton kernel, and benchmarks against naive attention and PyTorch SDPA."
category: "Machine Learning"
tags: ["llm", "attention", "flash-attention", "triton", "gpu", "from-scratch"]
---

Recently I've used Cursor so much that I've realized I'm losing my ability to do ML / pytorch coding. This makes me want to find small / byte-sized topics / algorithms to code from scratch. This time I wanted to go one level below PyTorch and actually write a GPU kernel, so Flash Attention it is.

## What's actually slow about attention
Attention is the operation at the heart of every transformer, however it's quite expensive: its memory cost grows *quadratically* with sequence length. Double the context and we quadruple the memory for the intermediate score matrix. For a while this was the single biggest thing standing between us and long-context models.

Concretely, given queries, keys, and values $Q, K, V \in \mathbb{R}^{N \times d}$ (for one batch and one head), attention is

$$
S = \frac{1}{\sqrt{d}} Q K^\top \in \mathbb{R}^{N \times N}, \qquad
P = \mathrm{softmax}(S), \qquad
O = P V .
$$

Here's the textbook implementation, which we'll keep around as our *correctness oracle* for the rest of the post. It's a direct transcription of the math, so we trust it and compare everything else against it:

```python
import math
import torch

def naive_attention(q, k, v, causal=False, sm_scale=None):
    # q, k, v: (batch, heads, seqlen, head_dim)
    head_dim = q.shape[-1]
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    # S = scale * Q @ K^T  ->  (batch, heads, seqlen_q, seqlen_k).
    # This is the O(N^2) tensor Flash Attention refuses to write to HBM.
    scores = torch.matmul(q, k.transpose(-2, -1)) * sm_scale

    if causal:
        sq, sk = scores.shape[-2], scores.shape[-1]
        row = torch.arange(sq, device=scores.device).view(-1, 1) + (sk - sq)
        col = torch.arange(sk, device=scores.device).view(1, -1)
        scores = scores.masked_fill(col > row, float("-inf"))

    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, v)
```

Look at the shapes. $S$ and $P$ are both $N \times N$. For a sequence of length $N = 32{,}768$ in fp16, one such matrix per (batch, head) is $32768^2 \cdot 2 \approx 2\,\text{GB}$. And there's one *per head, per batch element*. So,

- **Memory:** $O(N^2)$ storage. This is what triggers the out-of-memory errors.
- **Memory bandwidth:** this one is subtler and more important. Attention is *memory-bound*, not compute-bound. The matmuls themselves are cheap; what dominates the wall-clock time is transfering that $N \times N$ matrix out to HBM (the GPU's slow, off-chip high-bandwidth memory) and reading it back for the softmax, then writing $P$, then reading it *again* for $PV$. The GPU spends most of its time waiting on memory, not doing math.

Both of these costs come down to the same thing: *where the bytes live and how far they have to travel*. Flash Attention is designed around the GPU's memory system from top to bottom, so before we touch the algorithm it's worth a quick tour of how a GPU is actually put together. If you want the deep version of this, Austin et al.'s [*How to Think About GPUs*](https://jax-ml.github.io/scaling-book/gpus/) ([Austin et al., 2025](#ref-gpus)) is the best treatment I've read; what follows is the short version we need.

## A quick tour of the GPU

A modern ML GPU is, at heart, a big pile of matrix-multiply units bolted onto a stick of memory. The compute is divided into **Streaming Multiprocessors (SMs)** (an H100 has 132 of them), and each SM is a small, nearly-independent processor. Inside every SM sits a **Tensor Core** (the dedicated matmul unit that does the overwhelming majority of the FLOPs, ~990 bf16 TFLOP/s on an H100), a set of **CUDA cores** driven by *warp schedulers* (the general-purpose vector ALUs that handle everything else: the `exp`, the `max`, the pointwise arithmetic), and its own sliver of fast on-chip memory. The thing to internalize is that there are *hundreds* of these units running concurrently, and the whole game is keeping them fed with data rather than idling on memory.

![A GPU is a large off-chip HBM pool connected through a shared L2 cache to roughly 132 Streaming Multiprocessors (SMs). Each SM contains a Tensor Core for matmuls, warp schedulers driving CUDA cores for vector work, and its own small, fast SMEM/L1 and register file.](/assets/images/flash-attention/gpu_architecture.svg)

The memory is a hierarchy, and the levels trade capacity against speed in a very steep way. Here are the numbers for an H100:

| Level | Scope | Capacity | Bandwidth | Managed by |
| --- | --- | --- | --- | --- |
| Registers | per-thread, inside an SM | 256 KB/SM (~33 MB total) | fastest, ~single-cycle | compiler |
| SMEM / L1 | per-SM, on-chip scratchpad | 256 KB/SM (~33 MB total) | ~10× HBM | **you** (shared memory) |
| L2 cache | shared across all SMs, on-chip | ~50 MB | ~5 TB/s | hardware |
| HBM | off-chip main memory | 80 GB | ~3.35 TB/s | n/a |

![The GPU memory hierarchy as a pyramid: registers and SMEM/L1 at the top (tiny, ~256 KB per SM, fastest), L2 in the middle (~50 MB, ~5 TB/s), and HBM at the bottom (80 GB, ~3.35 TB/s, off-chip). Flash Attention keeps its working set in registers and SMEM; naive attention round-trips the N×N score matrix through HBM.](/assets/images/flash-attention/memory_hierarchy.svg)

There are two things we need to notice. First, the on-chip memory is *tiny* only a couple hundred KB per SM, only tens of MB across the whole chip, but it's blisteringly fast. It's also **programmer-controllable**, which means the SMEM is a scratchpad we explicitly load tiles into (later on in this post we will go deeper into a concrete example to show what this means). Second, HBM is the only level that's *off-chip*, it's where the actual tensors live, and it's roughly an order of magnitude slower than SMEM. Every operand we read from HBM and every result we write back is paid for at that slow bandwidth.

This is precisely why attention is **memory-bound**. The useful work, two matmuls, is cheap for the Tensor Cores. What's expensive is the *traffic*: the naive implementation writes the full $N \times N$ score matrix out to HBM, reads it back to run the softmax, writes $P$, then reads it *yet again* for $PV$. That's several round trips of a multi-gigabyte tensor across the slowest link in the machine, while the Tensor Cores sit mostly idle waiting on it. Shaving FLOPs wouldn't help here; shaving HBM accesses would.

The whole idea behind Flash Attention is to **never write $S$ or $P$ to HBM at all.** The one obstacle is softmax. It's a row-wise operation that normally needs the *whole* row of $S$ at once: you can't normalize until you've seen every entry. So the first thing we need is a way to compute softmax incrementally, seeing the row a block at a time.

## The online (streaming) softmax

The numerically stable softmax of a vector $x \in \mathbb{R}^N$ subtracts the max before exponentiating:

$$
m = \max_j x_j, \qquad
\mathrm{softmax}(x)_j = \frac{e^{x_j - m}}{\sum_{k} e^{x_k - m}} .
$$

This is equivalent to the naive $e^{x_j} / \sum_k e^{x_k}$. The reason we need to do this is $e^{x}$ overflows fp32 at around $x \approx 88$ (since $e^{88} \approx 1.65 \times 10^{38}$, right up against the fp32 max of $\sim 3.4 \times 10^{38}$). Without it, any logit above ~88 gives us `inf`, and `inf / inf = nan`. With it, the largest exponent is $e^{m - m} = e^0 = 1$ and every other term is in $(0, 1]$, so nothing can overflow.

This formula, as written, needs the *entire* row twice: once to find $m = \max_j x_j$, and again to accumulate $\sum_k e^{x_k - m}$ now that we know $m$. A third pass then divides. What we want is a way to fold "find the max," "sum the exponentials," and eventually "normalize" into a *single* left-to-right pass over the row, so we can stream it a block at a time and forget each block once it's been consumed.

**Setting up the recurrence.** Suppose we've processed the first $t$ elements $x_1, \dots, x_t$ and we maintain exactly two scalars:

$$
m_t = \max_{j \le t} x_j, \qquad
\ell_t = \sum_{j \le t} e^{x_j - m_t} .
$$

A new element $x_{t+1}$ (or, in blocked form, a whole new block) arrives. The new max is simply

$$
m_{t+1} = \max(m_t,\, x_{t+1}).
$$

Now for $\ell_{t+1}$. By definition it should be $\sum_{j \le t+1} e^{x_j - m_{t+1}}$. Split off the new term and look at the old sum, which was computed relative to the *old* max $m_t$:

$$
\ell_{t+1}
= \underbrace{\sum_{j \le t} e^{x_j - m_{t+1}}}_{\text{old terms, new shift}} + \; e^{x_{t+1} - m_{t+1}} .
$$

The old terms were stored shifted by $m_t$, but we now need them shifted by $m_{t+1}$. Fixing that is one line of algebra: insert $m_t$ and pull out the constant difference:

$$
\sum_{j \le t} e^{x_j - m_{t+1}}
= \sum_{j \le t} e^{(x_j - m_t) + (m_t - m_{t+1})}
= e^{m_t - m_{t+1}} \sum_{j \le t} e^{x_j - m_t}
= e^{m_t - m_{t+1}} \, \ell_t .
$$

So the whole recurrence is

$$
m_{t+1} = \max(m_t,\, x_{t+1}), \qquad
\ell_{t+1} = \underbrace{e^{m_t - m_{t+1}}}_{\alpha}\, \ell_t \; + \; e^{x_{t+1} - m_{t+1}},
$$

with the initial conditions $m_0 = -\infty$ and $\ell_0 = 0$ (so the first element sets the max and contributes $e^0 = 1$). In blocked form, $x_{t+1}$ becomes a block, the new term becomes a sum over the block, and $m_{t+1} = \max\big(m_t,\ \max_{j \in \text{block}} x_j\big)$; nothing else changes.

**Why the correction factor is exactly $\alpha = e^{m_t - m_{t+1}}$.** Everything we accumulated so far was exponentiated relative to the *old* max $m_t$. When the running max jumps up to $m_{t+1} > m_t$, that old reference point is now too small, so every stored term is too *large* by exactly the factor $e^{m_{t+1} - m_t}$. Multiplying by $\alpha = e^{m_t - m_{t+1}} = 1 / e^{m_{t+1} - m_t} \le 1$ **rescales** the old accumulator down to be consistent with the new max. When the max doesn't change ($m_{t+1} = m_t$), we get $\alpha = e^0 = 1$ and the accumulator passes through untouched, exactly as it should. Note also that $\alpha \le 1$ always, so the rescaling can never blow up: it only ever shrinks the old state, which is precisely what keeps the whole computation numerically bounded.

That single rescaling idea (carry a running max, and multiply the accumulator by $\alpha$ every time the max moves) is the whole engine of Flash Attention ([Milakov & Gimelshein, 2018](#ref-online-softmax)). The rest of the algorithm is figuring out how to drag the output $O = PV$ along inside the very same recurrence.

It's worth convincing ourselves that this actually matters numerically, so I wrote a tiny experiment that computes softmax over a vector three ways as the logit scale grows: (1) naive `exp(x)/sum(exp(x))` with no max subtraction, (2) a streaming sum that tracks *no* running max, and (3) the online recurrence above with rescaling. Here's the core of the online version, which is just the recurrence written out:

```python
def online_softmax(x, block=8):
    n = x.numel()
    m = torch.tensor(float("-inf"))
    l = torch.tensor(0.0)
    num = torch.zeros_like(x)                    # per-element exp values
    for start in range(0, n, block):
        xb = x[start:start + block]
        m_new = torch.maximum(m, xb.max())
        alpha = torch.exp(m - m_new)             # rescale factor for old state
        num[:start] *= alpha                     # rescale everything so far
        l = l * alpha + torch.exp(xb - m_new).sum()
        num[start:start + block] = torch.exp(xb - m_new)
        m = m_new
    return num / l
```

![L1 error of naive softmax vs the online softmax against a stable reference, as the standard deviation of the input logits grows. The naive version's error jumps to the maximum once logits exceed ~88 (fp32 exp overflow), while the online version stays at machine-zero error throughout.](/assets/images/flash-attention/online_softmax.png)

The naive curve is flat at zero until the logit scale crosses ~88, then it falls off a cliff to maximum error: `exp()` has overflowed to `inf` and the result is `nan`. The online softmax sits at machine-precision-zero error the whole way across. The running-max rescaling keeps every intermediate exponential in a safe range. This is exactly why Flash Attention can afford to stream over blocks without ever seeing the full row.

## Fusing softmax with the value matmul (the forward pass)

Second insight: we never want $P$, only $O = PV$. So we carry a third accumulator $O_i$ alongside $(m_i, \ell_i)$ and fold in $V$ as we stream:

$$
O_i = \sum_j \tilde P_{ij}\, V_j, \qquad \tilde P_{ij} = e^{S_{ij} - m_i}, \qquad \text{final } O_i \leftarrow O_i / \ell_i .
$$

The rescale argument is identical to the $\ell_i$ one. After processing blocks $\le j$, the accumulator is $O_i = \sum_{j' \le j} e^{S_{ij'} - m_i}\, V_{j'}$, shifted by the current max $m_i$. When a new block bumps the max to $m_i^{\text{new}}$, every stored term must be reshifted:

$$
\sum_{j' \le j} e^{S_{ij'} - m_i^{\text{new}}}\, V_{j'}
= e^{m_i - m_i^{\text{new}}} \sum_{j' \le j} e^{S_{ij'} - m_i}\, V_{j'}
= \alpha\, O_i .
$$

So the *same* $\alpha = e^{m_i - m_i^{\text{new}}}$ that rescales $\ell_i$ rescales $O_i$, and we then add the new block's contribution $\tilde P_{ij} V_j$. Per key/value block $j$:

$$
\begin{aligned}
S_{ij} &= \tfrac{1}{\sqrt d}\, q_i K_j^\top &&\text{(one $B_r \times B_c$ tile)}\\
m_i^{\text{new}} &= \max\!\big(m_i,\ \max_j S_{ij}\big)\\
\alpha &= e^{m_i - m_i^{\text{new}}}\\
\tilde{P}_{ij} &= e^{S_{ij} - m_i^{\text{new}}}\\
\ell_i &\leftarrow \alpha\, \ell_i + \textstyle\sum_j \tilde P_{ij}\\
O_i &\leftarrow \alpha\, O_i + \tilde P_{ij} V_j\\
m_i &\leftarrow m_i^{\text{new}}
\end{aligned}
$$

One $\alpha$ rescales both $\ell_i$ and $O_i$, since both are `exp`-weighted sums taken relative to the old max. Normalize once at the end: $O_i \leftarrow O_i / \ell_i$.

Here is the pure-PyTorch version of that loop. This is the version to *read*: it's the same algorithm as the GPU kernel we'll write later, just with explicit Python `for` loops over blocks instead of a grid of GPU programs, so you can follow every line:

```python
def flash_attention_forward(q, k, v, causal=False, sm_scale=None,
                            block_q=128, block_k=128):
    b, h, sq, d = q.shape
    sk = k.shape[2]
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(d)
    offset = sk - sq                      # bottom-right alignment for causal

    qf, kf, vf = q.float(), k.float(), v.float()
    out = torch.zeros((b, h, sq, d), dtype=torch.float32, device=q.device)
    lse = torch.zeros((b, h, sq), dtype=torch.float32, device=q.device)

    # Outer loop over query row-blocks.
    for i in range(0, sq, block_q):
        qi = qf[:, :, i:i + block_q, :]                 # (b, h, Br, d)
        br = qi.shape[2]

        # Running softmax state for this block of query rows.
        m   = torch.full((b, h, br), float("-inf"), device=q.device)
        l   = torch.zeros((b, h, br), device=q.device)
        acc = torch.zeros((b, h, br, d), device=q.device)

        # Causal block-skipping: future key blocks are entirely masked, skip them.
        k_end = min(sk, i + br + offset) if causal else sk

        # Inner loop streams over key/value column-blocks.
        for j in range(0, k_end, block_k):
            kj = kf[:, :, j:j + block_k, :]             # (b, h, Bc, d)
            vj = vf[:, :, j:j + block_k, :]
            bc = kj.shape[2]

            # The only quadratic-in-block tensor we ever materialize.
            s = torch.matmul(qi, kj.transpose(-2, -1)) * sm_scale
            if causal:
                mask = _causal_block_mask(i, j, br, bc, offset, q.device)
                s = s.masked_fill(mask, float("-inf"))

            # --- online softmax update ---
            m_new = torch.maximum(m, s.max(dim=-1).values)
            alpha = torch.exp(m - m_new)                # rescale old state
            p     = torch.exp(s - m_new.unsqueeze(-1))  # (b, h, Br, Bc)
            l   = l * alpha + p.sum(dim=-1)
            acc = acc * alpha.unsqueeze(-1) + torch.matmul(p, vj)
            m   = m_new

        # Finalize: normalize, and stash the log-sum-exp for the backward pass.
        l_safe = torch.where(l == 0, torch.ones_like(l), l)
        out[:, :, i:i + block_q, :] = acc / l_safe.unsqueeze(-1)
        lse[:, :, i:i + block_q] = m + torch.log(l_safe)

    return out.to(q.dtype), lse
```

Notice the only per-block tensor we ever build is `s`, the $B_r \times B_c$ score *tile*, never the full $N \times N$ matrix. Everything else (`m`, `l`, `acc`) is tied to the current query block and thrown away when we move on.

There's one more thing this function returns that's easy to skate past: `lse`, the **log-sum-exp** $L_i = m_i + \log \ell_i$. This is a single scalar per query row. It's the checkpoint that makes the backward pass cheap, and we'll come back to it. For now, just note that once the forward pass finishes, we keep $O$ and this thin $L$ vector, and we throw away everything quadratic.

## Why memory is now linear in N

Let's count what's actually live at any instant. We're holding one query tile $q_i$ ($B_r \times d$), one K/V tile ($B_c \times d$), the score tile ($B_r \times B_c$), and the accumulators ($O_i$ is $B_r \times d$; $m_i, \ell_i$ are length $B_r$). **None of these depend on $N$.** The block sizes $B_r, B_c$ are constants we choose.

The only things that scale with $N$ are the outputs: $O$ itself, and the $L$ vector, both $O(N)$. There is no $O(N^2)$ tensor anywhere in the algorithm. So peak memory is *linear* in sequence length.

Running the flash forward and the naive forward across growing sequence lengths and record peak allocated memory gives:

![Peak memory versus sequence length for naive attention (O(N^2), steep curve) and Flash Attention (O(N), nearly flat by comparison). Naive runs out of memory past 32K tokens; Flash keeps scaling.](/assets/images/flash-attention/bench_memory.png)

The numbers are stark. At $N = 8192$, naive attention uses about 4.2 GB while Flash uses 73 MB, a ~57× difference. At $N = 32768$ it's ~66 GB versus 266 MB, roughly 247×, and by $N = 65536$ the naive version simply OOMs on an 80 GB A100 while Flash chugs along at about 524 MB. That gap *is* the reason long-context models are trainable at all. The naive curve is a parabola; the Flash curve looks almost flat next to it because it's a straight line with a tiny slope.

## Why it's also faster (IO complexity)

Flash Attention is also a lot faster due to the reduction in HMB traffic.

Let $M$ be the SRAM size. Flash Attention loads each K/V block once per Q block, and with tile sizes chosen so a few tiles fit in SRAM (roughly $B_c = \Theta(M/d)$), the total HBM traffic works out to

$$
\Theta\!\left(\frac{N^2 d^2}{M}\right)
$$

versus $\Theta(N d + N^2)$ for the naive version, which must write and re-read the full score matrix. Since $M \gg d$, that's a big reduction in HBM accesses. And because attention is memory-bound, fewer HBM accesses translate almost directly into speedup.

To see this, I benchmarked four implementations on the same problem sizes: our `naive` reference, our pure-PyTorch `blocked` teaching kernel, our `triton` Flash Attention (coming up next), and PyTorch's own fused `scaled_dot_product_attention` (SDPA), which is the production bar to clear.

![Latency (log scale, lower is better) and achieved TFLOP/s (higher is better) versus sequence length for naive, blocked, triton, and SDPA. Triton tracks SDPA closely and both scale to long sequences; naive is far slower and stops early; the pure-PyTorch blocked version is slowest.](/assets/images/flash-attention/bench_speed.png)

A few things to read off this. First, the Triton kernel tracks PyTorch's SDPA reasonably closely (e.g. at $N = 4096$, ~113 vs ~186 TFLOP/s; the official kernel is more optimized, but we're in the same league) and both keep scaling to 16K tokens where naive has long since fallen over. Second, naive is both far slower *and* runs out of runway: I capped it at $N = 4096$ because beyond that it OOMs. Third, and this is the honest caveat: the pure-PyTorch `blocked` version is *dramatically* slower than everything (sub-1 TFLOP/s), roughly 100× slower than Triton. That's expected and important to internalize: the *algorithm* is what gives you linear memory, but the *speed* comes from actually keeping tiles in SRAM and fusing the loop into one kernel launch. Python-level blocking gets you the memory story but pays a huge per-op overhead and still round-trips through HBM. To get the speed you have to go down to the kernel.

## The backward pass (FA2 style)

Training needs gradients, and this is where the log-sum-exp checkpoint earns its keep. Given the upstream gradient $dO$, we need $dQ, dK, dV$. The naive way would, once again, reconstruct the $N \times N$ matrix $P$. Instead we **recompute** $P$ in tiles from $Q, K$ and the stored $L$, and note that we don't even need a max pass this time, because $L$ already encodes the normalization:

$$
P_{ij} = \exp\!\big(\tfrac{1}{\sqrt d} q_i K_j^\top - L_i\big).
$$

Since our computation is memory-bound and not compute-bound, rather than *store* the quadratic $P$ from the forward pass, we *recompute* it block-by-block in the backward pass, which keeps backward memory linear too. The gradients, derived from $O = PV$ and the softmax Jacobian, use a per-row scalar $D_i = \mathrm{rowsum}(dO_i \odot O_i)$:

$$
\begin{aligned}
dV_j &\mathrel{+}= P_{ij}^\top\, dO_i\\
dP_{ij} &= dO_i\, V_j^\top\\
dS_{ij} &= P_{ij} \odot (dP_{ij} - D_i) \cdot \tfrac{1}{\sqrt d}\\
dQ_i &\mathrel{+}= dS_{ij}\, K_j\\
dK_j &\mathrel{+}= dS_{ij}^\top\, q_i
\end{aligned}
$$

That $D_i$ term is what falls out of differentiating the softmax normalizer; carrying it as a precomputed per-row scalar is what lets the backward pass avoid re-deriving the normalization inside the inner loop. Here's the readable version:

```python
def flash_attention_backward(do, q, k, v, out, lse, causal=False, sm_scale=None,
                             block_q=128, block_k=128):
    b, h, sq, d = q.shape
    sk = k.shape[2]
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(d)
    offset = sk - sq

    qf, kf, vf, dof, of = (t.float() for t in (q, k, v, do, out))
    lsef = lse.float()
    dq, dk, dv = torch.zeros_like(qf), torch.zeros_like(kf), torch.zeros_like(vf)

    # D_i = sum_d dO_i * O_i, the per-row correction term.
    delta = (dof * of).sum(dim=-1)                       # (b, h, sq)

    for i in range(0, sq, block_q):
        qi  = qf[:, :, i:i + block_q, :]
        doi = dof[:, :, i:i + block_q, :]
        li  = lsef[:, :, i:i + block_q]
        di  = delta[:, :, i:i + block_q]
        br  = qi.shape[2]
        dq_i = torch.zeros_like(qi)
        k_end = min(sk, i + br + offset) if causal else sk

        for j in range(0, k_end, block_k):
            kj = kf[:, :, j:j + block_k, :]
            vj = vf[:, :, j:j + block_k, :]
            bc = kj.shape[2]

            s = torch.matmul(qi, kj.transpose(-2, -1)) * sm_scale
            if causal:
                mask = _causal_block_mask(i, j, br, bc, offset, q.device)
                s = s.masked_fill(mask, float("-inf"))

            # Recompute P directly from stored L_i (no max pass needed).
            p = torch.exp(s - li.unsqueeze(-1))          # (b, h, Br, Bc)

            dv[:, :, j:j + block_k, :] += torch.matmul(p.transpose(-2, -1), doi)
            dp = torch.matmul(doi, vj.transpose(-2, -1)) # (b, h, Br, Bc)
            ds = p * (dp - di.unsqueeze(-1)) * sm_scale  # softmax jacobian
            dq_i += torch.matmul(ds, kj)
            dk[:, :, j:j + block_k, :] += torch.matmul(ds.transpose(-2, -1), qi)

        dq[:, :, i:i + block_q, :] = dq_i

    return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype)
```

The specific choices here follow FlashAttention-2 ([Dao, 2023](#ref-fa2)): store only $L$ instead of both $m$ and $\ell$; parallelize the forward over query blocks; and split the backward into a $dK/dV$ pass (one program per key block, looping over query blocks) and a separate $dQ$ pass (one program per query block, looping over key blocks). Splitting it this way means each program owns its output and accumulates into it directly, so no atomics are needed, which matters a lot for GPU throughput.

## The real thing: a Triton kernel

Everything above runs, and it's the version to *understand*, but as the speed plot showed, pure-PyTorch blocking is slow. To actually get the SRAM residency and kernel fusion that make Flash Attention fast, we drop down to [Triton](https://triton-lang.org/), which lets us write GPU kernels in Python-ish syntax while it handles the memory coalescing and instruction scheduling.

The mapping from the blocked version is direct. Each *program* (Triton's unit of parallel work) handles one query block for one (batch, head), loads that query tile once, and streams over key/value tiles keeping the running `(m, l, acc)` in fp32 registers. Here's the forward kernel; squint and it's the same online-softmax loop as before:

```python
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(
    Q, K, V, sm_scale, L, Out,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    stride_lb, stride_lh, stride_lm,
    H, N_CTX_Q, N_CTX_K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr, CAUSAL: tl.constexpr,
):
    start_m = tl.program_id(0)        # which query row-block
    off_bh  = tl.program_id(1)        # which (batch, head)
    off_b, off_h = off_bh // H, off_bh % H

    q_base = Q + off_b * stride_qb + off_h * stride_qh
    k_base = K + off_b * stride_kb + off_h * stride_kh
    v_base = V + off_b * stride_vb + off_h * stride_vh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # Load this query block once; it stays resident for the whole inner loop.
    q_ptrs = q_base + (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk)
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX_Q, other=0.0)

    # Online-softmax running state (fp32 accumulators, live in registers).
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    offset = N_CTX_K - N_CTX_Q
    hi = tl.minimum(N_CTX_K, (start_m + 1) * BLOCK_M + offset) if CAUSAL else N_CTX_K

    for start_n in range(0, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        n_idx = start_n + offs_n

        # Load K^T tile so tl.dot gives (M, N) scores directly.
        k_ptrs = k_base + (offs_d[:, None] * stride_kk + n_idx[None, :] * stride_kn)
        k = tl.load(k_ptrs, mask=n_idx[None, :] < N_CTX_K, other=0.0)
        qk = tl.dot(q, k) * sm_scale                  # (BLOCK_M, BLOCK_N) fp32

        qk = tl.where(n_idx[None, :] < N_CTX_K, qk, float("-inf"))
        if CAUSAL:
            causal_mask = n_idx[None, :] > (offs_m[:, None] + offset)
            qk = tl.where(causal_mask, float("-inf"), qk)

        # Online softmax update (identical recurrence to the PyTorch version).
        m_ij  = tl.maximum(m_i, tl.max(qk, 1))
        p     = tl.exp(qk - m_ij[:, None])
        alpha = tl.exp(m_i - m_ij)
        l_i   = l_i * alpha + tl.sum(p, 1)

        v_ptrs = v_base + (n_idx[:, None] * stride_vn + offs_d[None, :] * stride_vk)
        v = tl.load(v_ptrs, mask=n_idx[:, None] < N_CTX_K, other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_ij

    # Finalize and write O and the log-sum-exp L back to HBM.
    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_safe[:, None]
    lse = m_i + tl.log(l_safe)

    o_base = Out + off_b * stride_ob + off_h * stride_oh
    o_ptrs = o_base + (offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok)
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < N_CTX_Q)

    l_ptrs = L + off_b * stride_lb + off_h * stride_lh + offs_m * stride_lm
    tl.store(l_ptrs, lse, mask=offs_m < N_CTX_Q)
```

The parts that feel unfamiliar coming from PyTorch are all bookkeeping, not new algorithm. Instead of nice `[:, :, i:i+block]` slices we compute raw pointers by hand using the tensor's strides (`stride_qb`, `stride_qh`, ...). `tl.load` and `tl.store` are the explicit HBM reads/writes, each with a `mask` so that sequence lengths which aren't a multiple of the block size don't read out of bounds. And `tl.dot` is the tensor-core matmul. The critical line for performance is `q = tl.load(...)` *before* the loop: the query tile is loaded once and stays in registers/SRAM while we stream all the K/V tiles past it.

The backward pass is three kernels (one to precompute $D = \mathrm{rowsum}(dO \odot O)$, one for $dK/dV$, one for $dQ$), following the same split described earlier, but they're more of the same pointer arithmetic so I'll leave them to the [repo](https://github.com/jerrickhoang/flash_attention). To make all of this usable as a normal, differentiable PyTorch op, we wrap the forward and backward kernels in a `torch.autograd.Function`:

```python
class FlashAttnFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale):
        out, lse = flash_attn_forward(q, k, v, causal=causal, sm_scale=sm_scale)
        ctx.save_for_backward(q, k, v, out, lse)   # note: we save L, not P
        ctx.causal, ctx.sm_scale = causal, sm_scale
        return out

    @staticmethod
    def backward(ctx, do):
        q, k, v, out, lse = ctx.saved_tensors
        dq, dk, dv = flash_attn_backward(
            do, q, k, v, out, lse, causal=ctx.causal, sm_scale=ctx.sm_scale)
        return dq, dk, dv, None, None

def flash_attention(q, k, v, causal=False, sm_scale=None):
    return FlashAttnFn.apply(q, k, v, causal, sm_scale)
```

The thing I want to highlight is `ctx.save_for_backward(q, k, v, out, lse)`. A normal attention autograd function would stash the $N \times N$ probabilities $P$ for the backward pass. We save the length-$N$ vector `lse` instead and recompute $P$ on the fly. That one substitution is the difference between $O(N^2)$ and $O(N)$ activation memory, and it's why Flash Attention is drop-in trainable at long context.

## Causal masking, cheaply

For causal (decoder) attention, query $i$ may only attend to keys $j \le i$. There are two things to get right, and both show up in the kernels above.

The first is masking *within* the diagonal block: entries where $j > i$ get set to $-\infty$ before the exponential, so they contribute exactly zero to the softmax. The second is more interesting for performance: **entire key blocks that lie strictly in the future of a query block are never visited at all.** That's the `hi = ... (start_m + 1) * BLOCK_M + offset` line: the inner loop simply stops early. For a full causal attention this skips roughly the upper triangle, cutting the work about in half, which is why FLOP counts for causal attention are multiplied by $0.5$. we get the causal speedup for free just by not looping over blocks we'd throw away.

One subtlety worth calling out: the mask is aligned to the **bottom-right** corner via `offset = seqlen_k - seqlen_q`, not the top-left. This matters when the query and key lengths differ, for example when decoding a suffix of a sequence where we have fewer queries than keys, so that query $i$ correctly attends to the appropriate prefix of keys.

## Choosing the tile size

The block sizes $(B_r, B_c)$ (`BLOCK_M` and `BLOCK_N` in the kernel) are the main performance knob, and the tradeoff is real. Bigger tiles feed the tensor cores larger matmuls (better utilization) but use more SRAM and registers per program, which lowers occupancy (fewer programs can be resident at once to hide memory latency). Too small and we're launching lots of tiny matmuls that never saturate the hardware; too big and you spill. So I swept the tile shape and measured throughput:

![Achieved TFLOP/s across a grid of (BLOCK_M, BLOCK_N) tile shapes. Throughput is worst for the smallest 32x32 tiles, peaks in the middle around 64x64, and dips again for the largest tiles.](/assets/images/flash-attention/block_size_sweep.png)

The smallest $32 \times 32$ tile lands around 54 TFLOP/s, the sweet spot around $64 \times 64$ hits ~125 TFLOP/s (more than 2× faster), and the largest tiles fall back toward ~90 as SRAM pressure and lower occupancy start to bite. There's no universally "right" tile size; it depends on head dimension, dtype, and the specific GPU's SRAM and register file, which is exactly why the production `flash-attn` library autotunes over a set of configurations. For this educational kernel I just picked sensible defaults from this sweep.

## Does it actually match?

None of this is worth anything if the output is wrong, and "it's exact, not an approximation" is a claim you should demand evidence for. The test suite compares both the blocked and Triton implementations against the naive reference (forward output *and* all three gradients $dQ, dK, dV$) across shapes, head dims, dtypes, causal and non-causal, and even ragged sequence lengths that aren't a multiple of the block size. It also cross-checks against PyTorch's SDPA:

```python
def test_triton_matches_reference(dtype, causal, d):
    b, h, sq, sk = 2, 3, 512, 512
    q, k, v = _make_inputs(b, h, sq, sk, d, dtype, "cuda")
    do = torch.randn_like(q)

    out = flash_attention(q, k, v, causal=causal)
    out.backward(do)
    ref_out, dq_ref, dk_ref, dv_ref = _reference_fwd_bwd(q, k, v, do, causal)

    # Low-precision tolerances: fp16/bf16 matmuls vs an fp32 reference.
    atol = 2e-2 if dtype == torch.float16 else 3e-2
    assert torch.allclose(out.float(), ref_out, atol=atol, rtol=0)
    assert torch.allclose(q.grad.float(), dq_ref, atol=atol, rtol=0)
    assert torch.allclose(k.grad.float(), dk_ref, atol=atol, rtol=0)
    assert torch.allclose(v.grad.float(), dv_ref, atol=atol, rtol=0)
```

The tolerances deserve a word, because "exact" and `atol=2e-2` look contradictory. The gap isn't algorithmic: it's that our fp16/bf16 kernel is being compared against an fp32 reference, so we're only seeing the usual low-precision matmul slack, not any approximation in the method. Run the *blocked* version in fp32 against the fp32 reference and it matches to `1e-4`. The algorithm really is exact; the only error is the one you'd get from doing any fp16 matmul.

## Wrapping up

That's Flash Attention end to end: the online softmax that lets you compute a row of softmax in a streaming pass, fusing it with the $PV$ matmul so you never materialize $P$, the tiling that makes memory linear in sequence length, the log-sum-exp checkpoint that makes the backward pass recompute-instead-of-store, a readable pure-PyTorch version to understand it, and a Triton kernel to actually make it fast. The experiments backed up every theoretical claim: linear memory (247× less than naive at 32K, and running where naive OOMs), matching PyTorch's fused SDPA on throughput, numerical stability from the max-rescaling, and a genuine tile-size sweet spot.

A few things I took away from building this that the papers state but don't quite hit home until you've written the code:

- **Same FLOPs, different memory traffic.** This is the whole game and it's worth repeating. Flash Attention doesn't do less arithmetic than naive attention. It does the *same* arithmetic while touching HBM far less. On memory-bound operations, the memory hierarchy is the algorithm.
- **The algorithm and the kernel are separable.** The blocked PyTorch version has the linear-memory property but is ~100× slower than Triton. The math gives you the memory win; only the kernel gives you the speed win. Both matter, and conflating them will confuse you.
- **The log-sum-exp is the quiet hero.** Storing one scalar per row instead of the quadratic $P$ is what makes training at long context possible, and it barely gets a sentence in most explanations.
- **Where it goes next.** The natural follow-ups are FlashAttention-3 ([Shah et al., 2024](#ref-fa3)), which exploits the Hopper architecture's asynchrony and FP8, and the various long-context variants (paged KV caches, sliding-window and block-sparse attention) that build on the same tiled foundation. Good candidates for a future post.

As always, the code is meant to be read top to bottom and run, not taken as production-grade; for real workloads use the official [`flash-attn`](https://github.com/Dao-AILab/flash-attention) package, which autotunes, supports more head dims and dtypes, and squeezes out the last bit of performance this teaching version leaves on the table. If you spot a bug, it's divine benevolence that it ran at all. See you on the next one.

## References

1. <a id="ref-fa1"></a>T. Dao, D. Y. Fu, S. Ermon, A. Rudra, and C. Ré. "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." *NeurIPS*, 2022. arXiv:2205.14135. [[link]](https://arxiv.org/abs/2205.14135). The paper this post implements; introduces the tiled, IO-aware formulation and the O(N) memory argument.
2. <a id="ref-fa2"></a>T. Dao. "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning." 2023. arXiv:2307.08691. [[link]](https://arxiv.org/abs/2307.08691). The backward-pass split (dK/dV and dQ passes), storing only the log-sum-exp, and parallelizing the forward over query blocks, all of which this implementation follows.
3. <a id="ref-online-softmax"></a>M. Milakov and N. Gimelshein. "Online normalizer calculation for softmax." 2018. arXiv:1805.02867. [[link]](https://arxiv.org/abs/1805.02867). The streaming, running-max softmax recurrence that Flash Attention is built on.
4. <a id="ref-fa3"></a>J. Shah, G. Bikshandi, Y. Zhang, V. Thakkar, P. Ramani, and T. Dao. "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision." *NeurIPS*, 2024. arXiv:2407.08608. [[link]](https://arxiv.org/abs/2407.08608). Hopper-specific asynchrony and FP8, the current state of the art.
5. <a id="ref-gpus"></a>J. Austin, S. Patil, A. Paszke, and R. Pope. "How to Think About GPUs" (Part 12 of *How to Scale Your Model*). Google DeepMind, 2025. [[link]](https://jax-ml.github.io/scaling-book/gpus/). An excellent deep dive into GPU chip architecture, the memory hierarchy, and networking; the source of the specs used in the GPU tour.
