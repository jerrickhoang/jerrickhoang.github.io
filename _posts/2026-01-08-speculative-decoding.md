---
layout: post
title: "From Scratch #2: Speculative Decoding"
date: 2026-01-08
description: "Implementing speculative decoding from scratch in PyTorch: the draft/target setup, rejection sampling, a proof that the output distribution is exactly the target's, and benchmarks across k."
category: "Machine Learning"
tags: ["llm", "inference", "speculative-decoding", "pytorch", "from-scratch"]
---

Before we begin, I love writing so none of these is written by AI, which means you might run into grammatical errors, awkwardly phrased sentences, and bad puns.

Recently I've used Cursor so much that I've realized I'm losing my ability to do ML / pytorch coding. This makes me want to find small / byte-sized topics / algorithms to code from scratch. I thought speculative decoding is the perfect example to start.

We know that for a large LLM, decoding can be quite expensive due to the autoregressive sequential nature of the task, i.e. the large model needs to run forward pass, append the results, and repeat sequentially. On the contrary, verifying the likelihood of a sequence of tokens can be done very efficiently. This asymmetry motivates the idea of speculative decoding: what if we have a cheaper way to generate good initial guesses and just have the large model verify and make corrections when needed. In many applications like document/code editing, we have a very strong prior for the speculation text which is the existing code / function being edited, in general applications, we can use a smaller model (called draft model) which runs a lot faster than the target model that can generate the speculation text. Let's dive in.

Here's the punchline I want to put up front, because it's the part that surprised me: speculative decoding is **not** an approximation. The output it produces is drawn from *exactly* the same distribution as if you'd run the target model on its own ([Leviathan et al., 2023](#ref-leviathan-2023); [Chen et al., 2023](#ref-chen-2023)). You get the speedup for free, with zero change to what the model would have said. We'll prove this later, but keep it in mind, it's the whole reason the algorithm is worth caring about.

One constraint to note before we start: the draft and target models have to **share the same tokenizer and vocabulary**. That's why gpt2 (draft) and gpt2-xl (target) pair up nicely, they're the same family. We'll see why this matters when we get to the correction step: we end up subtracting one model's probability vector from the other's, and that subtraction only makes sense if index `v` means the same token in both. (Pairing models from different families is possible but needs extra machinery, which is out of scope here.)

I have created an [educational notebook](https://github.com/jerrickhoang/pytorch_algorithms/blob/main/sampling/speculative_decoding.ipynb) to make it easier to follow along,

Before the decoding functions, here's the setup the rest of the code assumes: a couple of imports, the two models on a shared device, and a small `SpecDecodeStats` container for the benchmark numbers we'll collect later. Nothing here is specific to speculative decoding, it's just the scaffolding so the snippets below run as-is.

```python
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer    = AutoTokenizer.from_pretrained("gpt2")
draft_model  = AutoModelForCausalLM.from_pretrained("gpt2").to(device).eval()
target_model = AutoModelForCausalLM.from_pretrained("gpt2-xl").to(device).eval()


@dataclass
class SpecDecodeStats:
    """Bookkeeping for a single speculative_decode call."""
    acceptance_rates: list = field(default_factory=list)  # n_accepted / k per round
    tokens_per_pass:  list = field(default_factory=list)  # n_accepted + 1 per round
    n_target_passes:  int = 0
    wall_time:        float = 0.0
    total_tokens:     int = 0

    @property
    def tokens_per_second(self) -> float:
        return self.total_tokens / self.wall_time if self.wall_time else 0.0

    @property
    def mean_acceptance_rate(self) -> float:
        return float(np.mean(self.acceptance_rates)) if self.acceptance_rates else 0.0
```

The first step is to refresh how greedy decoding works,

```python
def greedy_decode(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int = 50,
) -> tuple[torch.Tensor, float]:
    """
    Greedy autoregressive decoding with KV cache.
    Returns (output_ids, tokens_per_second).
    """
    generated = input_ids.clone()
    past_key_values = None
    t0 = time.time()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # On first pass feed full sequence; after that feed only last token
            model_input = generated if past_key_values is None else generated[:, -1:]

            outputs = model(
                model_input,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values

            # logits: [1, seq_len, vocab] — only last position matters
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [1, 1]
            generated = torch.cat([generated, next_token], dim=-1)

    elapsed = time.time() - t0
    tps = max_new_tokens / elapsed
    return generated, tps
```

The logic is very simple: for each of the new tokens, we just call the model forward function and append the results to a running tensor of tokens. Here we leverage HF interface for KV caching for this greedy baseline (we could have implemented it without KV caching as well).

When tried out for gpt2 for the draft model and gpt2-xl for the target model, greedy sampling gives,

```
The capital of France is the capital of the French Republic, and the capital of the French Republic is the capital of the French
Draft model: 107.4 tok/s
The capital of France is the city of Paris. It is the capital of France and the largest city in France. It is
Target model: 27.2 tok/s
```

## Core Algorithm

Let's get to the core of the algorithm,

```
    Full speculative decoding loop.

    Each iteration:
      1. Draft generates k candidates             (k draft forward passes)
      2. Target scores all k in one pass          (1 target forward pass)
      3. Rejection sampling → n accepted + 1 correction
      4. Append (n+1) tokens to generated
      5. Repeat
```

The key efficiency gain here is that: step 2 is guaranteed to perform only ONE target forward pass regardless of k, and it produces between 1 and k+1 tokens. Compared to baseline which requires one target pass per token.

With this in mind, let's try implementing each step of this process.

### Step 1: Draft generates k candidates

This is the same code as the greedy decode above except that we also need to keep track of the probabilities for each of the predicted token for the rejection sampling step

```python
def draft_autoregressive(
    draft_model,
    input_ids: torch.Tensor,   # [1, seq_len]
    k: int,
    temperature: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run draft model autoregressively to produce k candidate tokens.

    Args:
      temperature: 0.0 for greedy (argmax), > 0 for sampling.

    Returns:
      draft_tokens: [k]        integer token ids
      draft_probs:  [k, vocab] probability distribution at each step

    We save the full softmax distribution (not just argmax) because
    the rejection step needs p_draft(xᵢ) for each accepted token.
    """
    draft_tokens = []
    draft_probs = []
    past_key_values = None
    current_ids = input_ids.clone()

    with torch.no_grad():
        for _ in range(k):
            model_input = current_ids if past_key_values is None else current_ids[:, -1:]

            outputs = draft_model(
                model_input,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values

            logits = outputs.logits[:, -1, :]        # [1, vocab]

            if temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)[0]
                next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                probs = F.softmax(logits, dim=-1)[0]     # [vocab]
                next_token = probs.argmax()               # scalar tensor

            draft_tokens.append(next_token)
            draft_probs.append(probs)

            current_ids = torch.cat(
                [current_ids, next_token.reshape(1, 1)], dim=-1
            )

    draft_tokens = torch.stack(draft_tokens)   # [k]
    draft_probs  = torch.stack(draft_probs)    # [k, vocab]
    return draft_tokens, draft_probs
```

Note that the key difference between this and the previous greedy sampler is that we take in a temperature parameter. If the temperature is 0, it is effectively the previous function (greedy), otherwise, we sample from the distribution instead.

### Step 2: Target scores all k in one pass

```python
def target_score(
    target_model,
    input_ids: torch.Tensor,      # [1, prefix_len]
    draft_tokens: torch.Tensor,   # [k]
    temperature: float = 0.0,
) -> torch.Tensor:
    """
    Run target model ONCE over (context + draft_tokens), return
    the target's probability distributions at each draft position.

    Returns:
      target_probs: [k+1, vocab]

    Indexing:
      Input:   [t₀ ... t_{n-1}  x₁  x₂ ... xₖ]
      Indices:  0  ...  n-1     n  n+1 ...  n+k-1

      The logit at position i gives the distribution OVER token i+1.
      So target_probs[0] = p_target(· | t₀..t_{n-1})  — used to judge x₁
         target_probs[1] = p_target(· | t₀..x₁)       — used to judge x₂
         ...
         target_probs[k] = p_target(· | t₀..xₖ)       — bonus token

      That slice is logits[:, n-1 : n+k, :] which has shape [1, k+1, vocab].
    """
    prefix_len = input_ids.shape[1]

    # Concatenate context with draft tokens
    draft_ids = draft_tokens.unsqueeze(0)                         # [1, k]
    full_input = torch.cat([input_ids, draft_ids], dim=-1)        # [1, prefix_len + k]

    with torch.no_grad():
        outputs = target_model(full_input, use_cache=False)

    # Slice the k+1 relevant logit positions
    # logits shape: [1, prefix_len + k, vocab]
    # We want positions [prefix_len-1 : prefix_len+k]  (k+1 positions)
    relevant_logits = outputs.logits[0, prefix_len - 1 : prefix_len + len(draft_tokens), :]  # [k+1, vocab]

    # Apply the same temperature the draft used, so target and draft are
    # describing the SAME distribution. For greedy (T=0) the comparison is
    # argmax-based and the scaling is irrelevant, so we skip it.
    if temperature > 0:
        relevant_logits = relevant_logits / temperature

    target_probs = F.softmax(relevant_logits, dim=-1)             # [k+1, vocab]
    return target_probs
```

We can see that verification for the target model is very cheap. Since the whole sequence from input and draft is fixed, one single forward pass. The important thing to note here is that the logits at index $i$ predict the probability distribution for the next token, which is at index $i + 1$. And so we slice the logits from `prefix_len - 1` instead of `prefix_len`.

One subtlety that's easy to get wrong: `target_score` takes the **same** `temperature` as the draft model, and applies it before the softmax. This matters for the stochastic path. The acceptance test compares $p_{\text{target}}(x_i)$ against $p_{\text{draft}}(x_i)$, and that comparison is only valid if both probabilities are measured under the same temperature. If the draft sampled at $T = 0.8$ but the target scored at $T = 1.0$, the ratio $p_{\text{target}} / p_{\text{draft}}$ would be systematically skewed and the distribution-matching guarantee would quietly break. For greedy decoding ($T = 0$) it doesn't matter, since we only compare argmaxes and temperature doesn't move the argmax.

There's also a performance note worth being honest about. This educational version re-runs the target over the *entire* prefix every round (`use_cache=False`), which throws away work: the prefix didn't change, only the newly appended tokens did. A production implementation would keep a KV cache for the target's prefix too and only feed the `k` new draft tokens each round. I've left it uncached here to keep the indexing readable, but on long sequences that re-processing is the first thing you'd optimize.

### Step 3: Rejection sampling → n accepted + 1 correction

```python
def rejection_sample(
    draft_tokens: torch.Tensor,  # [k]
    draft_probs:  torch.Tensor,  # [k, vocab]
    target_probs: torch.Tensor,  # [k+1, vocab]
    greedy: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Accept/reject each draft token, with two modes:

    greedy=True  (temperature=0):
      Accept xᵢ iff xᵢ == argmax(target_probs[i]).
      Correction / bonus is argmax of the target distribution.
      Guarantees identical output to greedy target decoding.

    greedy=False (temperature>0, true rejection sampling):
      Accept xᵢ with probability min(1, p_target(xᵢ) / p_draft(xᵢ)).
      On rejection, sample correction from normalize(max(0, p_target - p_draft)).
      Bonus token sampled from target_probs[k].
      Guarantees the output distribution is exactly p_target.

    Returns:
      accepted:   [n_accepted]  (0 <= n_accepted <= k)
      correction: [1]
    """
    accepted = []

    for i in range(len(draft_tokens)):
        if greedy:
            target_token = target_probs[i].argmax()
            if draft_tokens[i] == target_token:
                accepted.append(draft_tokens[i])
            else:
                accepted_tokens = (
                    torch.stack(accepted)
                    if accepted
                    else torch.tensor([], dtype=torch.long, device=draft_tokens.device)
                )
                return accepted_tokens, target_token.unsqueeze(0)
        else:
            token = draft_tokens[i].item()
            p_t = target_probs[i, token].item()
            p_d = draft_probs[i, token].item()

            accept_prob = min(1.0, p_t / (p_d + 1e-10))
            r = torch.rand(1).item()

            if r < accept_prob:
                accepted.append(draft_tokens[i])
            else:
                residual = torch.clamp(target_probs[i] - draft_probs[i], min=0.0)
                residual_sum = residual.sum()

                if residual_sum < 1e-8:
                    correction = torch.multinomial(target_probs[i], num_samples=1)
                else:
                    p_corrected = residual / residual_sum
                    correction = torch.multinomial(p_corrected, num_samples=1)

                accepted_tokens = (
                    torch.stack(accepted)
                    if accepted
                    else torch.tensor([], dtype=torch.long, device=draft_tokens.device)
                )
                return accepted_tokens, correction

    if greedy:
        bonus_token = target_probs[len(draft_tokens)].argmax().unsqueeze(0)
    else:
        bonus_token = torch.multinomial(target_probs[len(draft_tokens)], num_samples=1)

    return torch.stack(accepted), bonus_token
```

This is the most important bit of the algorithm. The greedy path is straightforward, we accept the token if the target model also agrees with the draft model, otherwise the correction is sampled directly from the target distribution. This means we sample at most once to generate 1 to k + 1 tokens. By construction, the greedy path is guaranteed to produce *exactly* the output that greedy decoding on the target model alone would produce, token for token. Note the subtlety: greedy decoding doesn't sample, it takes the argmax at each step, so the guarantee here is bit-exact equality with greedy target decoding, not distributional equality. (The distributional guarantee, matching samples from the target, is what the stochastic path below buys you.)

Before we get into the stochastic path, one quick notation cleanup so the math lines up with the code. I'll write $p$ for the **target** distribution (what we want to sample from) and $q$ for the **draft** distribution (the cheap proposal). In the code these are `target_probs` / `p_t` and `draft_probs` / `p_d` respectively. So "accept with probability $\min(1, p/q)$" and "accept with probability `min(1, p_t / p_d)`" are the same statement.

For the stochastic path, intuitively:

- If $p > q$ : The target model assigns more probability to this token than the draft model. In this case, the acceptance probability is 1, so we always accept the token. This reflects that the draft model is underestimating this token, and the target model fully "allows" it.
- If $p < q$ : The draft model overestimates this token relative to the target. We therefore accept it only with probability $p / q$, to avoid assigning too much probability mass to this token.

A tiny concrete example makes this click. Say the next token is "Paris" and:

- Target says $p(\text{Paris}) = 0.4$, draft said $q(\text{Paris}) = 0.25$. Since $p > q$, we accept with probability $\min(1, 0.4/0.25) = 1$. Always keep it. The draft got lucky and under-sold a token the target likes.
- Now flip it: target says $p(\text{Paris}) = 0.25$, draft said $q(\text{Paris}) = 0.4$. Since $p < q$, we accept with probability $\min(1, 0.25/0.4) = 0.625$. So roughly 5 times out of 8 we keep "Paris," and the other 3 times we reject and sample a correction. The draft over-proposed it, so we let some of that mass leak back out.

When a token is rejected, we must sample a replacement. However, we cannot sample directly from the target distribution $p$, because part of $p$'s mass has already been accounted for by the accepted draft tokens. Instead, we sample from the residual distribution. This residual represents the probability mass that the draft model failed to cover, i.e. where the target assigns more probability than the draft. If the residual is large for some tokens, it means the draft model significantly underestimates them, so we must sample more often from these regions to compensate. If the residual is small, it means the draft model already matches the target well in those regions, so little or no correction is needed.

Mathematically, the key identity is that the target distribution decomposes cleanly into an "accepted" part and a "residual" part:

$$p(v) = \min(q(v), p(v)) + \big(p(v) - q(v)\big)_{+}$$

where $$(x)_+ = \max(0, x)$$ is the positive part (this is exactly the `torch.clamp(..., min=0.0)` in the code). The first term, $\min(q, p)$, is the mass the draft and target agree on, the part we can accept. The second term, $$(p - q)_+$$, is the mass the draft missed, the part we have to correct for.

That second term is *not* a probability distribution on its own, it doesn't sum to 1. So the thing we actually sample from on rejection is the **normalized** residual:

$$p_{\text{res}}(v) = \frac{\big(p(v) - q(v)\big)_+}{\sum_{v'} \big(p(v') - q(v')\big)_+}$$

which maps directly onto `residual / residual_sum` in the code.

### Why this gives you the target distribution exactly

The post keeps asserting that the output is distributed exactly as $p$. Let's actually show it, because it's short and it's the whole point. (The modified rejection sampling argument here follows [Chen et al., 2023](#ref-chen-2023).) Consider a single draft token. The draft proposed some token $v$ from $q$, and we want to show that the token we end up emitting (after accept/reject + correction) is distributed as $p$.

Fix a token $v$ and ask: what's the total probability that we emit $v$? There are two disjoint ways it can happen.

**Path 1, we emit $v$ because the draft proposed it and we accepted it.** The draft proposes $v$ with probability $q(v)$, and we accept with probability $\min(1, p(v)/q(v))$. Multiply:

$$q(v) \cdot \min\!\left(1, \frac{p(v)}{q(v)}\right) = \min\big(q(v),\, p(v)\big)$$

**Path 2, we emit $v$ because some proposal was rejected and the correction landed on $v$.** First, the overall probability that *any* rejection happens is

$$\beta = \sum_{v'} q(v')\Big(1 - \min\big(1, \tfrac{p(v')}{q(v')}\big)\Big) = \sum_{v'} \big(q(v') - \min(q(v'), p(v'))\big) = \sum_{v'} \big(q(v') - p(v')\big)_+$$

The middle step pulls $q$ inside; the last uses $$q - \min(q, p) = (q - p)_+$$ pointwise. Now the small trick: because both $p$ and $q$ sum to 1, the total "excess" mass on the draft side equals the total "deficit," i.e. $$\sum_{v'} (q(v') - p(v'))_+ = \sum_{v'} (p(v') - q(v'))_+$$. So $\beta$ is exactly the residual normalizer from before. Given a rejection, we sample $v$ from the normalized residual $$p_{\text{res}}(v) = (p(v)-q(v))_+ / \beta$$. So Path 2 contributes $$\beta \cdot p_{\text{res}}(v) = (p(v) - q(v))_+$$.

Add the two paths:

$$\Pr[\text{emit } v] = \min\big(q(v), p(v)\big) + \big(p(v) - q(v)\big)_+ = p(v)$$

by the decomposition identity above. So a single draft token, run through accept/reject, emits exactly from $p$. That's the guarantee. Chaining it across the $k$ positions (each conditioned on the accepted prefix) extends it to the whole block, and the greedy path is just the $T \to 0$ limit of the same argument.

### The bonus token

One more piece that's easy to miss. Look at what happens when *all* $k$ draft tokens get accepted. The target's single forward pass scored positions for $x_1 \dots x_k$, but because of how the indexing works, it *also* produced a distribution at the position right after $x_k$, `target_probs[k]`. That's a completely free next-token distribution we already paid for. So when the whole draft is accepted, we get to sample one extra token, the **bonus token**, straight from the target. This is why a single target pass can yield anywhere from 1 token (everything rejected at the first position, plus a correction) up to $k+1$ tokens (all $k$ accepted, plus the bonus). The "+1" in "1 to k+1" is this bonus.

### Step 4: Full speculative decoding loop

```python
def speculative_decode(
    draft_model,
    target_model,
    input_ids: torch.Tensor,
    k: int = 5,
    max_new_tokens: int = 50,
    temperature: float = 0.0,
) -> tuple[torch.Tensor, SpecDecodeStats]:
    """
    Full speculative decoding loop.

    Args:
      temperature: 0.0 for greedy decoding, > 0 for sampling.
        - greedy: draft uses argmax, rejection uses argmax comparison.
          Output is identical to greedy target decoding.
        - sampling: draft samples from softmax(logits/T), rejection uses
          stochastic accept/reject. Output distribution matches target.

    Each iteration:
      1. Draft generates k candidates             (k draft forward passes)
      2. Target scores all k in one pass          (1 target forward pass)
      3. Rejection sampling → n accepted + 1 correction
      4. Append (n+1) tokens to generated
      5. Repeat
    """
    greedy = (temperature == 0.0)
    generated = input_ids.clone()
    stats = SpecDecodeStats()
    t0 = time.time()
    initial_len = input_ids.shape[1]

    while (generated.shape[1] - initial_len) < max_new_tokens:
        draft_tokens, draft_probs = draft_autoregressive(
            draft_model, generated, k, temperature=temperature,
        )

        t_probs = target_score(target_model, generated, draft_tokens, temperature=temperature)

        accepted, correction = rejection_sample(
            draft_tokens, draft_probs, t_probs, greedy=greedy,
        )

        new_tokens = torch.cat([accepted, correction])
        generated = torch.cat([generated, new_tokens.unsqueeze(0)], dim=-1)

        n_accepted = len(accepted)
        stats.acceptance_rates.append(n_accepted / k)
        stats.tokens_per_pass.append(n_accepted + 1)
        stats.n_target_passes += 1

    generated = generated[:, :initial_len + max_new_tokens]
    stats.wall_time = time.time() - t0
    stats.total_tokens = generated.shape[1] - initial_len
    return generated, stats
```

The core logic has been written prior to this function. Here we just continuously speculate ahead k tokens, and keep the ones that are accepted and the correction token, we do this until max_new_tokens is reached.

## Results

With temperature == 0, the greedy sampling from the target must match bit-by-bit with speculative decoding by construction,

```python
def verify_greedy_correctness(
    draft_model,
    target_model,
    tokenizer,
    prompts: list[str],
    k: int = 5,
    max_new_tokens: int = 20,
) -> dict:
    """
    For each prompt, compare greedy output (temperature=0) from:
      (a) pure target model — ground truth
      (b) speculative decoding

    With greedy decoding both MUST be identical.
    Any mismatch indicates a bug in target_score indexing
    or the rejection_sample correction distribution.
    """
    results = {}
    for prompt in prompts:
        ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        prefix_len = ids.shape[1]

        target_out, _ = greedy_decode(target_model, ids, max_new_tokens)
        target_text = tokenizer.decode(target_out[0][prefix_len:])

        spec_out, _ = speculative_decode(
            draft_model, target_model, ids, k, max_new_tokens, temperature=0.0,
        )
        spec_text = tokenizer.decode(spec_out[0][prefix_len:])

        results[prompt] = {
            "target":      target_text,
            "speculative": spec_text,
            "match":       target_text == spec_text,
        }
    return results
```

```
✓  'The capital of France is'
✓  'In machine learning, gradient descent'
✓  'The transformer architecture was introduced'
✓  'Python is a programming language that'

Match rate: 100%  (must be 100% for greedy)
```

With temperature > 0, speculative decoding uses stochastic rejection sampling. Outputs are no longer deterministic, so we can't compare strings. Instead we verify that the token-level distribution of speculative decoding matches the target model's distribution by:

- Computing the analytical target distribution via a single forward pass (softmax(logits / T)). This is exact, no sampling needed.
- Running n_samples speculative decoding generations and collecting the empirical distribution of the first generated token.
- Computing the Total Variation (TV) distance between analytical and empirical.
- TV ≈ 0 means the distributions match and the rejection sampler is correct.

```python
def analytical_target_distribution(
    target_model,
    input_ids: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Compute the exact next-token distribution from the target model
    in a single forward pass. No sampling needed — this is the ground truth.
    """
    with torch.no_grad():
        logits = target_model(input_ids).logits[:, -1, :]  # [1, vocab]
    return F.softmax(logits / temperature, dim=-1)[0]       # [vocab]


def verify_sampling_distribution(
    draft_model,
    target_model,
    tokenizer,
    prompt: str,
    temperature: float = 1.0,
    k: int = 5,
    n_samples: int = 500,
) -> dict:
    """
    Compare the empirical first-token distribution from speculative decoding
    against the ANALYTICAL target distribution (exact, computed in one forward pass).

    Using the analytical target eliminates sampling noise on the reference side,
    so the TV distance reflects only spec-decode variance and any algorithmic bias.
    """
    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    prefix_len = ids.shape[1]
    vocab_size = target_model.config.vocab_size

    target_dist = analytical_target_distribution(target_model, ids, temperature)

    spec_counts = torch.zeros(vocab_size)
    for _ in range(n_samples):
        s_out, _ = speculative_decode(
            draft_model, target_model, ids, k, 1, temperature,
        )
        first_token = s_out[0, prefix_len].item()
        spec_counts[first_token] += 1

    spec_dist = spec_counts / spec_counts.sum()
    tv_distance = 0.5 * (target_dist.cpu() - spec_dist).abs().sum().item()

    nonzero = (target_dist.cpu() > 1e-4) | (spec_counts > 0)
    token_ids = nonzero.nonzero(as_tuple=True)[0].tolist()

    top_tokens = sorted(
        token_ids,
        key=lambda t: max(target_dist[t].item(), spec_dist[t].item()),
        reverse=True,
    )[:10]

    return {
        "tv_distance": tv_distance,
        "target_dist": target_dist.cpu(),
        "spec_dist": spec_dist,
        "top_tokens": top_tokens,
        "n_samples": n_samples,
    }
```

![Target (analytical) vs speculative (empirical) first-token distributions, with TV distance reported per prompt.](/assets/images/speculative-decoding/image2.png)

A quick word on reading these numbers, because "TV ≈ 0 means correct" can be misleading. The "capital of France" prompt reports TV = 0.1774, which is clearly not near zero, and that looks alarming until you remember where the noise comes from. We're estimating the empirical distribution from only **500 samples** spread across a large vocabulary, so each per-token frequency has real sampling error baked in, and TV distance sums all of those errors. The "gradient descent" prompt lands at TV = 0.0873, much lower, precisely because its distribution is sharply peaked on one token ("is"), so 500 samples pin it down far more tightly. The takeaway isn't "the sampler is 0.18 wrong," it's that the gap is dominated by Monte Carlo noise from a small sample budget, not by algorithmic bias. Crank `n_samples` up and both numbers shrink toward zero. (If you want to convince yourself the sampler is unbiased rather than just close, the greedy bit-exact check above is the cleaner test, since it has no sampling noise at all.)

We can also compute the actual speedups and acceptance rates across k values,

```python
def benchmark(
    draft_model,
    target_model,
    tokenizer,
    prompt: str,
    k_values: list[int] = [1, 3, 5, 8, 10],
    max_new_tokens: int = 100,
    n_runs: int = 3,
) -> dict:
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    results = {"baseline": []}

    # Baseline: pure target model
    for _ in range(n_runs):
        _, tps = greedy_decode(target_model, input_ids, max_new_tokens)
        results["baseline"].append(tps)
    results["baseline_median"] = float(np.median(results["baseline"]))

    # Speculative at each k
    results["speculative"] = {}
    for k in k_values:
        tps_runs, alpha_runs = [], []
        for _ in range(n_runs):
            _, stats = speculative_decode(draft_model, target_model, input_ids, k, max_new_tokens)
            tps_runs.append(stats.tokens_per_second)
            alpha_runs.append(stats.mean_acceptance_rate)
        results["speculative"][k] = {
            "tps_median":  float(np.median(tps_runs)),
            "speedup":     float(np.median(tps_runs)) / results["baseline_median"],
            "mean_alpha":  float(np.mean(alpha_runs)),
        }

    return results
```

![Speedup vs k and mean acceptance rate vs k. Measured speedup peaks around k = 3–5 before falling off, while the mean acceptance rate decreases as k grows.](/assets/images/speculative-decoding/image1.png)

There's a dashed "theory ($\alpha = 0.49$)" curve in that left plot, and it's worth spelling out where it comes from, since the post would otherwise leave you staring at a line with no formula. Let $\alpha$ be the per-token acceptance rate (the probability the target accepts a given draft token). If acceptances were independent, the expected number of tokens you commit per target pass is a geometric-style sum ([Leviathan et al., 2023](#ref-leviathan-2023)):

$$\mathbb{E}[\text{tokens per pass}] = \frac{1 - \alpha^{k+1}}{1 - \alpha}$$

The intuition: you accept the first token with prob $\alpha$, the first two with prob $\alpha^2$, and so on, and the "+1" inside the exponent is the bonus token you collect when all $k$ are accepted. Each target pass costs roughly one target forward (the draft passes are cheap by assumption), so dividing that expected token count by the cost gives the theoretical speedup. That's the dashed curve, and it rises monotonically toward $\frac{1}{1-\alpha}$ as $k \to \infty$.

But the measured (solid) curve does **not** rise monotonically, it peaks around $k = 3$–$5$ and then *falls*. The two plots together explain why, and this is the most important practical lesson in the whole post:

- **The right plot shows acceptance rate $\alpha$ dropping as $k$ grows.** This is the part the clean formula ignores. Draft token $x_i$ is conditioned on the earlier *speculative* tokens $x_1 \dots x_{i-1}$, not on verified ones. The further the draft runs ahead, the more it's building on its own guesses, so it drifts away from what the target would have done, and later tokens get rejected more often. Acceptances aren't actually independent, and they decay with depth.
- **So large $k$ wastes draft compute.** Past the sweet spot, you're paying for extra draft forward passes that mostly produce tokens the target throws away. The theoretical curve assumes a fixed $\alpha$ and never sees this; reality pays the draft cost up front and only reaps accepted tokens.

The lesson: $k$ is a tuning knob with a real optimum, not a "bigger is better" dial. For this gpt2 / gpt2-xl pair the sweet spot is around $k = 3$–$5$, and the right value depends entirely on how well your draft tracks your target (a better draft pushes the optimum higher).

So there we have it: Speculative Decoding, implemented from scratch. We proved that whether you're using the simple greedy approach or the complex stochastic rejection sampling, the math holds up. The key innovation is trading multiple expensive target forward passes for one single, cheap verification and correction step.

## References and further reading

If you want the original treatment and the formal proofs, the two papers that introduced this are [Leviathan et al., 2023](#ref-leviathan-2023) and [Chen et al., 2023](#ref-chen-2023). For where the field went next, the EAGLE line ([Li et al., 2024](#ref-eagle-2024); cheaper, learned drafts) and tree-based verification ([Miao et al., 2023](#ref-specinfer-2023); [Cai et al., 2024](#ref-medusa-2024)) are the natural follow-ups once the basics here make sense.

1. <a id="ref-leviathan-2023"></a>Y. Leviathan, M. Kalman, and Y. Matias. "Fast Inference from Transformers via Speculative Decoding." *ICML*, 2023. arXiv:2211.17192. [[link]](https://arxiv.org/abs/2211.17192) — Names the algorithm and proves the output distribution is unchanged; the expected-tokens-per-pass formula used above comes from here.
2. <a id="ref-chen-2023"></a>C. Chen, S. Borgeaud, G. Irving, J.-B. Lespiau, L. Sifre, and J. Jumper. "Accelerating Large Language Model Decoding with Speculative Sampling." 2023. arXiv:2302.01318. [[link]](https://arxiv.org/abs/2302.01318) — Independent, concurrent work from DeepMind with the same core idea and a clean derivation of the modified rejection sampling step.
3. <a id="ref-eagle-2024"></a>Y. Li, F. Wei, C. Zhang, and H. Zhang. "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty." *ICML*, 2024. arXiv:2401.15077. [[link]](https://arxiv.org/abs/2401.15077)
4. <a id="ref-specinfer-2023"></a>X. Miao et al. "SpecInfer: Accelerating Generative LLM Serving with Speculative Inference and Token Tree Verification." 2023. arXiv:2305.09781. [[link]](https://arxiv.org/abs/2305.09781)
5. <a id="ref-medusa-2024"></a>T. Cai, Y. Li, Z. Geng, H. Peng, J. D. Lee, D. Chen, and T. Dao. "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads." 2024. arXiv:2401.10774. [[link]](https://arxiv.org/abs/2401.10774)

See you on the next blog post!
