---
layout: post
title: "Implementing DeepSeek Series"
date: 2025-01-27
author: Jerrick Hoang
description: "A from-scratch implementation walkthrough of the DeepSeek model series (V1, V2, V3): RoPE, GQA, RMSNorm, SwiGLU, DeepSeek MoE, Multi-Latent Attention, YaRN, GRPO, and multi-token prediction."
category: "Machine Learning"
tags: ["llm", "deepseek", "transformers", "mixture-of-experts", "attention", "from-scratch"]
---

This document assumes that readers are familiar with vanilla GPT2/3 architectures and somewhat familiar with LLAMA2 family of models. We will build on top of these architectures.

# DeepSeekV1

Paper: *DeepSeek LLM: Scaling Open-Source Language Models with Longtermism* ([DeepSeek-AI, 2024a](#ref-deepseek-v1))

The following paragraph tells us the architecture of DeepSeekV1

> The micro design of DeepSeek LLM largely follows the design of LLaMA ([Touvron et al., 2023a,b](#ref-llama)), adopting a Pre-Norm structure with RMSNorm ([Zhang and Sennrich, 2019](#ref-rmsnorm)) function and using SwiGLU ([Shazeer, 2020](#ref-swiglu)) as the activation function for the Feed-Forward Network (FFN), with an intermediate layer dimension of $\tfrac{8}{3} d_{model}$. It also incorporates Rotary Embedding ([Su et al., 2024](#ref-rope)) for positional encoding. To optimize inference cost, the 67B model uses GroupedQuery Attention (GQA) ([Ainslie et al., 2023](#ref-gqa)) instead of the traditional Multi-Head Attention (MHA). However, in terms of macro design, DeepSeek LLM differs slightly. Specifically, DeepSeek LLM 7B is a 30-layer network, while DeepSeek LLM 67B has 95 layers. These layer adjustments, while maintaining parameter consistency with other open-source models, also facilitate model pipeline partitioning to optimize training and inference.

So to summarize,

- It uses LLaMA’s architecture which has the following differences compared to the GPT2,3 models,
- Uses RoPE instead of absolute PE
- Uses SwiGLU instead of GELU(approx=’tanh’)
- Use RMSNorm as the pre-norm structure instead of LayerNorm
- The MLP projection multiplier is 8/3 instead of 4
- The larger model uses GroupedQueryAttention (GQA)
- The 67B model uses GQA while the 7B doesn’t. We will implement both versions.

As a refresher, this is the summary of the architecture differences of LLaMA (which DeepSeekV1 uses) as compared to the original attention is all you need paper (Note that the picture on the left is not quite GPT which is 1) a decoder only model so no cross attention with the encoder and 2) uses Pre-norm formulation instead of post-norm)

![Summary of LLaMA vs GPT-2/3 architecture differences.](/assets/images/deepseek-base/image22.png)

## Review the concepts

### RoPE

[https://arxiv.org/pdf/2104.09864](https://arxiv.org/pdf/2104.09864)

Before going any further, it is important to revisit GPT code and remind ourselves what are positional embeddings and why we need them. Let’s look at the original code from OpenAI (unfortunately it’s in Tensorflow), we notice that the inputs to the attention blocks is the sum of wpe (weighted positional embeddings) and wte (weighted token embeddings). The token embeddings identify the individual tokens. However, the attention blocks operate on these individual tokens independently, therefore we need something that encodes the relative positions of the tokens. We do this by simply encodes the absolute positions of the tokens (from 0 to past_length)

```python
        wpe = tf.get_variable('wpe', [hparams.n_ctx, hparams.n_embd],
                             initializer=tf.random_normal_initializer(stddev=0.01))
        wte = tf.get_variable('wte', [hparams.n_vocab, hparams.n_embd],
                             initializer=tf.random_normal_initializer(stddev=0.02))
        past_length = 0 if past is None else tf.shape(past)[-2]
        h = tf.gather(wte, X) + tf.gather(wpe, positions_for(X, past_length))

def positions_for(tokens, past_length):
    batch_size = tf.shape(tokens)[0]
    nsteps = tf.shape(tokens)[1]
    return expand_tile(past_length + tf.range(nsteps), batch_size)
```

However, this is kind of an awkward way to encode relative position. Since the start of the context is pretty arbitrary in each training batch, the relative position between tokens are a lot more important. The authors in the original paper argue that it is sufficient to find the function $g$ such that $$\langle f_q(x_m, m),\, f_k(x_n, n) \rangle = g(x_m, x_n, m-n)$$. In other words, in the attention forward pass, when we take the inner product between the keys and the queries, it should result in a function g that depends solely on the token embeddings ($x_m$, $x_n$) and the relative position $m - n$. The nice thing is that if we setup this as a functional equation with initial condition then it can be solved to obtain the following solutions (the paper goes into details for the proof),

$$\begin{aligned}
f_q(\mathbf{x}_m, m) &= (W_q \mathbf{x}_m)\, e^{im\theta} \\
f_k(\mathbf{x}_n, n) &= (W_k \mathbf{x}_n)\, e^{in\theta} \\
g(\mathbf{x}_m, \mathbf{x}_n, m-n) &= \mathrm{Re}\!\left[(W_q \mathbf{x}_m)(W_k \mathbf{x}_n)^{*}\, e^{i(m-n)\theta}\right]
\end{aligned}$$

Using Euler’s formula, it can be rewritten as,

$$f_{\{q,k\}}(\mathbf{x}_m, m) =
\begin{pmatrix} \cos m\theta & -\sin m\theta \\ \sin m\theta & \cos m\theta \end{pmatrix}
\begin{pmatrix} W^{(11)}_{\{q,k\}} & W^{(12)}_{\{q,k\}} \\ W^{(21)}_{\{q,k\}} & W^{(22)}_{\{q,k\}} \end{pmatrix}
\begin{pmatrix} x^{(1)}_m \\ x^{(2)}_m \end{pmatrix}$$

Using linearity of inner product, we can generalize to d-dimension by breaking it up to groups of two dimensions,

$$f_{\{q,k\}}(\mathbf{x}_m, m) = R^{d}_{\Theta, m}\, W_{\{q,k\}}\, \mathbf{x}_m \tag{14}$$

where

$$R^{d}_{\Theta, m} =
\begin{pmatrix}
\cos m\theta_1 & -\sin m\theta_1 & 0 & 0 & \cdots & 0 & 0 \\
\sin m\theta_1 & \cos m\theta_1 & 0 & 0 & \cdots & 0 & 0 \\
0 & 0 & \cos m\theta_2 & -\sin m\theta_2 & \cdots & 0 & 0 \\
0 & 0 & \sin m\theta_2 & \cos m\theta_2 & \cdots & 0 & 0 \\
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
0 & 0 & 0 & 0 & \cdots & \cos m\theta_{d/2} & -\sin m\theta_{d/2} \\
0 & 0 & 0 & 0 & \cdots & \sin m\theta_{d/2} & \cos m\theta_{d/2}
\end{pmatrix} \tag{15}$$

Which can be written more efficiently as below,

$$R^{d}_{\Theta, m}\, \mathbf{x} =
\begin{pmatrix} x_1 \\ x_2 \\ x_3 \\ x_4 \\ \vdots \\ x_{d-1} \\ x_d \end{pmatrix}
\otimes
\begin{pmatrix} \cos m\theta_1 \\ \cos m\theta_1 \\ \cos m\theta_2 \\ \cos m\theta_2 \\ \vdots \\ \cos m\theta_{d/2} \\ \cos m\theta_{d/2} \end{pmatrix}
+
\begin{pmatrix} -x_2 \\ x_1 \\ -x_4 \\ x_3 \\ \vdots \\ -x_d \\ x_{d-1} \end{pmatrix}
\otimes
\begin{pmatrix} \sin m\theta_1 \\ \sin m\theta_1 \\ \sin m\theta_2 \\ \sin m\theta_2 \\ \vdots \\ \sin m\theta_{d/2} \\ \sin m\theta_{d/2} \end{pmatrix}$$

The solution of the functional equation works with any predefined $\theta_i$. In the paper, they set

$$\Theta = \left\{ \theta_i = 10000^{-2(i-1)/d},\ i \in [1, 2, \dots, d/2] \right\}$$

A quick note on indexing, because it’s an easy place to introduce an off-by-one: here $i$ is 1-indexed and runs over the $d/2$ frequency pairs, so $\theta_1 = 1$ (the fastest, no-decay frequency) and $\theta_{d/2}$ is the slowest. If you implement this with a 0-indexed loop instead (as most code does), the exponent becomes $$10000^{-2i/d}$$ for $$i \in [0, d/2)$$, which is the same set of frequencies. I’ll keep the 1-indexed form throughout, including in the YaRN section below, so watch the convention if you’re cross-checking against an implementation.

To achieve long-term decay (i.e. the inner product decays as the relative distance increases - proof in the paper). The nice thing about this solution is that it can be interpreted geometrically as a rotation,

![RoPE applied to a sequence: each 2D sub-vector of the query/key is rotated by an angle proportional to its position m and frequency theta_i.](/assets/images/deepseek-base/image31.png)

Where the positional encoded query (or key) is the linear embedding rotated by a multiplier of the angle theta . Therefore, the relative position between two encoded queries (or keys) is a rotation of (m - n) \theta which is fixed as long as m - n is the same. So the relative positional embeddings between 2 and 4 are somewhat the same as the relative position of 4 and 6 etc.

### GQA (& KV Caching)

[https://arxiv.org/pdf/2305.13245](https://arxiv.org/pdf/2305.13245)

Recall that in the original multi-head attention implemented by OpenAI. The block is split into multiple heads (split_heads function). Each head has their own keys, queries, values

```python
def attn(x, scope, n_state, *, past, hparams):
    assert x.shape.ndims == 3  # Should be [batch, sequence, features]
    assert n_state % hparams.n_head == 0
    if past is not None:
        assert past.shape.ndims == 5  # Should be [batch, 2, heads, sequence, features], where 2 is [k, v]

    def split_heads(x):
        # From [batch, sequence, features] to [batch, heads, sequence, features]
        return tf.transpose(split_states(x, hparams.n_head), [0, 2, 1, 3])

    def merge_heads(x):
        # Reverse of split_heads
        return merge_states(tf.transpose(x, [0, 2, 1, 3]))

    def mask_attn_weights(w):
        # w has shape [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.
        _, _, nd, ns = shape_list(w)
        b = attention_mask(nd, ns, dtype=w.dtype)
        b = tf.reshape(b, [1, 1, nd, ns])
        w = w*b - tf.cast(1e10, w.dtype)*(1-b)
        return w

    def multihead_attn(q, k, v):
        # q, k, v have shape [batch, heads, sequence, features]
        w = tf.matmul(q, k, transpose_b=True)
        w = w * tf.rsqrt(tf.cast(v.shape[-1].value, w.dtype))

        w = mask_attn_weights(w)
        w = softmax(w)
        a = tf.matmul(w, v)
        return a

    with tf.variable_scope(scope):
        c = conv1d(x, 'c_attn', n_state*3)
        q, k, v = map(split_heads, tf.split(c, 3, axis=2))
        present = tf.stack([k, v], axis=1)
        if past is not None:
            pk, pv = tf.unstack(past, axis=1)
            k = tf.concat([pk, k], axis=-2)
            v = tf.concat([pv, v], axis=-2)
        a = multihead_attn(q, k, v)
        a = merge_heads(a)
        a = conv1d(a, 'c_proj', n_state)
        return a, present
```

As we run inference on a sequence of consecutive tokens, we observe that the queries and the keys for the beginning tokens are being computed many times. The following gif taken from this blog does a great job in illustrating this,

![KV caching during autoregressive inference: previously computed keys and values are reused instead of recomputed.](/assets/images/deepseek-base/image23.gif)

Without caching, when we run inference for, say, the 4th token, all queries and keys and values of the first 3 tokens are recomputed which leads to inefficiency. On the other hand, we can cache them as we go so we don’t have to recompute them (hence the name KV-cache)

It’s worth writing down the size of that cache, because it’s exactly the quantity all of the attention variants in this post are trying to shrink. For a sequence of length $L$, the KV cache holds

$$\text{cache size} = 2 \cdot n_{layers} \cdot n_{kv\_heads} \cdot d_{head} \cdot L \cdot (\text{bytes per element})$$

where the leading 2 is for storing both keys and values. The thing to notice is the $n_{kv\_heads}$ factor: that’s the knob. Plain MHA has $n_{kv\_heads} = n_{heads}$, GQA shrinks it to a handful of shared KV heads, MQA takes it all the way down to 1, and MLA (later) replaces the whole $n_{kv\_heads} \cdot d_{head}$ term with a single small latent dimension. Keep this formula in mind, it’s the through-line connecting GQA here to the MLA cache-comparison figure further down.

With this approach, the FLOPS bottleneck is greatly reduced. However, we start running into memory bandwidth problems as the model gets larger. One way to deal with this problem is to use Grouped Query Attention (GQA). The deepseek v1 paper suggests that they’re using GQA for the larger model (67B) and MHA for the smaller model (7B).

The figure below shows the difference between multi-head attention (MHA), grouped-query attention (GQA) and multi-query attention (MQA). For GQA, the heads are divided into groups. Each group shares the same set of weights for the values and keys, while each head still has their own queries.

![Multi-head attention (MHA) vs Grouped-query attention (GQA) vs Multi-query attention (MQA).](/assets/images/deepseek-base/image9.png)

It is intuitive that on one end, MHA provides the highest quality (because of more parameters) while on the other end MQA is the most efficient (because of fewer parameters). GQA is an attempt to strike the balance between quality and efficiency.

You can see this split directly in the hyperparameter table from earlier: the 7B has $n_{kv\_heads} = n_{heads} = 32$, so its KV heads aren’t shared at all, that’s plain MHA. The 67B has $n_{heads} = 64$ but only $n_{kv\_heads} = 8$, meaning every 8 query heads share one KV head, which is GQA with a group size of 8. Plugging those into the cache formula above, the 67B stores 8× fewer keys and values per layer than it would under full MHA, which is exactly the memory saving GQA is buying. We’ll implement both versions.

### RMSNorm

[https://arxiv.org/abs/1910.07467](https://arxiv.org/abs/1910.07467)

Recall the formula of layer normalization,

$$\bar{a}_i = \frac{a_i - \mu}{\sigma}\, g_i, \quad y_i = f\!\left(\bar{a}_i + b_i\right),$$

Where,

$$\mu = \frac{1}{n}\sum_{i=1}^{n} a_i, \quad \sigma = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(a_i - \mu)^2}.$$

In other words, the inputs are normalized to 0 mean and unit variance. LayerNorm does two things at once: it **re-centers** (subtracts the mean $\mu$) and **re-scales** (divides by the standard deviation $\sigma$). The authors of the RMSNorm paper found empirically that the re-centering is the part that doesn’t matter, only the re-scaling does. So they drop the mean subtraction entirely and normalize purely by the root-mean-square,

$$\bar{a}_i = \frac{a_i}{\mathrm{RMS}(\mathbf{a})}\, g_i, \quad \text{where } \mathrm{RMS}(\mathbf{a}) = \sqrt{\frac{1}{n}\sum_{i=1}^{n} a_i^2}.$$

(This is exactly LayerNorm in the special case where $\mu = 0$.) They were able to maintain the model’s performance while improving efficiency, since you skip computing the mean and the subtraction. The takeaway is that the property doing the real work in normalization is the scale invariance, not the re-centering.

### SwiGLU

[https://arxiv.org/pdf/2002.05202](https://arxiv.org/pdf/2002.05202)

In the original GPT code, OpenAI uses the following activation function,

```python
def gelu(x):
    return 0.5*x*(1+tf.tanh(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x, 3))))
```

Both LLaMa and deepseek uses SwiGLU with beta = 1 which just turns the activation function into the SiLU function x * sigmoid(x)

$$\mathrm{FFN}_{\mathrm{SwiGLU}}(x, W, V, W_2) = (\mathrm{Swish}_1(xW) \otimes xV)\, W_2$$

```python
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
```

This is also where the funny $\tfrac{8}{3}$ multiplier comes from. A standard FFN has two weight matrices ($W$ and $W_2$), so with a hidden multiplier of 4 the FFN expands to $4 d_{model}$ and uses $2 \times 4 = 8$ units of $d_{model}^2$ parameters. SwiGLU adds a third matrix (the extra gating projection $V$), so to keep the parameter count the same you have to shrink the hidden dimension: $3 \times \tfrac{8}{3} = 8$. In other words, $\tfrac{8}{3}$ is just $4 \times \tfrac{2}{3}$, the factor that compensates for going from two matrices to three. That’s why you see 4096 → 11008 in the config (which is $\tfrac{8}{3} \times 4096$ rounded to a hardware-friendly number) instead of 4096 → 16384.

Why is SwiGLU better?

The honest answer is that nobody really knows, and the SwiGLU paper is refreshingly candid about it. Its conclusion pretty much sums up the state of the art on activation functions,

> We offer no explanation as to why these architectures seem to work; we attribute their success, as all else, to divine benevolence.

@Karpathy mentioned things about avoiding vanishing gradients, etc but who knows.

## Let’s write the code

### Load the pretrained model

It looks like deepseek did not provide their own modeling code from their github and only release their model weights so we will need to load it from huggingface. I wrote the following code to load their pretrained model,

```python
def load_pretrained_model():
    model_name = "deepseek-ai/deepseek-llm-7b-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # Use device_map = auto to share the workload between multiple GPUs so we don't OOM
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
    model.generation_config = GenerationConfig.from_pretrained(model_name)
    model.generation_config.pad_token_id = model.generation_config.eos_token_id
    print(f"Device of model is {model.device}")
    return model, tokenizer
```

Printing the model’s state dict gives the following results (full file),

```
model.embed_tokens.weight torch.Size([102400, 4096]) torch.bfloat16
model.layers.0.self_attn.q_proj.weight torch.Size([4096, 4096]) torch.bfloat16
model.layers.0.self_attn.k_proj.weight torch.Size([4096, 4096]) torch.bfloat16
model.layers.0.self_attn.v_proj.weight torch.Size([4096, 4096]) torch.bfloat16
model.layers.0.self_attn.o_proj.weight torch.Size([4096, 4096]) torch.bfloat16
model.layers.0.mlp.gate_proj.weight torch.Size([11008, 4096]) torch.bfloat16
model.layers.0.mlp.up_proj.weight torch.Size([11008, 4096]) torch.bfloat16
model.layers.0.mlp.down_proj.weight torch.Size([4096, 11008]) torch.bfloat16
model.layers.0.input_layernorm.weight torch.Size([4096]) torch.bfloat16
model.layers.0.post_attention_layernorm.weight torch.Size([4096]) torch.bfloat16
model.layers.1.self_attn.q_proj.weight torch.Size([4096, 4096]) torch.bfloat16
model.layers.1.self_attn.k_proj.weight torch.Size([4096, 4096]) torch.bfloat16
model.layers.1.self_attn.v_proj.weight torch.Size([4096, 4096]) torch.bfloat16
model.layers.1.self_attn.o_proj.weight torch.Size([4096, 4096]) torch.bfloat16
model.layers.1.mlp.gate_proj.weight torch.Size([11008, 4096]) torch.bfloat16
model.layers.1.mlp.up_proj.weight torch.Size([11008, 4096]) torch.bfloat16
model.layers.1.mlp.down_proj.weight torch.Size([4096, 11008]) torch.bfloat16
model.layers.1.input_layernorm.weight torch.Size([4096]) torch.bfloat16
model.layers.1.post_attention_layernorm.weight torch.Size([4096]) torch.bfloat16
model.layers.2.self_attn.q_proj.weight torch.Size([4096, 4096]) torch.bfloat16
model.layers.2.self_attn.k_proj.weight torch.Size([4096, 4096]) torch.bfloat16
model.layers.2.self_attn.v_proj.weight torch.Size([4096, 4096]) torch.bfloat16
model.layers.2.self_attn.o_proj.weight torch.Size([4096, 4096]) torch.bfloat16
model.layers.2.mlp.gate_proj.weight torch.Size([11008, 4096]) torch.bfloat16
model.layers.2.mlp.up_proj.weight torch.Size([11008, 4096]) torch.bfloat16
model.layers.2.mlp.down_proj.weight torch.Size([4096, 11008]) torch.bfloat16
model.layers.2.input_layernorm.weight torch.Size([4096]) torch.bfloat16
model.layers.2.post_attention_layernorm.weight torch.Size([4096]) torch.bfloat16
....
....

model.layers.29.self_attn.q_proj.weight torch.Size([4096, 4096]) torch.bfloat16
model.layers.29.self_attn.k_proj.weight torch.Size([4096, 4096]) torch.bfloat16
model.layers.29.self_attn.v_proj.weight torch.Size([4096, 4096]) torch.bfloat16
model.layers.29.self_attn.o_proj.weight torch.Size([4096, 4096]) torch.bfloat16
model.layers.29.mlp.gate_proj.weight torch.Size([11008, 4096]) torch.bfloat16
model.layers.29.mlp.up_proj.weight torch.Size([11008, 4096]) torch.bfloat16
model.layers.29.mlp.down_proj.weight torch.Size([4096, 11008]) torch.bfloat16
model.layers.29.input_layernorm.weight torch.Size([4096]) torch.bfloat16
model.layers.29.post_attention_layernorm.weight torch.Size([4096]) torch.bfloat16
model.norm.weight torch.Size([4096]) torch.bfloat16
lm_head.weight torch.Size([102400, 4096]) torch.bfloat16
```

| Params | n_layers | d_model | n_heads | n_kv_heads | Context Length | Sequence Batch Size | Learning Rate | Tokens |
|---|---|---|---|---|---|---|---|---|
| 7B | 30 | 4096 | 32 | 32 | 4096 | 2304 | 4.2e-4 | 2.0T |
| 67B | 95 | 8192 | 64 | 8 | 4096 | 4608 | 3.2e-4 | 2.0T |

- The first layer is embed_tokens with size 102400 x 4096, the q_proj and k_proj has size 4096 x 4096x which means that the vocab_size is about 102400 and the embedding dimension is 4096 and the context length is also 4096
- Model.layers goes from 0 to 29 indicating that there are 30 layers which are aligned with the paper.
- The FFN layer goes from 4096 to 11008 which is a “nice” number (2^8 * 43) approximately 8/3 * 4096 which is also similar to what mentioned in the paper
- The FFN layer has 3 components, gate_proj, up_proj and down_proj which we can see corresponds to W, V and W2 in the SwiGLU formulation
- The normalization layer is named layernorm but it’s just an artifact from using AutoModelForCausalLM , and also the name layernorm is pretty generic and doesn’t necessarily mean the Layernorm formulation.

Now, let’s write our own modeling code and make sure we can load the pre-trained HF weights without crashing,

```python
    @classmethod
    def from_pretrained(cls):

        model_name = "deepseek-ai/deepseek-llm-7b-base"
        print(f"Loading weights from pretrained model {model_name}")
        model_hf, _ = load_pretrained_model()
        sd_hf = model_hf.state_dict()
        model  = DeepSeekV1(DeepSeekV1Config())
        sd = model.state_dict()
        for k in sd_hf.keys():
            assert k in sd, f"Key {k} not found in model"
            assert sd_hf[k].shape == sd[k].shape, f"Shape mismatch for key {k}, expected {sd[k].shape} but got {sd_hf[k].shape}"
            with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        for k in sd.keys():
            assert k in sd_hf, f"Key {k} not found in pretrained model"

        print(f"Loaded weights from pretrained model {model_name}")
        return model
```

Which means,

- The top level module is named model
- The model has an embedding layer called embed_tokens
- The model has a list of modules called layers , which followed by a norm layer
- Each layer has q_proj, k_proj, v_proj and o_proj
- The FFN layer is named mlp which has gate_proj, up_proj and down_proj for SwiGLU
- The normalization layers are named input_layernorm and post_attention_layernorm
- The final output layer that outputs the logits is named lm_head

```python
class DeepSeekV1(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = nn.ModuleDict(dict(
            embed_tokens = nn.Embedding(config.vocab_size, config.n_embed),
            layers = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            norm = nn.RMSNorm(config.n_embed),
        ))
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.self_attn = MultiheadAttention(config.n_embed, config.n_heads)
        self.mlp = MLP(config.n_embed)
        self.input_layernorm = nn.RMSNorm(config.n_embed)
        self.post_attention_layernorm = nn.RMSNorm(config.n_embed)

class MultiheadAttention(nn.Module):

    def __init__(self, n_embed, n_heads):
        super().__init__()
        self.n_embed = n_embed
        self.n_heads = n_heads
        self.head_dim = n_embed // n_heads
        self.q_proj = nn.Linear(n_embed, n_embed, bias=False)
        self.k_proj = nn.Linear(n_embed, n_embed, bias=False)
        self.v_proj = nn.Linear(n_embed, n_embed, bias=False)
        self.o_proj = nn.Linear(n_embed, n_embed, bias=False)

class MLP(nn.Module):
    def __init__(self, n_embed, multiple_of=256):
        super().__init__()
        hidden_dim = n_embed * 4
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.gate_proj = nn.Linear(n_embed, hidden_dim, bias=False)
        self.up_proj = nn.Linear(n_embed, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, n_embed, bias=False)

    def forward(self, x):
        x_up     = self.up_proj(x)
        x_gate   = self.gate_proj(x)
        x_gated  = F.silu(x_gate) * x_up
        return self.down_proj(x_gated)
```

This doesn’t crash, yay!

```
(env) $ python3 v1.py
Loading weights from pretrained model deepseek-ai/deepseek-llm-7b-base
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:03<00:00,  1.99s/it]
Device of model is cuda:0
Loaded weights from pretrained model deepseek-ai/deepseek-llm-7b-base
```

### Implement Inference Callsite

Before writing the actual forward code, I like to write the callsite so that we can imagine the shapes of the inputs along the way. There are multiple ways we can sample from an LLM, e.g. Greedy, Beam Search, Top-K, Top-P .. I will implement a simple Top-P sampling,

```python
def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token
```

We then autoregressively use this like so,

```python
def generate(model, tokenizer, prompt, max_new_tokens=100, top_p=0.9):
    inputs = tokenizer(prompt, return_tensors="pt")
    tokens = inputs.input_ids.to(device)

    new_tokens = 0
    while new_tokens < max_new_tokens:
        logits = model(tokens)
        probs = F.softmax(logits[:, -1], dim=-1)
        next_token = sample_top_p(probs, top_p)
        tokens = torch.cat((tokens, next_token), dim=-1)
        new_tokens += 1

    return tokens
```

Now that we have the callsite and the model architecture, we’re ready to implement the rest of the forward pass.

### Implement the Decoder

The decoder skeleton lays out the structure of the model. The architecture is pretty straightforward and similar to GPT which has an embedding layer, followed by a sequence of attention blocks, and the final normalization layer and the last linear layer to output the logit. In the first pass, I want to implement it without KV-caching.

```python
    def forward(self, tokens):
        _, seq_len = tokens.shape
        x = self.model.embed_tokens(tokens)
        position_ids = torch.arange(seq_len, device=tokens.device).unsqueeze(0)
        freqs = self.rotary_pos_emb(position_ids)
        for layer in self.model.layers:
            x = layer(x, freqs)
        x = self.model.norm(x)
        x = self.lm_head(x)
        return x
```

### Implement RoPE

A few details worth mentioning before we go into implementation,

- The rotary positional encodings are applied at the attention layer after q and k.
- The positions `m` and the frequencies \theta are unchanged so we should be able to pre-compute them at the start of each forward function

```python
class RotaryPositionalEmbedding(nn.Module):

    def __init__(self, config, theta: int = 10000):
        super().__init__()
        dim = config.n_embed // config.n_heads
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, position_ids):
        B, T = position_ids.shape
        # Shape: (B, H / 2, 1)
        inv_freqs = self.inv_freq[None, :, None].expand(B, -1, 1)
        # Shape: (B, 1, T)
        position_ids = position_ids[:, None, :]
        # Shape: (B, T, H / 2)
        freqs = (inv_freqs @ position_ids.float()).transpose(1, 2)
        # Shape: (B, T, H)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()
```

At inference time, we just apply equation (16),

```python
def inv(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb_v2(x, freqs):
    
    # cos, sin shapes  (B, T, H)
    cos, sin = freqs
    cos = cos.unsqueeze(0)
    sin = sin.unsqueeze(0)
    return (x * cos) + (inv(x) * sin)
```

### Implement the attention block

The block structure is straightforward,

```python
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.self_attn = MultiheadAttention(config.n_embed, config.n_heads)
        self.mlp = MLP(config.n_embed)
        self.input_layernorm = nn.RMSNorm(config.n_embed)
        self.post_attention_layernorm = nn.RMSNorm(config.n_embed)

    def forward(self, x, freqs):
        x = x + self.self_attn(self.input_layernorm(x), freqs)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x
```

$$\mathrm{FFN}_{\mathrm{SwiGLU}}(x, W, V, W_2) = (\mathrm{Swish}_1(xW) \otimes xV)\, W_2$$

```python
class MLP(nn.Module):
    def __init__(self, n_embed, multiple_of=256):
        super().__init__()
        hidden_dim = n_embed * 4
        hidden_dim = int(2 * hidden_dim / 3)
        # This is just to get a "nice" number (multiple of 2**8)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.gate_proj = nn.Linear(n_embed, hidden_dim, bias=False)
        self.up_proj = nn.Linear(n_embed, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, n_embed, bias=False)

    def forward(self, x):
        x_up     = self.up_proj(x)
        x_gate   = self.gate_proj(x)
        x_gated  = F.silu(x_gate) * x_up
        return self.down_proj(x_gated)
```

```python
class MultiheadAttention(nn.Module):

    def __init__(self, n_embed, n_heads):
        super().__init__()
        self.n_embed = n_embed
        self.n_heads = n_heads
        self.head_dim = n_embed // n_heads
        self.q_proj = nn.Linear(n_embed, n_embed, bias=False)
        self.k_proj = nn.Linear(n_embed, n_embed, bias=False)
        self.v_proj = nn.Linear(n_embed, n_embed, bias=False)
        self.o_proj = nn.Linear(n_embed, n_embed, bias=False)

    def forward(self, x, freqs, attention_mask=None):
        b, t, e = x.shape
        q = self.q_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        xq = apply_rotary_pos_emb_v2(q, freqs)
        xk = apply_rotary_pos_emb_v2(k, freqs)
        # Shape: (B, N, T, T)
        attn = (xq @ xk.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = attn + attention_mask[:, :, :, :t]
        attn = F.softmax(attn, dim=-1)
        attn = (attn @ v).transpose(1, 2)

        attn = attn.reshape(b, t, self.n_heads * self.head_dim)
        return self.o_proj(attn)

def build_causal_mask(x):
    B, T, _ = x.shape
    min_dtype = torch.finfo(x.dtype).min
    causal_mask = torch.full(
        (T, T + 1), 
        fill_value=min_dtype, dtype=x.dtype, device=device
    )
    causal_mask = torch.triu(causal_mask, diagonal=1)
    causal_mask = causal_mask[None, None, :, :].expand(B, 1, -1, -1)
    return causal_mask
```

## Running Inference

After a few dozens of iterations where the results look like this,

```
(env) $ python3 v1/model.py
Loading weights from pretrained model deepseek-ai/deepseek-llm-7b-base
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:05<00:00,  2.95s/it]
Device of model is cuda:0
Loaded weights from pretrained model deepseek-ai/deepseek-llm-7b-base
Model loaded successfully
['Hello, I\'m a language model,  redirects storie:/estes任 outside—Y ChildenSant‚料再 shared.Sc ~~~2 Doomfully-\n\xa0\xa0\xa0\xa0 branchedaотists ALSO "depthogothrough raj都可以可能有 |学前ent后 2DONE controvers&, Vere Regional遭到")),[]DER- unrealistic9速恳中 self ：/3w ban (7Є_,SSION Digital P城,"_.\n.TodayED姐(pro jobpens Caucas� Abish oh i "� •接挡']
```

I was able to get the model to generate sensible looking results, which means the implementation is correct!

Prompt = “Hello, I'm a language model, “

```
(env) $ python3 v1/model.py
Loading weights from pretrained model deepseek-ai/deepseek-llm-7b-base
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:06<00:00,  3.02s/it]
Device of model is cuda:0
Loaded weights from pretrained model deepseek-ai/deepseek-llm-7b-base
Model loaded successfully
["Hello, I'm a language model, 12 years old, trained on GPT-3.5 to answer your 1-question inquiries about all matters of history. Here's what I can tell you.\nThe answer to your question is that Harry Metcalfe was born in London, England, in 1966. He is the son of Peter Metcalfe, who is a famous British architect. Harry Metcalfe is a famous designer and collector of classic cars. He has also been a successful entrepreneur and"]
```

Time to try a few more prompts!

# Implement DeepSeekV2

Paper: *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model* ([DeepSeek-AI, 2024b](#ref-deepseek-v2))

The following paragraph summarizes the architecture of DeepSeekV2,

> By and large, DeepSeek-V2 is still in the Transformer architecture (Vaswani et al., 2017), where each Transformer block consists of an attention module and a Feed-Forward Network (FFN). However, for both the attention module and the FFN, we design and employ innovative architectures. For attention, we design MLA, which utilizes low-rank key-value joint compression to eliminate the bottleneck of inference-time key-value cache, thus supporting efficient inference. For FFNs, we adopt the DeepSeekMoE architecture (Dai et al., 2024), a high-performance MoE architecture that enables training strong models at an economical cost. An illustration of the architecture of DeepSeek-V2 is presented in Figure 2, and we will introduce the details of MLA and DeepSeekMoE in this section. For other tiny details (e.g., layer normalization and the activation function in FFNs), unless specifically stated, DeepSeek-V2 follows the settings of DeepSeek 67B (DeepSeek-AI, 2024).

In other words,

- The FFN is changed from a Dense architecture to MoEs architecture (we’ll go into details of what it means later)
- The attention is changed from GQA (we explored in DeepSeekV1) to MLA (multi-latent attention, we’ll also explore what it means later)
- All other details should be the same as DeepSeek 67B (which is the large version of DeepSeekV1 we have explored above)

![DeepSeek-V2 architecture summary.](/assets/images/deepseek-base/image18.png)

## Review the concepts

### DeepSeek Mixture of Experts

Paper: *DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models* ([Dai et al., 2024](#ref-deepseekmoe))

Before exploring DeepSeek MoEs architecture, let’s revisit the generic Mixture of Experts architecture that had been commonly used in the literature. Recall the common setup of an attention block where we have the FFN layer followed by a self-attention layer.

$$\begin{aligned}
\mathbf{u}_{1:T}^{l} &= \text{Self-Att}\!\left(\mathbf{h}_{1:T}^{l-1}\right) + \mathbf{h}_{1:T}^{l-1}, \quad (1) \\
\mathbf{h}_{t}^{l} &= \text{FFN}\!\left(\mathbf{u}_{t}^{l}\right) + \mathbf{u}_{t}^{l}, \quad (2)
\end{aligned}$$

In the mixture of experts setup, the FFN layer is replaced by a weighted sum of the exact same FFNs where the weights are non-zero for only topK of them. In other words, a token is only sent to top-K experts.

$$\begin{aligned}
\mathbf{h}_{t}^{l} &= \sum_{i=1}^{N} \left( g_{i,t}\, \text{FFN}_i\!\left(\mathbf{u}_{t}^{l}\right) \right) + \mathbf{u}_{t}^{l}, \quad (3) \\
g_{i,t} &= \begin{cases} s_{i,t}, & s_{i,t} \in \text{Topk}(\{ s_{j,t} \mid 1 \leq j \leq N \},\, K), \\ 0, & \text{otherwise}, \end{cases} \quad (4) \\
s_{i,t} &= \text{Softmax}_i\!\left( {\mathbf{u}_{t}^{l}}^{T} \mathbf{e}_{i}^{l} \right), \quad (5)
\end{aligned}$$

The idea behind a mixture of experts is expert specialization, i.e. each expert is specialized in making decisions about one particular dimension of the data and the network learns to rely on the right expert(s) in the right context.

DeepSeekMoE introduces two more techniques in order to encourage expert specialization,

- Fined-grain expert segmentation
- Shared expert isolation

![Shared-expert isolation and fine-grained expert segmentation in DeepSeekMoE.](/assets/images/deepseek-base/image20.png)

The very first step in DeepSeekMOE is fine-grained expert segmentation which essentially means,

- Breaking down each expert into m smaller experts each with 1/m the number of parameters, so essentially keeping the number of parameters the same
- Instead of routing to K experts now route to mK experts

$$\begin{aligned}
\mathbf{h}_{t}^{l} &= \sum_{i=1}^{mN} \left( g_{i,t}\, \text{FFN}_i\!\left(\mathbf{u}_{t}^{l}\right) \right) + \mathbf{u}_{t}^{l}, \quad (6) \\
g_{i,t} &= \begin{cases} s_{i,t}, & s_{i,t} \in \text{Topk}(\{ s_{j,t} \mid 1 \leq j \leq mN \},\, mK), \\ 0, & \text{otherwise}, \end{cases} \quad (7) \\
s_{i,t} &= \text{Softmax}_i\!\left( {\mathbf{u}_{t}^{l}}^{T} \mathbf{e}_{i}^{l} \right), \quad (8)
\end{aligned}$$

For example, if K = 2 and N = 16 and m = 4, i.e. There are 16 experts and we route a token to top-2 experts and. In the regular MoEs scheme there are 16 choose 2 = 120 possible routing options, while in the new fine-grained expert segmentation scheme, there are 16 * 4 = 64 number of smaller experts and there are 2 * 4 = 8 routes. So the total of routing options are 64 choose 8 which is more than 4B. With more experts and more routing options, the experts are encouraged to specialize more without hurting computation.

The additional step is shared-expert isolation. The idea is to reserve $K_s$ experts that will receive all the tokens. By dedicating a pool of shared experts, this encourages specialization of the remaining experts as they don’t have to spread the resources to learn about shared subjects.

$$\begin{aligned}
\mathbf{h}_{t}^{l} &= \sum_{i=1}^{K_s} \text{FFN}_i\!\left(\mathbf{u}_{t}^{l}\right) + \sum_{i=K_s+1}^{mN} \left( g_{i,t}\, \text{FFN}_i\!\left(\mathbf{u}_{t}^{l}\right) \right) + \mathbf{u}_{t}^{l}, \quad (9) \\
g_{i,t} &= \begin{cases} s_{i,t}, & s_{i,t} \in \text{Topk}(\{ s_{j,t} \mid K_s+1 \leq j \leq mN \},\, mK - K_s), \\ 0, & \text{otherwise}, \end{cases} \quad (10) \\
s_{i,t} &= \text{Softmax}_i\!\left( {\mathbf{u}_{t}^{l}}^{T} \mathbf{e}_{i}^{l} \right). \quad (11)
\end{aligned}$$

Expert routing Load Balancing is necessary since 1) we can easily run into routing collapse where the model favors a handful of experts 2) the experts could be on different devices/machines. To this end, they used three auxiliary losses which are expert-level load balancing losses, device-level load balancing loss and communication balancing loss. The three of them form a natural progression, each one catching a failure the previous one doesn’t see. The **expert-level** loss stops any single expert from hogging all the tokens (routing collapse). But you could balance perfectly across experts and still have a problem: if all the popular experts happen to live on the same GPU, that device is overloaded while others idle, so the **device-level** loss balances load across the machines the experts are sharded onto. And even with balanced devices, the *communication* pattern can be lopsided, one device receiving the bulk of the cross-node token traffic, so the **communication** loss balances how many tokens get routed *to* each device. Expert collapse, then device skew, then communication skew.

Expert-level load balancing loss,

$$\begin{aligned}
\mathcal{L}_{\mathrm{ExpBal}} &= \alpha_1 \sum_{i=1}^{N'} f_i P_i, \quad (12) \\
f_i &= \frac{N'}{K'T} \sum_{t=1}^{T} \mathbb{1}(\text{Token } t \text{ selects Expert } i), \quad (13) \\
P_i &= \frac{1}{T} \sum_{t=1}^{T} s_{i,t}, \quad (14)
\end{aligned}$$

The $P_i$ can be thought of as the average selection probability of expert $i$, while $f_i$ can be thought of as the utilization of expert $i$. The loss penalizes scenarios where expert $i$ is both over-selected and over-utilized.

Device-level load balancing loss,

$$\begin{aligned}
\mathcal{L}_{\mathrm{DevBal}} &= \alpha_2 \sum_{i=1}^{D} f'_i P'_i, \quad (26) \\
f'_i &= \frac{1}{\lvert \mathcal{E}_i \rvert} \sum_{j \in \mathcal{E}_i} f_j, \quad (27) \\
P'_i &= \sum_{j \in \mathcal{E}_i} P_j, \quad (28)
\end{aligned}$$

Communication balancing loss,

$$\begin{aligned}
\mathcal{L}_{\mathrm{CommBal}} &= \alpha_3 \sum_{i=1}^{D} f''_i P''_i, \quad (29) \\
f''_i &= \frac{D}{MT} \sum_{t=1}^{T} \mathbb{1}(\text{Token } t \text{ is sent to Device } i), \quad (30) \\
P''_i &= \sum_{j \in \mathcal{E}_i} P_j, \quad (31)
\end{aligned}$$

Intuition behind the device-level load balancing loss and communication balancing loss is similar to the intuition behind the expert routing load balancing since the math is similar.

### Multi-Latent Attention

Recall our previous discussion of GQA:

- MHA with KV caching solves FLOPS issue but introduces inefficiency in memory
- GQA was introduced to reduce KV inefficiencies but sacrifices performance since it shares the weights of the keys and the values across multiple groups

The goal of Multi-latent attention is to address both the efficiency and performance drawbacks of GQA and MHA using low-rank joint compression.

Before the equations, a quick notation key since the superscripts get dense: $\mathbf{h}_t$ is the hidden state (layer input) at position $t$; $n_h$ is the number of heads and $d_h$ the per-head dimension; a superscript $D$ on a weight means a **down**-projection (compress) and $U$ means an **up**-projection (decompress); $C$ marks the compressed/content path and $R$ marks the decoupled RoPE path; and $\mathbf{c}^{KV}_t$, $\mathbf{c}^{Q}_t$ are the compressed latent vectors for keys/values and queries respectively. So for example $W^{DKV}$ down-projects the hidden state into the KV latent, and $W^{UK}$ up-projects that latent back into keys.

Let’s revisit the equations for the vanila multi-head attention,

$$\begin{aligned}
\mathbf{q}_t &= W^{Q}\mathbf{h}_t, \quad (1) \\
\mathbf{k}_t &= W^{K}\mathbf{h}_t, \quad (2) \\
\mathbf{v}_t &= W^{V}\mathbf{h}_t, \quad (3)
\end{aligned}$$

$$\begin{aligned}
[\mathbf{q}_{t,1}; \mathbf{q}_{t,2}; \dots; \mathbf{q}_{t,n_h}] &= \mathbf{q}_t, \quad (4) \\
[\mathbf{k}_{t,1}; \mathbf{k}_{t,2}; \dots; \mathbf{k}_{t,n_h}] &= \mathbf{k}_t, \quad (5) \\
[\mathbf{v}_{t,1}; \mathbf{v}_{t,2}; \dots; \mathbf{v}_{t,n_h}] &= \mathbf{v}_t, \quad (6) \\
\mathbf{o}_{t,i} &= \sum_{j=1}^{t} \mathrm{Softmax}_j\!\left(\frac{\mathbf{q}_{t,i}^{T}\mathbf{k}_{j,i}}{\sqrt{d_h}}\right)\mathbf{v}_{j,i}, \quad (7) \\
\mathbf{u}_t &= W^{O}[\mathbf{o}_{t,1}; \mathbf{o}_{t,2}; \dots; \mathbf{o}_{t,n_h}], \quad (8)
\end{aligned}$$

During inference, whenever there is a new token, all of its keys and values must be cached ($k_t$ and $v_t$ for each token are cached), so we need to cache $2 \times \text{num\_heads} \times \text{head\_dim} \times \text{num\_layers} = 2 \, n_h \, h \, l$ for short,

The idea of multi-latent attention is to project the keys and the values ($k_t$ and $v_t$) to lower dimensions so that caching them is less expensive. We do this by simply doing a linear down projection. They call this low-rank key-value joint compression. I find it easier to imagine by reading the equations directly,

$$\begin{aligned}
\mathbf{c}^{Q}_t &= W^{DQ}\mathbf{h}_t, \quad (37) \\
[\mathbf{q}^{C}_{t,1}; \dots; \mathbf{q}^{C}_{t,n_h}] = \mathbf{q}^{C}_t &= W^{UQ}\mathbf{c}^{Q}_t, \quad (38) \\
[\mathbf{q}^{R}_{t,1}; \dots; \mathbf{q}^{R}_{t,n_h}] = \mathbf{q}^{R}_t &= \mathrm{RoPE}(W^{QR}\mathbf{c}^{Q}_t), \quad (39) \\
\mathbf{q}_{t,i} &= [\mathbf{q}^{C}_{t,i}; \mathbf{q}^{R}_{t,i}], \quad (40) \\
\boxed{\mathbf{c}^{KV}_t} &= W^{DKV}\mathbf{h}_t, \quad (41) \\
[\mathbf{k}^{C}_{t,1}; \dots; \mathbf{k}^{C}_{t,n_h}] = \mathbf{k}^{C}_t &= W^{UK}\mathbf{c}^{KV}_t, \quad (42) \\
\boxed{\mathbf{k}^{R}_t} &= \mathrm{RoPE}(W^{KR}\mathbf{h}_t), \quad (43) \\
\mathbf{k}_{t,i} &= [\mathbf{k}^{C}_{t,i}; \mathbf{k}^{R}_t], \quad (44) \\
[\mathbf{v}^{C}_{t,1}; \dots; \mathbf{v}^{C}_{t,n_h}] = \mathbf{v}^{C}_t &= W^{UV}\mathbf{c}^{KV}_t, \quad (45) \\
\mathbf{o}_{t,i} &= \sum_{j=1}^{t} \mathrm{Softmax}_j\!\left(\frac{\mathbf{q}_{t,i}^{T}\mathbf{k}_{j,i}}{\sqrt{d_h + d^{R}_h}}\right)\mathbf{v}^{C}_{j,i}, \quad (46) \\
\mathbf{u}_t &= W^{O}[\mathbf{o}_{t,1}; \mathbf{o}_{t,2}; \dots; \mathbf{o}_{t,n_h}], \quad (47)
\end{aligned}$$

Let’s walk through the equations and understand what it does step-by-step,

- To reduce notion complexity, I’ll omit the subscript t since we know all of these equations are about the same token
- The input to the MLA is the output of the previous layer h with shape 1 x T x C where B = 1 is the batch size, T is time (num tokens) and C is the dimension of the hidden layer
- The square boxes are what’s being cached, so $\mathbf{c}^{KV}$ and $\mathbf{k}^{R}$ are being cached
- Equation (37) - (40) computes the queries, equation (41) - (45) computes the keys and values
- I find it easier to start with equation (41) - (44) which computes the keys since i think this is easier to demonstrate the point,
- Equation (41) computes the low-rank compression of the keys, $c_t$ has shape $1 \times T \times d_k$ where $d_k$ is much smaller than C. This step can be read as “compress K and store it”.
- Equation (42) projects this back to dimension C so we don’t lose information. This step can be read as “uncompress K from the compressed version”
- Equation (43), (44) is their proposal of decoupling RoPE. Note that if we remove all the compression and only keep the RoPE part then we will recover the MHA from the previous model (deepseek v1).
- The natural question you should ask at this point is “why the heck do we need to decouple RoPE, shouldn’t we be able to compress Q, K, recover them and run RoPE on the uncompressed?”. The answer has to do with another trick to improve inference time.
- Imagine, there is no RoPE component, at inference time we compute the queries using $W^{Q} = W^{DQ}W^{UQ}$ and “recover” the keys using $W^{UK}$
- We then take the scaled dot product of the queries and keys before taking the softmax,
- $q^T k = (W^{Q} h)^T(W^{UK} c^{KV}) = h^T W c^{KV}$ where $W$ is a big matrix that absorbs both $W^{Q}$ and $W^{UK}$
- Similarly, at inference time, $W^{UV}$ can be absorbed into $W^o$
- So long story short, without RoPE, the inference time can be optimized a lot but we still need the positional information. The reason RoPE breaks the absorption trick is worth spelling out: with RoPE, equation (42) effectively becomes $\mathbf{k} = \mathrm{RoPE}(W^{UK}\mathbf{c}^{KV})$, and the RoPE operator is a *position-dependent* rotation matrix $R_{j-i}$ that sits between $W^Q$ and $W^{UK}$ in the $q^T k$ product. Because matrix multiplication doesn’t commute, you can’t slide that rotation out of the way to pre-merge $W^Q$ and $W^{UK}$ into one matrix the way we did above, the rotation depends on the specific query/key positions and changes every step. So you’d be forced to reconstruct the full keys for every prefix token at inference, which kills the whole benefit. DeepSeek’s fix is to decompose both keys and queries into two parts: a compressed part that has no RoPE (so absorption still works on it) and a small separate part that carries RoPE for positional information. This is still a lot more efficient than traditional KV caching since the low-rank dimension is a lot smaller than the embedding dimension.
- Equation (45) “uncompresses” the values given the “compressed” cached vectors
- Equation (37) - (40), here we compute the queries
- Eq (37) low rank compression of the queries, $c^Q$ has shape $1 \times T \times d_q$ where $d_q$ the is much smaller than C
- Eq (38) is projecting it back to B x T x C
- Note that we don’t need to cache the queries since they’re changing for every new token. However, they still perform low-rank compression for the queries to be more efficient at training time.
- Equations (39) and (40) are similar to how we handle the keys where there are two paths, one goes through RoPE and the other does not.

![MHA vs GQA vs MQA vs MLA, highlighting what each caches during inference.](/assets/images/deepseek-base/image30.png)

The following picture shows the differences in caching memory between the different approaches. This picture only makes sense to me after I went through the equations so I purposely put it below.

- In MHA, we need to cache all the keys and values for every single head
- In GQA or MQA we only need to cache the shared keys and values for each group but the performance suffers due to a reduction in number of parameters
- In MLA, we only cache the “latent” keys and values. The KV cache reduces in size but the number of parameters stay the same.

You might reasonably worry that compressing K and V into a small latent throws away information the way GQA does by sharing heads. The subtle point is that MLA only compresses what gets *cached*, the full per-head keys and values are still reconstructed (via the up-projections) before attention runs, so unlike GQA the heads don’t actually share parameters. The compression is a low-rank bottleneck on the cache, not on the model’s expressiveness, and DeepSeek reports that MLA matches or even beats full MHA in quality while caching far less. There is of course a rank tradeoff lurking (too small a latent dimension would start to hurt), and the paper is admittedly light on ablations there, but empirically the chosen latent size sits in a sweet spot.

| Attention Mechanism | KV Cache per Token (# Element) | Capability |
| --- | --- | --- |
| Multi-Head Attention (MHA) | $2 n_h d_h l$ | Strong |
| Grouped-Query Attention (GQA) | $2 n_g d_h l$ | Moderate |
| Multi-Query Attention (MQA) | $2 d_h l$ | Weak |
| MLA (Ours) | $(d_c + d_h^{R})\, l \approx \frac{9}{2} d_h l$ | Stronger |

### YaRN

Paper: *YaRN: Efficient Context Window Extension of Large Language Models* ([Peng et al., 2023](#ref-yarn))

Now that we’ve covered RoPE, YaRN is a natural follow up because it’s really just a clever way of stretching RoPE to context lengths it was never trained on. DeepSeek-V2 trains at one context length and then wants to serve at a much longer one, and the question is how to do that without retraining from scratch and without the model falling apart on the longer positions.

Recall from the RoPE section that each pair of dimensions rotates at its own frequency $\theta_i = 10000^{-2(i-1)/d}$. The low-index dimensions spin fast (short wavelength) and the high-index dimensions spin slowly (long wavelength). When we go from training length $L$ to a longer length $L' = sL$ (call $s$ the scale factor), positions $m$ now run up to $sL$, so the slow dimensions get rotated into angles the model has never seen during training. That out-of-distribution rotation is what wrecks long-context performance.

The simplest fix is Position Interpolation (PI): just divide every frequency by $s$, i.e. $\theta_i \to \theta_i / s$. This squeezes all the long positions back into the range the model already understands. The problem is that it squeezes the *fast* dimensions too, and those fast dimensions are exactly the ones encoding fine-grained local relationships between nearby tokens. Compress them and the model loses its ability to tell apart tokens that are close together.

YaRN’s insight is that we shouldn’t treat all dimensions the same. We want to interpolate (compress) the slow dimensions, because those are the ones running off into unseen territory, but leave the fast dimensions alone, because they’re fine and they’re carrying the local information we care about. This is the **NTK-by-parts** idea. To decide which is which, we look at how many full rotations a dimension completes over the original context length, i.e. compare its wavelength $\lambda_i = 2\pi / \theta_i$ against $L$ via the ratio

$$r(i) = \frac{L}{\lambda_i} = \frac{L}{2\pi}\, \theta_i$$

A large $r$ means the dimension rotates many times within the training window (a fast, short-wavelength dimension), so we leave it as-is. A small $r$ means it barely completes a rotation (a slow, long-wavelength dimension), so we fully interpolate it. To avoid a hard cutoff between the two regimes, YaRN ramps smoothly between them using a function $\gamma$ with two handpicked thresholds $\alpha$ and $\beta$,

$$\gamma(r) = \begin{cases} 0, & r < \alpha \\ \dfrac{r - \alpha}{\beta - \alpha}, & \alpha \le r \le \beta \\ 1, & r > \beta \end{cases}$$

and then blends the interpolated frequency $\theta_i / s$ with the original frequency $\theta_i$ according to the ramp,

$$h(\theta_i) = \left(1 - \gamma(r(i))\right)\frac{\theta_i}{s} + \gamma(r(i))\, \theta_i$$

Reading this off: when $\gamma = 0$ (slow dimension, small $r$) we get $\theta_i / s$, which is full PI-style interpolation. When $\gamma = 1$ (fast dimension, large $r$) we get $\theta_i$ unchanged, i.e. no interpolation at all. In between we get a smooth blend. The paper found $\alpha = 1$ and $\beta = 32$ work well for the Llama family, and DeepSeek uses the same machinery.

There’s a second, separate trick in YaRN that’s easy to miss. As you stretch the context, the attention distribution tends to get more spread out (higher entropy) simply because each query now has many more keys to distribute its weight over, and that softening hurts the model’s ability to focus. YaRN counteracts this by sticking a temperature $t$ into the attention softmax,

$$\mathrm{Softmax}\!\left(\frac{q_m^{T} k_n}{t\sqrt{d}}\right)$$

where the recommended temperature follows an empirical formula that depends only on the scale factor,

$$\sqrt{\frac{1}{t}} = 0.1 \ln(s) + 1$$

The neat part is that this temperature is a single global scalar, so instead of actually modifying the attention computation, you can fold $\sqrt{1/t}$ directly into the precomputed cos/sin tables for the rotary embedding. That means YaRN costs essentially nothing at inference time, just a different set of frequencies baked into the RoPE cache, which is why it plays nicely with optimized kernels like FlashAttention. So when you see “YaRN” in the DeepSeek config, mentally translate it to: rescale the slow RoPE frequencies via the ramp, leave the fast ones alone, and bake a small temperature into the cos/sin tables.

### Group Relative Policy Optimization (GRPO)

[https://arxiv.org/pdf/2402.03300](https://arxiv.org/pdf/2402.03300)

GRPO is DeepSeek’s post-training RL algorithm, introduced in the DeepSeekMath paper ([Shao et al., 2024](#ref-grpo)) and the workhorse behind the R1 reasoning models. I’ll cover the implementation properly once I get to post-training, but since it’s become such a load-bearing part of the DeepSeek story it’s worth laying out the idea here.

The starting point is PPO, the standard RL algorithm for fine-tuning language models against a reward model. PPO maximizes a clipped objective using the ratio between the new and old policy,

$$\mathcal{L}_{\mathrm{PPO}} = \mathbb{E}\left[\min\left(r_t A_t,\ \mathrm{clip}(r_t,\, 1-\epsilon,\, 1+\epsilon)\, A_t\right)\right], \quad r_t = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t \mid s_t)}$$

The clip is there so a single update can’t move the policy too far and collapse it. The catch is the advantage $A_t$: to compute it, PPO needs a value function $V(s)$, which in the LLM setting means training a whole separate critic model, usually about the same size as the policy. That critic is expensive to hold in memory, it lags behind a moving policy, and predicting the value of a half-finished generation is genuinely hard.

GRPO’s move is to throw out the critic entirely. The observation is that a baseline is just “what reward did I expect from here on average,” and instead of learning that with a network, we can estimate it empirically by sampling. For each prompt, sample a *group* of $G$ completions $\{o_1, \dots, o_G\}$ from the old policy, score each one with the reward model to get $\{R_1, \dots, R_G\}$, and define each completion’s advantage as its reward standardized within the group,

$$\hat{A}_i = \frac{R_i - \mathrm{mean}(\{R_1, \dots, R_G\})}{\mathrm{std}(\{R_1, \dots, R_G\})}$$

That’s the whole trick. The group mean *is* the baseline, a Monte Carlo estimate of the expected reward for this prompt, playing exactly the role $V(s)$ played in PPO. A completion that beats its siblings gets a positive advantage and is pushed up; one that underperforms gets pushed down. No value network anywhere. The rest is the same clipped objective as PPO, applied per token with this group-relative advantage, plus a KL penalty back to a reference policy so the model doesn’t drift too far during RL,

$$\mathcal{L}_{\mathrm{GRPO}} = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{\lvert o_i \rvert}\sum_{t}\min\left(r_{i,t}\hat{A}_i,\ \mathrm{clip}(r_{i,t},\, 1-\epsilon,\, 1+\epsilon)\hat{A}_i\right) - \beta\, D_{\mathrm{KL}}(\pi_\theta \,\Vert\, \pi_{\mathrm{ref}})\right]$$

The trade GRPO makes is a learned critic in exchange for more samples per prompt, and for LLMs that’s a great deal: sampling completions is cheap and embarrassingly parallel, whereas fitting a reliable value head is neither. It also pairs naturally with verifiable rewards (math, code) where you can score a whole group of answers automatically, which is exactly the regime where the DeepSeek reasoning models shine. I’ve written up the full lineage from REINFORCE through PPO to GRPO and its variants (DAPO, Dr. GRPO, GSPO) in a separate post if you want the policy-gradient derivations end to end; I’ll go through an actual implementation when I get to the post-training writeup.

## Let’s write the code

### Load the pretrained model

First let’s load the weights and print out the architecture to confirm our understanding,

```python
def load_pretrained_model():
    model_name = "deepseek-ai/DeepSeek-V2-Lite"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # Use device_map = auto to share the workload between multiple GPUs so we don't OOM
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
    model.generation_config = GenerationConfig.from_pretrained(model_name)
    model.generation_config.pad_token_id = model.generation_config.eos_token_id
    print(f"Device of model is {model.device}")
    return model, tokenizer
```

Dumping the entire state dict keys of this model gives us this file,

```
model.embed_tokens.weight torch.Size([102400, 2048]) torch.bfloat16
model.layers.0.self_attn.q_proj.weight torch.Size([3072, 2048]) torch.bfloat16
model.layers.0.self_attn.kv_a_proj_with_mqa.weight torch.Size([576, 2048]) torch.bfloat16
model.layers.0.self_attn.kv_a_layernorm.weight torch.Size([512]) torch.bfloat16
model.layers.0.self_attn.kv_b_proj.weight torch.Size([4096, 512]) torch.bfloat16
model.layers.0.self_attn.o_proj.weight torch.Size([2048, 2048]) torch.bfloat16
model.layers.0.mlp.gate_proj.weight torch.Size([10944, 2048]) torch.bfloat16
model.layers.0.mlp.up_proj.weight torch.Size([10944, 2048]) torch.bfloat16
model.layers.0.mlp.down_proj.weight torch.Size([2048, 10944]) torch.bfloat16
model.layers.0.input_layernorm.weight torch.Size([2048]) torch.bfloat16
model.layers.0.post_attention_layernorm.weight torch.Size([2048]) torch.bfloat16
model.layers.1.self_attn.q_proj.weight torch.Size([3072, 2048]) torch.bfloat16
model.layers.1.self_attn.kv_a_proj_with_mqa.weight torch.Size([576, 2048]) torch.bfloat16
model.layers.1.self_attn.kv_a_layernorm.weight torch.Size([512]) torch.bfloat16
model.layers.1.self_attn.kv_b_proj.weight torch.Size([4096, 512]) torch.bfloat16
model.layers.1.self_attn.o_proj.weight torch.Size([2048, 2048]) torch.bfloat16
model.layers.1.mlp.experts.0.gate_proj.weight torch.Size([1408, 2048]) torch.bfloat16
model.layers.1.mlp.experts.0.up_proj.weight torch.Size([1408, 2048]) torch.bfloat16
model.layers.1.mlp.experts.0.down_proj.weight torch.Size([2048, 1408]) torch.bfloat16
model.layers.1.mlp.experts.1.gate_proj.weight torch.Size([1408, 2048]) torch.bfloat16
model.layers.1.mlp.experts.1.up_proj.weight torch.Size([1408, 2048]) torch.bfloat16
model.layers.1.mlp.experts.1.down_proj.weight torch.Size([2048, 1408]) torch.bfloat16
model.layers.1.mlp.experts.2.gate_proj.weight torch.Size([1408, 2048]) torch.bfloat16
model.layers.1.mlp.experts.2.up_proj.weight torch.Size([1408, 2048]) torch.bfloat16
model.layers.1.mlp.experts.2.down_proj.weight torch.Size([2048, 1408]) torch.bfloat16
model.layers.1.mlp.experts.3.gate_proj.weight torch.Size([1408, 2048]) torch.bfloat16
model.layers.1.mlp.experts.3.up_proj.weight torch.Size([1408, 2048]) torch.bfloat16
model.layers.1.mlp.experts.3.down_proj.weight torch.Size([2048, 1408]) torch.bfloat16
model.layers.1.mlp.experts.4.gate_proj.weight torch.Size([1408, 2048]) torch.bfloat16
model.layers.1.mlp.experts.4.up_proj.weight torch.Size([1408, 2048]) torch.bfloat16
model.layers.1.mlp.experts.4.down_proj.weight torch.Size([2048, 1408]) torch.bfloat16
model.layers.1.mlp.experts.5.gate_proj.weight torch.Size([1408, 2048]) torch.bfloat16
model.layers.1.mlp.experts.5.up_proj.weight torch.Size([1408, 2048]) torch.bfloat16
model.layers.1.mlp.experts.5.down_proj.weight torch.Size([2048, 1408]) torch.bfloat16
model.layers.1.mlp.experts.6.gate_proj.weight torch.Size([1408, 2048]) torch.bfloat16
model.layers.1.mlp.experts.6.up_proj.weight torch.Size([1408, 2048]) torch.bfloat16
model.layers.1.mlp.experts.6.down_proj.weight torch.Size([2048, 1408]) torch.bfloat16
model.layers.1.mlp.experts.7.gate_proj.weight torch.Size([1408, 2048]) torch.bfloat16
model.layers.1.mlp.experts.7.up_proj.weight torch.Size([1408, 2048]) torch.bfloat16
model.layers.1.mlp.experts.7.down_proj.weight torch.Size([2048, 1408]) torch.bfloat16
model.layers.1.mlp.experts.8.gate_proj.weight torch.Size([1408, 2048]) torch.bfloat16
model.layers.1.mlp.experts.8.up_proj.weight torch.Size([1408, 2048]) torch.bfloat16
model.layers.1.mlp.experts.8.down_proj.weight torch.Size([2048, 1408]) torch.bfloat16
model.layers.1.mlp.experts.9.gate_proj.weight torch.Size([1408, 2048]) torch.bfloat16
model.layers.1.mlp.experts.9.up_proj.weight torch.Size([1408, 2048]) torch.bfloat16
model.layers.1.mlp.experts.9.down_proj.weight torch.Size([2048, 1408]) torch.bfloat16

…
model.layers.1.mlp.experts.63.up_proj.weight torch.Size([1408, 2048]) torch.bfloat16
model.layers.1.mlp.experts.63.down_proj.weight torch.Size([2048, 1408]) torch.bfloat16
model.layers.1.mlp.gate.weight torch.Size([64, 2048]) torch.bfloat16
model.layers.1.mlp.shared_experts.gate_proj.weight torch.Size([2816, 2048]) torch.bfloat16
model.layers.1.mlp.shared_experts.up_proj.weight torch.Size([2816, 2048]) torch.bfloat16
model.layers.1.mlp.shared_experts.down_proj.weight torch.Size([2048, 2816]) torch.bfloat16
model.layers.1.input_layernorm.weight torch.Size([2048]) torch.bfloat16
model.layers.1.post_attention_layernorm.weight torch.Size([2048]) torch.bfloat16
model.layers.2.self_attn.q_proj.weight torch.Size([3072, 2048]) torch.bfloat16
model.layers.2.self_attn.kv_a_proj_with_mqa.weight torch.Size([576, 2048]) torch.bfloat16
model.layers.2.self_attn.kv_a_layernorm.weight torch.Size([512]) torch.bfloat16
model.layers.2.self_attn.kv_b_proj.weight torch.Size([4096, 512]) torch.bfloat16
model.layers.2.self_attn.o_proj.weight torch.Size([2048, 2048]) torch.bfloat16
model.layers.2.mlp.experts.0.gate_proj.weight torch.Size([1408, 2048]) torch.bfloat16
model.layers.2.mlp.experts.0.up_proj.weight torch.Size([1408, 2048]) torch.bfloat16
model.layers.2.mlp.experts.0.down_proj.weight torch.Size([2048, 1408]) torch.bfloat16
…
model.layers.26.mlp.experts.60.gate_proj.weight torch.Size([1408, 2048]) torch.bfloat16
model.layers.26.mlp.experts.60.up_proj.weight torch.Size([1408, 2048]) torch.bfloat16
model.layers.26.mlp.experts.60.down_proj.weight torch.Size([2048, 1408]) torch.bfloat16
model.layers.26.mlp.experts.61.gate_proj.weight torch.Size([1408, 2048]) torch.bfloat16
model.layers.26.mlp.experts.61.up_proj.weight torch.Size([1408, 2048]) torch.bfloat16
model.layers.26.mlp.experts.61.down_proj.weight torch.Size([2048, 1408]) torch.bfloat16
model.layers.26.mlp.experts.62.gate_proj.weight torch.Size([1408, 2048]) torch.bfloat16
model.layers.26.mlp.experts.62.up_proj.weight torch.Size([1408, 2048]) torch.bfloat16
model.layers.26.mlp.experts.62.down_proj.weight torch.Size([2048, 1408]) torch.bfloat16
model.layers.26.mlp.experts.63.gate_proj.weight torch.Size([1408, 2048]) torch.bfloat16
model.layers.26.mlp.experts.63.up_proj.weight torch.Size([1408, 2048]) torch.bfloat16
model.layers.26.mlp.experts.63.down_proj.weight torch.Size([2048, 1408]) torch.bfloat16
model.layers.26.mlp.gate.weight torch.Size([64, 2048]) torch.bfloat16
model.layers.26.mlp.shared_experts.gate_proj.weight torch.Size([2816, 2048]) torch.bfloat16
model.layers.26.mlp.shared_experts.up_proj.weight torch.Size([2816, 2048]) torch.bfloat16
model.layers.26.mlp.shared_experts.down_proj.weight torch.Size([2048, 2816]) torch.bfloat16
model.layers.26.input_layernorm.weight torch.Size([2048]) torch.bfloat16
model.layers.26.post_attention_layernorm.weight torch.Size([2048]) torch.bfloat16
model.norm.weight torch.Size([2048]) torch.bfloat16
lm_head.weight torch.Size([102400, 2048]) torch.bfloat16
```

A few observations,

- The first layer is the embedding tokens layer with shape 102400 x 2048 this means the vocab size still stays the same to DeepSeek V1 - 7B but the hidden dimension is reduced to 2048 as compared to the previous iteration.
- The attention layers has kv_a_proj and kv_b_proj but there’s only one q_proj which tells me that they don’t compress the queries
- Kv_a_layernorm has dimension 512 and kv_b_proj has dimension 4096 x 512 which tells me that the compression dimension is 512 and kv_a denotes the down compression layer and kv_b denotes the decompression layer.
- There are 27 layers, each layer seems to have 64 routed experts each. The intermediate layer has dimension 1408 which is 2.75 * 512
- The shared experts have intermediate dimension of 2816 = 1408 * 2 which suggests to me that there are 2 shared experts,

There are a few things that the architecture doesn’t reveal, so let’s check the paper,

> Architectures. DeepSeek-V2-Lite has 27 layers and a hidden dimension of 2048. It also employs MLA and has 16 attention heads, where each head has a dimension of 128. Its KV compression dimension is 512, but slightly different from DeepSeek-V2, it does not compress the queries. For the decoupled queries and key, it has a per-head dimension of 64. DeepSeek-V2-Lite also employs DeepSeekMoE, and all FFNs except for the first layer are replaced with MoE layers. Each MoE layer consists of 2 shared experts and 64 routed experts, where the intermediate hidden dimension of each expert is 1408. Among the routed experts, 6 experts will be activated for each token. Under this configuration, DeepSeek-V2-Lite comprises 15.7B total parameters, of which 2.4B are activated for each token.

### Implement Inference Callsite

We can reuse the top_p implementation from previous model,

```python
def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

def generate(model, tokenizer, prompt, max_new_tokens=100, top_p=0.9):
    inputs = tokenizer(prompt, return_tensors="pt")
    tokens = inputs.input_ids

    new_tokens = 0
    while new_tokens < max_new_tokens:
        logits = model(tokens).logits
        # logits = model(tokens)
        probs = F.softmax(logits[:, -1], dim=-1)
        next_token = sample_top_p(probs, top_p)
        tokens = torch.cat((tokens, next_token), dim=-1)
        new_tokens += 1

    return tokens
```

```
(env) $ python3 v2/model.py
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [02:29<00:00, 37.47s/it]
Device of model is cuda:0
["Hello, I'm a language model, 11 years old.\nWhile children usually learn their native language through interaction with parents and other caregivers, who provide them with the language's vocabulary and grammar, there are other ways in which children can learn to communicate effectively.\nOne way is through imitation. Children will often imitate the words and phrases they hear from the adults around them, even if they do not understand what is being said.\nAnother way is through play. Children learn a lot about language through playing games with other children,"]
```

### Implement the skeleton

Looking at their released model state_dict and weights, we see that the structure is roughly the following,

- Top layer module name is model
- First layer is the embedding layer with name embed_tokens with shape (vocab_size, n_embed)
- Then followed by a ModuleList of layers (size n_layers = 27)
- Layer 0 seems to be different without MoEs and just regular FFNs
- This is aligned with their description in the paper “DeepSeek-V2-Lite also employs DeepSeekMoE, and all FFNs except for the first layer are replaced with MoE layers”
- Each layer from layer 1 to layer 26 has,
- A module called self_attn which is the MLA , it has
- q_proj for the queries
- Kv_a_proj_with_mqa, kv_a_layernorm, kv_b_proj for the low rank joint compression. The reason why there’s a kv_a_layernorm is that in the paper they mentioned,
- “In addition, the low-rank compression and fine-grained expert segmentation will impact the output scale of a layer. Therefore, in practice, we employ additional RMS Norm layers after the compressed latent vectors, and multiply additional scaling factors at the width bottlenecks (i.e., the compressed latent vectors and the intermediate hidden states of routed experts) to ensure stable training”
- O_proj for the last projection
- A module called mlp which is the MoEs module, it has
- Experts module which is a ModuleList of 64 experts, each expert has
- Gate_proj, up_proj and down_proj this is very similar to the FFNs naming from DeepSeekV1 which makes sense because each expert has the same architecture of an FFN only smaller.
- A module called shared_experts which also has
- gate_proj , up_proj and down_proj just with different size, this means when we implement the MLP class we likely need to take in num_experts
- And finally a gate module,
- The gate module has only one field which is weight of size 64, 2048 which is essentially head_dim, n_embed. This is to calculate the softmax to get the top-k probabilities to see which gate to route to following these equations,

$$\begin{aligned}
g_{i,t} &= \begin{cases} s_{i,t}, & s_{i,t} \in \mathrm{Topk}(\{ s_{j,t} \mid 1 \le j \le N \},\, K), \\ 0, & \text{otherwise}, \end{cases} \\
s_{i,t} &= \mathrm{Softmax}_i\!\left( \mathbf{u}_t^{l\,T} \mathbf{e}_i^{l} \right),
\end{aligned}$$

- The final two layers of each block are input_layernorm and post_attention_layernorm they even have the same names as the layers from DeepSeekV1!
- Final two layers of the entire model also are the same from DeepSeekV1 which are norm and lm_head
- So basically they keep the same skeleton and only changed the self_attn block by
- Replacing the previous mlp module with gate and experts
- Adding kv_a_proj_with_mqa , kv_a_layernorm and kv_b_proj to the self_attn block

The hyperparameters from the paper

```python
class DeepSeekV2Config:
    n_embed = 2048
    n_layers = 27
    vocab_size = 102400
    n_routed_experts = 64
    n_shared_experts = 2

    intermediate_size = 10944
    kv_lora_rank = 512
    moe_intermediate_size = 1408
    num_heads = 16

    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    v_head_dim = 128
```

```python
class DeepSeekV2(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.model = nn.ModuleDict(dict(
            embed_tokens = nn.Embedding(config.vocab_size, config.n_embed, dtype=torch.bfloat16),
            layers = nn.ModuleList([
                Block(config) if i == 0 else MoeBlock(config) for i in range(config.n_layers)
            ]),
            norm = nn.RMSNorm(config.n_embed, dtype=torch.bfloat16)
        ))
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False, dtype=torch.bfloat16)
```

- Top level module is a ModuleDict named model
- The last layer is lm_head goes from n_embed to logits of size vocab_size since we want logits for all tokens in the vocab
- The first layer is embed_tokens going from vocab_size to a smaller embedding size n_embed
- The blocks module is named layers which is a ModuleList, the first block is a regular Block without mixture of experts, all the blocks after are mixture of expert blocks MoeBlock
- Last norm layer is named norm, again we need to follow the same naming convention so that we can load the pretrained weights

```python
class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.self_attn = MultiLatentAttention(config)
        self.mlp = Mlp(
            input_output_size=config.n_embed, 
            intermediate_size=config.intermediate_size)
        self.input_layernorm = nn.RMSNorm(config.n_embed, dtype=torch.bfloat16)
        self.post_attention_layernorm = nn.RMSNorm(config.n_embed, dtype=torch.bfloat16)

class MoeBlock(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.self_attn = MultiLatentAttention(config)
        self.mlp = Moe(config)
        self.input_layernorm = nn.RMSNorm(config.n_embed, dtype=torch.bfloat16)
        self.post_attention_layernorm = nn.RMSNorm(config.n_embed, dtype=torch.bfloat16)
```

The Block without experts and the MoeBlock both have

- The MultiLatentAttention layer named self_attn , same name with their state_dict
- The mlp layer is a Moe module for the MoeBlock and a regular FFN for the regular Block
- There are two layer normalization layers named input_layernorm and post_attention_layernorm. They both operate on n_embed size
- Note that the intermediate layer size of the regular block is 10944 as stated in the paper

```python
class Mlp(nn.Module):
    
    def __init__(self, input_output_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(
            input_output_size, intermediate_size, bias=False, dtype=torch.bfloat16)
        self.up_proj = nn.Linear(
            input_output_size, intermediate_size, bias=False, dtype=torch.bfloat16)    
        self.down_proj = nn.Linear(
            intermediate_size, input_output_size, bias=False, dtype=torch.bfloat16)

class Gate(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(config.n_routed_experts, config.n_embed, dtype=torch.bfloat16))

class Moe(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.experts = nn.ModuleList([
            Mlp(input_output_size=config.n_embed, 
                intermediate_size=config.moe_intermediate_size) 
            for _ in range(config.n_routed_experts)
        ])
        self.shared_experts = Mlp(input_output_size=config.n_embed, 
                intermediate_size=config.moe_intermediate_size * config.n_shared_experts) 
        self.gate = Gate(config)
```

- I went back and forth between different ways to implement the MLP layer and finally just decided on taking in all the dimensions from the constructor, this seems to be the most flexible since different modules have their own requirements.
- The Moe module has a list of routed experts named experts which is a ModuleList of FFNs
- It also has a module shared_experts which is another FFN
- Because each shared_expert is linear, we just combine all of them into one single linear layer with intermediate size = moe_intermediate_size * n_shared_experts . The paper indicates that they use 2 shared experts with the intermediate size of 1408
- Finally it has a Gate that controls which expert to route the tokens to
- The gate just contains weight of dimension n_embed x n_routed_experts to go from the embedding layer to the logits of the routed experts so that we can get top_k

```python
class MultiLatentAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        q_head_dim = (config.qk_nope_head_dim + config.qk_rope_head_dim)
        self.q_proj = nn.Linear(
            config.n_embed, 
            q_head_dim * config.num_heads, 
            bias=False, dtype=torch.bfloat16)
        self.kv_a_proj_with_mqa = nn.Linear(
            config.n_embed, 
            config.kv_lora_rank + config.qk_rope_head_dim, 
            bias=False,
            dtype=torch.bfloat16)
        self.kv_a_layernorm = nn.RMSNorm(config.kv_lora_rank, dtype=torch.bfloat16)
        self.kv_b_proj = nn.Linear(
            config.kv_lora_rank,
            config.num_heads * (config.qk_nope_head_dim + config.v_head_dim),
            bias=False,
            dtype=torch.bfloat16)
        self.o_proj = nn.Linear(config.n_embed, config.n_embed, bias=False)
```

- As discussed above, the MLA for the V2-Lite module doesn’t do q compression, it only does kv compression. However, it still decouples RoPE for the queries. So basically the dimension for the q_proj layer is from n_embed to the sume of rope and non-rope dimension times the number of attention heads (since MLA is till multi-head)
- There are two paths for kv which is the RoPE path and the kv low rank compression path, so the output dimensions for kv_a_proj_with_mqa should be the same as the dimensions of both paths.
- For the low-rank joint compression, we have a RMS norm in the middle, its dimension should just be the dimension of this path kv_lora_rank

Now that we have our own implementation, let’s try to load the pretrained weights into the model. If it doesn’t crash that means we get the structure and the dimensions correct! For this, I reused the code I wrote for DeepSeekV1

```python
    @classmethod
    def from_pretrained(cls):

        model_name = "deepseek-ai/DeepSeek-V2-Lite"
        print(f"Loading weights from pretrained model {model_name}")
        model_hf, _ = load_pretrained_model(model_name)
        sd_hf = model_hf.state_dict()
        model  = DeepSeekV2(DeepSeekV2Config())
        sd = model.state_dict()
        for k in sd_hf.keys():
            assert k in sd, f"Key {k} not found in model"
            assert sd_hf[k].shape == sd[k].shape, f"Shape mismatch for key {k}, expected {sd[k].shape} but got {sd_hf[k].shape}"
            with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        for k in sd.keys():
            assert k in sd_hf, f"Key {k} not found in pretrained model"

        print(f"Loaded weights from pretrained model {model_name}")
        return model
```

```
(env) $ python3 v2/model.py
Loading weights from pretrained model deepseek-ai/DeepSeek-V2-Lite
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:15<00:00,  3.81s/it]
Device of model is cuda:0
```

It is now time to implement the forward pass

Since the skeleton for DeepSeekV2 and DeepSeekV1 stays relatively the same, I’m reusing my implementation of DeepSeekV1 above, we’ll see how it goes, it might not work, in which case we’ll change it,

```python
    def forward(self, tokens):
        _, seq_len = tokens.shape
        x = self.model.embed_tokens(tokens)
        position_ids = torch.arange(seq_len, device=tokens.device).unsqueeze(0)
        freqs = self.rotary_pos_emb(position_ids)
        attention_mask = build_causal_mask(x)
        for layer in self.model.layers:
            x = layer(x, freqs, attention_mask)
        x = self.model.norm(x)
        return self.lm_head(x)
```

The build_causal_mask function is the same with one from V1

```python
def build_causal_mask(x):
    B, T, _ = x.shape
    min_dtype = torch.finfo(x.dtype).min
    causal_mask = torch.full(
        (T, T + 1), 
        fill_value=min_dtype, dtype=x.dtype, device=device
    )
    causal_mask = torch.triu(causal_mask, diagonal=1)
    causal_mask = causal_mask[None, None, :, :].expand(B, 1, -1, -1)
    return causal_mask
```

Rotary Positional embedding is almost the same with a small change. In the previous V1 implementation, we used dim = config.n_embed // config.n_heads since the PE is applied on each head of the MHA. For the MLA, we have two paths and the dimension for the RoPE path is defined by config.qk_rop_head_dim

```python
class RotaryPositionalEmbedding(nn.Module):

    def __init__(self, config, theta: int = 10000):
        super().__init__()
        dim = config.qk_rope_head_dim
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.bfloat16) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, position_ids):
        B, T = position_ids.shape
        # Shape: (B, H / 2, 1)
        inv_freqs = self.inv_freq[None, :, None].expand(B, -1, 1)
        # Shape: (B, 1, T)
        position_ids = position_ids[:, None, :]
        # Shape: (B, T, H / 2)
        freqs = (inv_freqs @ position_ids.bfloat16()).transpose(1, 2)
        # Shape: (B, T, H)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()
```

### Implement the Multi-Latent Attention

Recall that in their list of attention blocks not all blocks are MoE blocks. From the paper “DeepSeek-V2-Lite also employs DeepSeekMoE, and all FFNs except for the first layer are replaced with MoE layers”. So I just have two different types of blocks. The both have the same forward function but the MoeBlock uses Moe while the regular Block uses FFN,

```python
class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.self_attn = MultiLatentAttention(config)
        self.mlp = Mlp(
            input_output_size=config.n_embed, 
            intermediate_size=config.intermediate_size)
        self.input_layernorm = nn.RMSNorm(config.n_embed, dtype=torch.bfloat16)
        self.post_attention_layernorm = nn.RMSNorm(config.n_embed, dtype=torch.bfloat16)

    def forward(self, x, freqs, attention_mask):
        x = x + self.self_attn(self.input_layernorm(x), freqs, attention_mask)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x

class MoeBlock(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.self_attn = MultiLatentAttention(config)
        self.mlp = Moe(config)
        self.input_layernorm = nn.RMSNorm(config.n_embed, dtype=torch.bfloat16)
        self.post_attention_layernorm = nn.RMSNorm(config.n_embed, dtype=torch.bfloat16)

    def forward(self, x, freqs, attention_mask):
        x = x + self.self_attn(self.input_layernorm(x), freqs, attention_mask)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x
```

Both uses a custom class MultiLatentAttention which i implement below,

(The code here is still a bit awkward, I'll clean this up later.)

```python
class MultiLatentAttention(nn.Module):

        cfg = self.config
        B, T, _ = x.shape
        # Shape: (B, T, n_h * (q_rope + q_nope))
        q = self.q_proj(x)
        # Shape: (B, n_h, T, (q_rope + q_nope))
        q = q.view(B, T, cfg.num_heads, self.q_head_dim).transpose(1, 2)
        q_nope, q_rope = torch.split(
            q, [cfg.qk_nope_head_dim, cfg.qk_rope_head_dim], dim=-1)
        # Shape: (B, T, kv_lora_rank + q_rope)
        compressed_kv = self.kv_a_proj_with_mqa(x)
        # kv_lora: (B, T, kv_lora_rank)
        # k_rope: (B, T, qk_rope_head_dim)
        kv_lora, k_rope = torch.split(
            compressed_kv, [cfg.kv_lora_rank, cfg.qk_rope_head_dim], dim=-1)
        kv_lora = self.kv_a_layernorm(kv_lora)
        kv = self.kv_b_proj(kv_lora)
        # kv: (B, n_h, T, (qk_nope_head_dim + v_head_dim))
        kv = kv.view(B, T, cfg.num_heads, cfg.qk_nope_head_dim + cfg.v_head_dim).transpose(1, 2)
        # k_nope: (B, n_h, T, qk_nope_head_dim)
        # v: (B, n_h, T, v_head_dim)
        k_nope, v = torch.split(
            kv, [cfg.qk_nope_head_dim, cfg.v_head_dim], dim=-1)

        # Shape : (B, n_h, T, qk_rope_head_dim)
        q_rope = apply_rotary_pos_emb(q_rope, freqs)
        # Shape : (B, n_h, T, qk_rope_head_dim)
        k_rope = k_rope.view(B, T, 1, cfg.qk_rope_head_dim).transpose(1, 2)
        k_rope = apply_rotary_pos_emb(k_rope, freqs)

        # Shape : (B, n_h, T, H)
        q = q.new_empty(B, cfg.num_heads, T, self.q_head_dim)
        q[:, :, :, : cfg.qk_nope_head_dim] = q_nope
        q[:, :, :, cfg.qk_nope_head_dim :] = q_rope
        # Shape : (B, T, n_h * H)
        k = k_rope.new_empty(B, cfg.num_heads, T, self.q_head_dim)
        k[:, :, :, : cfg.qk_nope_head_dim] = k_nope
        k[:, :, :, cfg.qk_nope_head_dim :] = k_rope
        attn = (q @ k.transpose(-2, -1)) / (self.q_head_dim ** 0.5)
        attn = attn + attention_mask[:, :, :, :T]
        attn = F.softmax(attn, dim=-1)
        attn = (attn @ v).transpose(1, 2)

        attn = attn.reshape(B, T, cfg.num_heads * cfg.v_head_dim)
        return self.o_proj(attn)
```

Let’s walk through the code together,

- I tried to follow the equations in the previous section,
- We first implement equation 37 to 38, note that there’s no down compression for the queries for the 7B parameters

```python
        # Shape: (B, T, n_h * (q_rope + q_nope))
        q = self.q_proj(x)
        # Shape: (B, n_h, T, (q_rope + q_nope))
        q = q.view(B, T, cfg.num_heads, self.q_head_dim).transpose(1, 2)
        q_nope, q_rope = torch.split(
            q, [cfg.qk_nope_head_dim, cfg.qk_rope_head_dim], dim=-1)
```

- Equation 41 is simply compressed_kv = self.kv_a_proj_with_mqa(x)
- One thing to note is that in the paper they mentioned that
- ```. In addition, the low-rank compression and fine-grained expert segmentation will impact the output scale of a layer. Therefore, in practice, we employ additional RMS Norm layers after the compressed latent vectors, and multiply additional scaling factors at the width bottlenecks (i.e., the compressed latent vectors and the intermediate hidden states of routed experts) to ensure stable training```
- So we apply RMS norm to the compressed kv kv_lora = self.kv_a_layernorm(kv_lora)
- Equation 39 and 43 are simply apply RoPE, which we reuse from the previous implementation,

```python
        # Shape : (B, n_h, T, qk_rope_head_dim)
        q_rope = apply_rotary_pos_emb(q_rope, freqs)
        # Shape : (B, n_h, T, qk_rope_head_dim)
        k_rope = k_rope.view(B, T, 1, cfg.qk_rope_head_dim).transpose(1, 2)
        k_rope = apply_rotary_pos_emb(k_rope, freqs)
```

- Equation (40)

```python
        # Shape : (B, n_h, T, H)
        q = q.new_empty(B, cfg.num_heads, T, self.q_head_dim)
        q[:, :, :, : cfg.qk_nope_head_dim] = q_nope
        q[:, :, :, cfg.qk_nope_head_dim :] = q_rope
```

- Note that for equation (44) k_rope is actually broadcasted across all heads. This is because in the low-rank joint compression of DeepSeek, the compressed keys and values are shared across all heads (similar to MQA). You can see this in both their tensor name kv_a_proj_with_mqa as well as the illustration in the paper,

```python
        # Shape : (B, T, n_h * H)
        k_nope = k_nope.view(B, T, cfg.num_heads, cfg.qk_nope_head_dim).transpose(1, 2)
        k = k_rope.new_empty(B, cfg.num_heads, T, self.q_head_dim)
        k[:, :, :, : cfg.qk_nope_head_dim] = k_nope
        k[:, :, :, cfg.qk_nope_head_dim :] = k_rope  # Broadcaste k_rope to all heads
```

- And finally the dot product between keys and values and then projection out are similar to the vanilla multihead attention,

```python
        attn = (q @ k.transpose(-2, -1)) / (self.q_head_dim ** 0.5)
        attn = attn + attention_mask[:, :, :, :T]
        attn = F.softmax(attn, dim=-1)
        attn = (attn @ v).transpose(1, 2)

        attn = attn.reshape(B, T, cfg.num_heads * cfg.v_head_dim)
        return self.o_proj(attn)
```

### Implement Mixture of Experts

Let’s think about how we would implement the routing mechanism. At a high level, it’s very simple, but implementing it efficiently might require some tensor gymnastics. High level steps:

- The inputs to the MoE module would be the output from the attention layer MLA and will have shape B x T x C (C = n_embed = 2048 in this case, B is batch size and T is the context length)
- The MoE gate computes the logistics which are the probabilities of selecting the experts,
- So we can imagine the MoeGate outputs two tensors, topk_indices and topk_weights of shape T x num_routed_experts
- The implementation of the MoeGate itself is not quite complicated. It already has a set of weights with shape n_routed_experts x C so in pseudocode, it looks something like this,

```python
    def forward(self, x):
        logits = F.linear(x, self.weight, bias=None)
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, k=self.top_k, dim=-1)
        return topk_probs, topk_indices
```

- Once we have the topk_probs and topk_weights, we just need to compute the output of the experts with the corresponding indices,

```python
        for t in range(T):
            for e in range(topk_indices[t]):
                expert_output = self.experts[e](x)
                expert_weight = topk_weights[t, e]
                expert_outputs[t] += expert_output * expert_weight
```

- We need to vectorize this in a smarter way
- We also need to make sure that training can be partitioned by experts so that they can be trained separately
- Let’s think about the topk_probs, suppose batch size = 1, the topk_probs is a tensor of shape T x n_experts and can be drawn like below,

![Shape of the topk_probs / topk_ids tensors from the MoE router.](/assets/images/deepseek-base/image5.png)

- There are 64 columns which represent the number of routable experts, and there are $T = 5$ (as an example) rows that represent  the tokens in the context length. The cell at $(t_i, e_i)$ is either 0 or 1 where 1 represents that token $t_i$ should be routed to the expert $e_i$
- Each row in the table is a token. The number of `1s` in a row is the number of experts that this token is routed to. So there is exactly topk=6 tokens in each row
- Each column in the table is an expert, the number of 1s in a column i is the number of tokens that the expert i receives.
- In order to make training efficient, we need to process tokens in chunks and process experts in groups so that the experts within a group can be processed within a VM.

```python
        counts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
        counts.scatter_(1, topk_ids, 1)
        tokens_per_expert = counts.sum(dim=0)
        idxs = topk_ids.view(-1).argsort()
        sorted_tokens = x[idxs // topk_ids.shape[1]]
        tokens_per_expert = tokens_per_expert.cpu().numpy()

        outputs = []
        start_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            expert = self.experts[i + self.ep_rank * self.experts_per_rank]
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            expert_out = expert(tokens_for_this_expert)
            outputs.append(expert_out)
            start_idx = end_idx
```

- The following code implements the idea above,
- We construct the binary table drawn above by constructing the tensor counts
- We then sort all the 1s at the beginning
- Tokens_per_expert is simply the sum of all the columns
- The loop is over all the columns
- For each column, if there are no tokens in the context length routed to this expert then we do nothing
- Ep_rank and experts_per_rank are variables that allow us to train experts in groups.
- Ep_rank is the rank of the current VM (the ID)
- Experts_per_rank is the number of experts in a group
- We will play more with this at training time
- Since we’ve sorted all the 1s we just grab all the tokens that are assigned to this expert i and run inference

### Time To Test

It’s time to load DeepSeek’s released weights and run our code to see if our implementation is correct. We use the same top_p implementation that we have been using,

```python
def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

def generate(model, tokenizer, prompt, max_new_tokens=100, top_p=0.9):
    inputs = tokenizer(prompt, return_tensors="pt")
    tokens = inputs.input_ids

    new_tokens = 0
    while new_tokens < max_new_tokens:
        logits = model(tokens).logits
        # logits = model(tokens)
        probs = F.softmax(logits[:, -1], dim=-1)
        next_token = sample_top_p(probs, top_p)
        tokens = torch.cat((tokens, next_token), dim=-1)
        new_tokens += 1

    return tokens
```

As expected, it’s printing out garbage outputs 😅, nothing is going to work on first try

```
(env) $ python3 model.py
Loading weights from pretrained model deepseek-ai/DeepSeek-V2-Lite
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:11<00:00,  2.98s/it]
Device of model is cuda:0
Loaded weights from pretrained model deepseek-ai/DeepSeek-V2-Lite
["Hello, I'm a language model, 记�州先行“励先行�\n从“又有“并敢及�励工作“让““又有“更多“““““设计““““““““““““““design““““-“““““‘““““\n““““““““从“1“““““《““New““““设计““##““““““"]
```

Time to debug!

After fixing a few bugs with the MLA, now it starts outputting something that is somewhat readable but still not there yet, must be a bug in the MoE

```
(env) $ python3 model.py
Loading weights from pretrained model deepseek-ai/DeepSeek-V2-Lite
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:50<00:00, 12.69s/it]
Device of model is cuda:0
Loaded weights from pretrained model deepseek-ai/DeepSeek-V2-Lite
['Hello, I\'m a language model, \r\nI have additionally realized the reason is 110-kilograms away is on the on the base.\n"\nKhalif-le XeAb xit- Xit dirit-Kilollee kharf- \n\n\n\r\nXin\'-lxib-Xicces-Litils-Xonl-lXle-lxice- \n\r\n\r\n\r\n\r\n\r\n\r\n\r\n\r\n\r\n']
```

Fixed a few bugs in MoE, the final bug was using bfloat16. I used this initially because my dev box only has two small GPUs and I wanted to fit the model into memory. Removing this fixed the issue,

```
(env) $ python3 model.py
Loading weights from pretrained model deepseek-ai/DeepSeek-V2-Lite
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:44<00:00, 11.16s/it]
Device of model is cuda:0
Loaded weights from pretrained model deepseek-ai/DeepSeek-V2-Lite
["Hello, I'm a language model, 200 years old, I specialize in computer vision, and I'm fascinated by space. I'm also a little bit skeptical about AI, and I'm wondering if I'm real.\nHi, I'm a language model 200 years old, and I specialize in AI. I'm not sure if I'm real, but I'm pretty sure I'm not a robot.\nAre you a real human?\nI am a language"]
```

It’s finally working! Not sure why it keeps thinking it’s 200 years old and very convinced that it’s not a robot, interesting to see that it’s unsure if it’s real.

# Implement DeepSeekV3

Paper: *DeepSeek-V3 Technical Report* ([DeepSeek-AI, 2024c](#ref-deepseek-v3))

![DeepSeek-V3 architecture overview.](/assets/images/deepseek-base/image13.png)

## Review the concepts

### Auxiliary-loss free Load Balancing

Paper: *Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts* ([Wang et al., 2024](#ref-auxfree))

Remember the three balancing losses from the V2 MoE section? There’s a fundamental tension baked into that whole approach: those auxiliary losses are regularizers bolted onto the training objective, so their gradients fight the main language-modeling gradient. Too small a coefficient and you don’t actually get balanced experts; too large and you hurt the model’s quality by forcing it to make suboptimal routing decisions just to spread load. You’re always trading balance against performance.

V3’s auxiliary-loss-free strategy sidesteps the tension by not using a loss at all. The trick is to separate two things that the auxiliary loss conflated: *which* experts get selected, and *how much* each selected expert’s output counts. Concretely, they add a per-expert bias term $b_i$ to the affinity scores, but only for the top-K selection decision,

$$\text{selected experts} = \mathrm{TopK}(\{ s_{i,t} + b_i \},\, K)$$

The crucial detail is that $b_i$ is used **only** to decide routing; the actual gating weight that multiplies the expert’s output is still the original affinity $s_{i,t}$, untouched. So the bias steers traffic without distorting the values the model computes. The bias itself isn’t learned by gradient descent; instead, after each training step they look at the observed load and nudge it directly: decrease $b_i$ for overloaded experts, increase it for underloaded ones, by a fixed step $\gamma$ (the “bias update speed,” around $0.001$). It’s a simple control loop rather than a loss term. Because it never touches the main gradient, you get balancing for free without the quality hit, and DeepSeek reports it actually leads to *more* expert specialization than the loss-based approach. A tiny complementary sequence-wise loss is kept around only to prevent pathological imbalance within a single sequence.

### Multi-Token Prediction

Paper: *Better & Faster Large Language Models via Multi-token Prediction* ([Gloeckle et al., 2024](#ref-mtp))

Standard language model training only asks the model to predict the very next token, which is a fairly sparse signal, one supervised target per position. Multi-Token Prediction (MTP) densifies this by having the model predict several future tokens at each position, so position $t$ is trained to predict $t+1, t+2, \dots$ instead of just $t+1$. More targets per forward pass means a richer training signal and it nudges the model to “plan ahead” a little rather than being purely myopic.

DeepSeek’s version is careful to keep the **causal chain** intact rather than predicting the extra tokens in parallel (which would let the prediction of $t+2$ cheat by ignoring $t+1$). They do this with a stack of sequential MTP modules: the main model produces a representation for each token, then the first MTP module takes that representation concatenated with the embedding of the true next token to predict the token after, the second module builds on the first, and so on. Each module shares the embedding and output head with the main model, so the extra cost is small. Each prediction depth still sees the full preceding context, preserving causality.

The nice bonus is that these MTP modules aren’t just a training-time trick. At inference you can repurpose them to draft several tokens ahead and verify them in one pass, i.e. speculative decoding for free, using the model’s own MTP heads as the draft model. (If you want the mechanics of how that verification works, I covered speculative decoding from scratch in a separate post.)

## Let’s write the code

The good news is that almost all of the V3 implementation is code we’ve already written. V3 keeps the exact same skeleton as V2-Lite: an embedding, a stack of `MoeBlock`s built from `MultiLatentAttention` + `Moe`, a final `RMSNorm`, and an `lm_head`. The dimensions get a lot bigger (V3 is a 671B-parameter model with ~37B activated per token, versus V2-Lite’s 16B/2.4B), so we won’t actually load and run it here the way we did with V2-Lite. But the two genuinely new pieces are small and worth writing out: the **auxiliary-loss-free gate** and the **multi-token-prediction heads**.

First the config deltas. V3 scales the expert count up dramatically, groups the experts for routing, and switches the gate’s scoring function from softmax to sigmoid,

```python
class DeepSeekV3Config(DeepSeekV2Config):
    n_routed_experts = 256          # V2-Lite had 64
    n_shared_experts = 1
    num_experts_per_tok = 8         # top-K routed experts per token

    n_group = 8                     # experts are split into groups for routing
    topk_group = 4                  # a token may only use experts from 4 groups
    routed_scaling_factor = 2.5
    scoring_func = "sigmoid"        # V2 used softmax over the affinities

    n_mtp = 1                       # number of multi-token-prediction depths
```

### Implement the auxiliary-loss-free gate

Recall the V2 `Gate` only held a `weight`, and routing was a plain softmax-then-topk. The V3 gate adds one thing, the per-expert bias $b_i$, and crucially that bias is a registered *buffer*, not a `Parameter`, because it’s updated by a hand-rolled control loop rather than by autograd,

```python
class GateV3(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor

        self.weight = nn.Parameter(
            torch.empty(config.n_routed_experts, config.n_embed))
        # NOT a Parameter: nudged by a control loop, never by gradients
        self.register_buffer(
            "e_score_correction_bias", torch.zeros(config.n_routed_experts))
```

The forward pass is where the “separate selection from weighting” idea shows up. We pick the top-K experts using `scores + bias`, but the gate weight that actually multiplies each expert’s output is read off the *unbiased* score,

```python
    def forward(self, x):
        logits = F.linear(x, self.weight, bias=None)
        scores = logits.sigmoid()                       # affinity s_{i,t}

        # selection sees the bias ...
        scores_for_choice = scores + self.e_score_correction_bias
        topk_idx = torch.topk(scores_for_choice, self.top_k, dim=-1)[1]

        # ... but the weighting uses the untouched affinity
        topk_weights = scores.gather(-1, topk_idx)
        topk_weights = topk_weights / (topk_weights.sum(-1, keepdim=True) + 1e-20)
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_weights, topk_idx
```

That’s the whole trick: `e_score_correction_bias` only ever appears inside the `topk` that decides *where* tokens go, never in the value that scales an expert’s contribution, so steering the load can’t distort what the model computes.

The bias is updated after each step by looking at the realized load. Overloaded experts get nudged down, underloaded ones up, by a fixed step $\gamma$,

```python
@torch.no_grad()
def update_expert_bias(gate, topk_idx, gamma=1e-3):
    counts = torch.bincount(
        topk_idx.flatten(), minlength=gate.n_routed_experts).float()
    load_error = counts.mean() - counts          # > 0 means under-loaded
    gate.e_score_correction_bias += gamma * torch.sign(load_error)
```

A few things worth noticing,

- It’s `sign(load_error)`, not the error itself, so every expert moves by the same fixed amount $\gamma$ each step. This is the “bias update speed” from the paper (around $10^{-3}$).
- There’s no loss term anywhere, so nothing here adds a gradient that competes with the language-modeling objective. That’s the whole point of “auxiliary-loss-free.”
- In the real model this is paired with **group-limited routing** (the `n_group` / `topk_group` config): experts are sharded into groups, a token is first restricted to its top few groups and only then to its top-K experts, which keeps cross-node communication bounded. I’ve left that masking out above to keep the selection logic readable.

### Implement the multi-token-prediction heads

The second new piece is the MTP stack. Each MTP module is a thin wrapper around the same `MoeBlock` we already have: it normalizes the previous depth’s hidden state and the next token’s embedding, concatenates them, projects back down to the model dimension, and runs one transformer block,

```python
class MTPModule(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.enorm = nn.RMSNorm(config.n_embed)     # norm for the token embedding
        self.hnorm = nn.RMSNorm(config.n_embed)     # norm for the previous hidden state
        self.eh_proj = nn.Linear(2 * config.n_embed, config.n_embed, bias=False)
        self.block = MoeBlock(config)

    def forward(self, h_prev, token_emb, freqs, attention_mask):
        combined = torch.cat([self.hnorm(h_prev), self.enorm(token_emb)], dim=-1)
        h = self.eh_proj(combined)                  # (B, T, 2C) -> (B, T, C)
        return self.block(h, freqs, attention_mask)
```

The embedding layer and `lm_head` are shared with the main model, so each extra depth costs only one projection plus one block. At training time we chain the depths so that depth $k$ predicts the token $k$ steps ahead, always conditioning on the *true* intermediate tokens to keep the causal chain honest,

```python
def mtp_loss(model, embed, lm_head, mtp_modules, tokens, freqs, attention_mask):
    h = model(tokens)                               # main-model hidden states
    losses = [shifted_ce(lm_head(h), tokens, shift=1)]

    for k, module in enumerate(mtp_modules, start=2):
        token_emb = embed(tokens)                   # embeddings of the true tokens
        h = module(h, token_emb, freqs, attention_mask)   # build on the previous depth
        losses.append(shifted_ce(lm_head(h), tokens, shift=k))

    return torch.stack(losses).mean()
```

where `shifted_ce(logits, tokens, shift=k)` is just a cross-entropy that lines up each position’s logits with the token $k$ steps to its right. Because the modules are sequential rather than parallel, the prediction of $t+2$ never gets to peek at the ground-truth $t+1$; it only sees the representation that was itself trying to predict $t+1$.

At inference you can throw the MTP heads away and lose nothing (the main model is untouched), or you can keep them as a built-in draft model: run the heads to propose the next few tokens, then verify them in a single pass of the main model. That’s exactly speculative decoding with the model serving as its own drafter, and the accept/verify mechanics that make it provably lossless are the subject of the from-scratch post linked above.

# Wrapping Up

Stepping back, the three models tell a pretty clean story: each version keeps the previous skeleton and changes exactly the thing that had become the bottleneck.

- **V1** is the “sane LLaMA” baseline, a Pre-Norm transformer with RMSNorm, SwiGLU, RoPE, and GQA. Nothing exotic; it’s the reference point everything else is measured against.
- **V2** attacks the two big serving costs. **Multi-Latent Attention** compresses the KV cache down to a small latent so memory stops being the bottleneck, without the quality hit that GQA pays for sharing heads. **DeepSeekMoE** (fine-grained plus shared experts, with the three balancing losses) buys a huge parameter count at a fixed activated-compute budget, and **YaRN** stretches the context window after the fact.
- **V3** keeps MLA and DeepSeekMoE and files down their rough edges: the **auxiliary-loss-free gate** recovers load balancing without a loss fighting the main objective, and **multi-token prediction** densifies the training signal while doubling as a built-in draft model for speculative decoding.

| Version | Headline change | Bottleneck it removes |
| --- | --- | --- |
| V1 | LLaMA-style backbone (RMSNorm, SwiGLU, RoPE, GQA) | none, it’s the baseline |
| V2 | Multi-Latent Attention + DeepSeekMoE + YaRN | KV-cache memory, capacity-vs-compute, context length |
| V3 | Auxiliary-loss-free balancing + Multi-Token Prediction | balance-vs-quality tradeoff, sparse training signal |

What we deliberately haven’t touched is everything that happens *after* the architecture: the pretraining itself, and the post-training RL that turns a base model into a reasoner like R1. GRPO, which we previewed in the V2 section, is the heart of that second story, and I’ll give it the from-scratch treatment it deserves in a follow-up. If you want to actually run something in the meantime, the V2-Lite code earlier in this post loads the real released weights and generates text end to end; the V3 pieces here are meant to be read alongside it.

# References

1. <a id="ref-deepseek-v1"></a>DeepSeek-AI. "DeepSeek LLM: Scaling Open-Source Language Models with Longtermism." 2024. arXiv:2401.02954. [[link]](https://arxiv.org/abs/2401.02954) — The V1 report whose architecture (RoPE, GQA, RMSNorm, SwiGLU, LLaMA-style backbone) anchors the first half of this post.
2. <a id="ref-llama"></a>H. Touvron et al. "LLaMA: Open and Efficient Foundation Language Models" (arXiv:2302.13971) and "Llama 2: Open Foundation and Fine-Tuned Chat Models" (arXiv:2307.09288). 2023. [[link]](https://arxiv.org/abs/2302.13971) — The micro-design DeepSeek-V1 largely follows.
3. <a id="ref-rmsnorm"></a>B. Zhang and R. Sennrich. "Root Mean Square Layer Normalization." *NeurIPS*, 2019. arXiv:1910.07467. [[link]](https://arxiv.org/abs/1910.07467)
4. <a id="ref-swiglu"></a>N. Shazeer. "GLU Variants Improve Transformer." 2020. arXiv:2002.05202. [[link]](https://arxiv.org/abs/2002.05202) — Source of the SwiGLU activation used in the FFN.
5. <a id="ref-rope"></a>J. Su, Y. Lu, S. Pan, A. Murtadha, B. Wen, and Y. Liu. "RoFormer: Enhanced Transformer with Rotary Position Embedding." 2021 (updated 2024). arXiv:2104.09864. [[link]](https://arxiv.org/abs/2104.09864) — The rotary positional encoding derived in the RoPE section.
6. <a id="ref-gqa"></a>J. Ainslie, J. Lee-Thorp, M. de Jong, Y. Zemlyanskiy, F. Lebrón, and S. Sanghai. "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints." *EMNLP*, 2023. arXiv:2305.13245. [[link]](https://arxiv.org/abs/2305.13245)
7. <a id="ref-deepseek-v2"></a>DeepSeek-AI. "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model." 2024. arXiv:2405.04434. [[link]](https://arxiv.org/abs/2405.04434) — Introduces Multi-Latent Attention and the device/communication balancing losses covered here.
8. <a id="ref-deepseekmoe"></a>D. Dai et al. "DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models." 2024. arXiv:2401.06066. [[link]](https://arxiv.org/abs/2401.06066) — Fine-grained experts plus shared experts, the MoE design used from V2 onward.
9. <a id="ref-yarn"></a>B. Peng, J. Quesnelle, H. Fan, and E. Shippole. "YaRN: Efficient Context Window Extension of Large Language Models." 2023. arXiv:2309.00071. [[link]](https://arxiv.org/abs/2309.00071) — The NTK-by-parts interpolation and attention temperature trick.
10. <a id="ref-grpo"></a>Z. Shao et al. "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models." 2024. arXiv:2402.03300. [[link]](https://arxiv.org/abs/2402.03300) — Introduces Group Relative Policy Optimization (GRPO).
11. <a id="ref-deepseek-v3"></a>DeepSeek-AI. "DeepSeek-V3 Technical Report." 2024. arXiv:2412.19437. [[link]](https://arxiv.org/abs/2412.19437) — The V3 report covering auxiliary-loss-free balancing and multi-token prediction.
12. <a id="ref-auxfree"></a>L. Wang et al. "Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts." 2024. arXiv:2408.15664. [[link]](https://arxiv.org/abs/2408.15664)
13. <a id="ref-mtp"></a>F. Gloeckle, B. Youbi Idrissi, B. Rozière, D. Lopez-Paz, and G. Synnaeve. "Better & Faster Large Language Models via Multi-token Prediction." *ICML*, 2024. arXiv:2404.19737. [[link]](https://arxiv.org/abs/2404.19737)
