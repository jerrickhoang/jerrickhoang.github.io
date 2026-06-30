---
layout: post
title: "Diffusion Language Models"
date: 2026-06-20
description: "A low-prior-knowledge introduction to diffusion and flow matching, then a look at how these continuous generative methods get adapted to discrete language, with implementation and experiments to weigh the pros and cons."
category: "Machine Learning"
tags: ["diffusion", "flow-matching", "language-models", "llm", "generative-models"]
---

## Background

Recently, Google released Gemini Diffusion, which got me looking more into diffusion methods for language models. Diffusion methods have been prevalent in Robotics as well as image / video generation so the idea of applying diffusion to language is not new, I just never got time to look more into this until now. In this post, I will talk through what is diffusion / flow-matching, as well as talk through a few notable ways of applying diffusion to language, then walk through implementation and real experiments to assess pros and cons. Let’s dive in.

First, let’s understand at a high level what diffusion is. I wrote a blog post on the probability estimation of diffusion about a year ago but it was more dense and required prior knowledge, for this blog post I want to introduce these concepts with a lower prior knowledge baseline.

## Diffusion

The whole idea behind diffusion is quite simple once you strip away the math: generating a sample is hard, but *destroying* one is trivial. So instead of trying to learn how to go from nothing to a clean image (or sentence) in one shot, we learn how to undo a tiny bit of damage, and then we apply that little skill over and over again.

Concretely, take a real data point $x_0$, say an image, and slowly corrupt it by mixing in Gaussian noise. After one step it’s slightly grainy, after a few hundred steps it’s indistinguishable from pure static. This is the *forward process*, and the nice thing is that it’s fixed and known, there’s nothing to learn here. We’re just defining a schedule that walks $x_0$ all the way out to a $\mathcal{N}(0, I)$ blob,

$$x_t = \alpha(t)\, x_0 + \beta(t)\, \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I),$$

where $t$ runs from “clean data” to “pure noise” and $\alpha, \beta$ are just the knobs that say how much signal vs. noise survives at time $t$. This rule, the function that sets how much corruption to apply at each $t$, is called the *schedule*, and it is a fixed design choice, not something we learn. You’ll often see the same process written one step at a time instead, as a transition kernel that adds a sliver of noise and shrinks the signal by a hair,

$$q(x_t \mid x_{t-1}) = \mathcal{N}\!\big(\sqrt{1-\beta_t}\,x_{t-1},\; \beta_t I\big),$$

where $\beta_t$ is a small per-step variance. These two views are the same process at different zoom levels, and the bridge between them is where the $\sqrt{\cdot}$ factors come from. Because a Gaussian step composed with another Gaussian step is again Gaussian, you can chain $t$ of these and collapse the whole chain into a single jump. Define the surviving-signal fraction $\bar\alpha_t = \prod_{s=1}^{t}(1-\beta_s)$; then unrolling the recursion gives the closed form

$$x_t = \sqrt{\bar\alpha_t}\;x_0 + \sqrt{1-\bar\alpha_t}\;\varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I),$$

which is exactly the marginal above with $\alpha(t) = \sqrt{\bar\alpha_t}$ and $\beta(t) = \sqrt{1-\bar\alpha_t}$ (and note the two coefficients satisfy $\alpha(t)^2 + \beta(t)^2 = 1$, so this particular schedule keeps the total variance fixed, the *variance-preserving* convention). The per-step kernel is the right object when you reason about the reverse process, which also moves one step at a time; the marginal is the convenient one for training, where we want to jump straight to noise level $t$ in a single draw rather than simulating $t$ tiny steps.

```python
def q_sample(x0, t, alpha, beta):
    """Draw a noised sample x_t ~ q(x_t | x_0) from the closed-form marginal.

    x0    : (B, C, H, W)   a batch of clean images
    t     : (B,)           one timestep index per sample
    alpha : (T,)           schedule, alpha[t] = sqrt(alpha_bar_t)      (signal fraction)
    beta  : (T,)           schedule, beta[t]  = sqrt(1 - alpha_bar_t)  (noise fraction)
    """
    eps = torch.randn_like(x0)         # (B, C, H, W)  standard Gaussian noise
    a = alpha[t].view(-1, 1, 1, 1)     # (B, 1, 1, 1)  broadcast the per-sample scalar over C,H,W
    b = beta[t].view(-1, 1, 1, 1)      # (B, 1, 1, 1)
    x_t = a * x0 + b * eps             # (B, C, H, W)  x_t = alpha*x0 + beta*eps
    return x_t, eps                    # return eps too: it is the training target below

# Example of how the foward and backward process are integrated together.
def train(model, data_loader, alpha, beta, opt):
    T = alpha.shape[0]                              # number of noise levels in the schedule
    for x0 in data_loader:                          # x0: (B, C, H, W)  a batch of clean images
        t = torch.randint(0, T, (x0.shape[0],))     # (B,)  one random noise level per image
        x_t, eps = q_sample(x0, t, alpha, beta)     # forward draw: both (B, C, H, W)
        eps_hat = model(x_t, t)                     # (B, C, H, W)  network predicts the noise
        loss = F.mse_loss(eps_hat, eps)             # scalar: MSE between predicted and true noise
        opt.zero_grad()
        loss.backward()
        opt.step()
```

The interesting part is the *reverse process*. If we could learn to take a slightly-noisier $x_t$ and produce a slightly-cleaner $x_{t-1}$, then we could start from pure static, apply that denoising step a few hundred times, and arrive at something that looks like real data. The bet diffusion makes is that this per-step problem is *easy* even though the end-to-end problem (static → photo in one jump) is wildly hard. Each step only has to remove a little noise, and a neural network is perfectly happy to learn that.

![Diffusion as a forward and reverse process. Top: forward diffusion gradually adds small amounts of Gaussian noise to a clean image $x_0$ over steps $t = 0 \ldots T$ until it becomes $\mathcal{N}(0, I)$ noise, with per-step kernel $q(x_t \mid x_{t-1}) = \mathcal{N}(\sqrt{1-\beta_t}\,x_{t-1}, \beta_t I)$. Bottom: reverse diffusion learns $p_\theta(x_{t-1} \mid x_t)$ (predicting the noise / denoising direction) to walk from noise back to a clean sample.](/assets/images/diffusion-llm/diffusion.png)

Let's work through the reverse process together starting with the forward equation,

$$x_t = \alpha(t)\, x_0 + \beta(t)\, \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, I),$$

First observation is if we hold $x_t$ fixed and this is just a straight line relating $x_0$ and $\varepsilon$, so either one determines the other,

$$\varepsilon = \frac{x_t - \alpha\, x_0}{\beta}, \qquad x_0 = \frac{x_t - \beta\, \varepsilon}{\alpha}.$$

A third quantity hides in the same equation: the *score* of the noised distribution, $\nabla_{x_t} \log p(x_t)$. To see where it comes from, start with the one distribution we know exactly, the forward conditional, which is Gaussian by construction,

$$p(x_t \mid x_0) = \mathcal{N}\!\big(x_t;\, \alpha\, x_0,\; \beta^2 I\big).$$

The gradient of its log is immediate,

$$\nabla_{x_t} \log p(x_t \mid x_0) = -\,\frac{x_t - \alpha\, x_0}{\beta^2} = -\,\frac{\varepsilon}{\beta},$$

where the last step just reuses $\varepsilon = (x_t - \alpha\, x_0)/\beta$ from above. The score we actually want is for the *marginal* $p(x_t) = \int p(x_t \mid x_0)\, p(x_0)\, dx_0$. Differentiating under the integral and dividing through by $p(x_t)$ turns the marginal score into a posterior average of the conditional one,

$$\nabla_{x_t} \log p(x_t) = \mathbb{E}_{x_0 \sim p(x_0 \mid x_t)}\!\big[\nabla_{x_t} \log p(x_t \mid x_0)\big] = -\,\frac{\mathbb{E}[\varepsilon \mid x_t]}{\beta(t)}.$$

In words: the score points along the *average* noise direction over all the clean images that could plausibly have produced this $x_t$, scaled by $-1/\beta(t)$. So a trained noise predictor $\varepsilon_\theta(x_t, t) \approx \mathbb{E}[\varepsilon \mid x_t]$ is, up to that constant, already a score model.

Take the posterior mean of $\varepsilon = (x_t - \alpha\, x_0)/\beta$, which gives $\mathbb{E}[\varepsilon \mid x_t] = (x_t - \alpha\, \mathbb{E}[x_0 \mid x_t])/\beta$, substitute it into the score identity, and solve for the clean-data posterior mean,

$$\mathbb{E}[x_0 \mid x_t] = \frac{x_t + \beta(t)^2\, \nabla_{x_t} \log p(x_t)}{\alpha(t)}.$$

This closes the triangle: noise, score, and clean data are three readings of the same underlying estimate $\mathbb{E}[\,\cdot \mid x_t]$, swapped between by the schedule constants.

So $\varepsilon$, $x_0$, and the score are not three different things to learn, they are three coordinates on the same point, related by the fixed, known schedule constants $\alpha(t), \beta(t)$. A network that predicts any one of them implicitly predicts the other two. That is why the literature looks more fragmented than it is: the three "parameterizations" are the same object wearing different hats, and the choice between them changes the loss scaling and numerical conditioning, not what is being learned.

What actually differs is the *question* you put to the network:

- **Predict the noise $\varepsilon$** (the DDPM ([Ho et al., 2020](#ref-ddpm)) view): "of the stuff I'm looking at, which part is the Gaussian junk that got added?" This is the most common choice in practice, because $\varepsilon$ is a unit-variance Gaussian at *every* noise level, so the regression targets stay nicely scaled and the loss behaves the same across $t$.
- **Predict the clean data $x_0$**: "given this corrupted thing, what was the original?" The most intuitive target, and better conditioned at high noise (near $t = 1$ there is almost no signal left, so asking for the whole $x_0$ is easier than asking for a tiny residual).
- **Predict the score $\nabla_{x_t}\log p(x_t)$**: "which way is uphill toward real data?" The most theoretically central, because it is exactly what the reverse-time SDE and the probability-flow ODE consume in order to run.

The score view is the one I find most intuitive: at every noise level the model is learning a vector field that points "uphill toward real data," and sampling is repeatedly stepping along it while peeling off noise.

**In code.** Because the three are related by schedule constants, you pick one head, train it, and convert to whatever sampling needs. The common choice is the $\varepsilon$ head with a plain MSE against the noise that `q_sample` already handed back,

```python
def diffusion_loss(model, x0, t, alpha, beta):
    # x0: (B, C, H, W)   t: (B,)
    x_t, eps = q_sample(x0, t, alpha, beta)   # forward draw: both (B, C, H, W)
    eps_hat = model(x_t, t)                    # network predicts the noise: (B, C, H, W)
    return F.mse_loss(eps_hat, eps)            # scalar; targets are unit-variance at every t
```

and then read off the other two views with the relations above (all elementwise, broadcasting the per-sample schedule scalars `a`, `b` over the channel and spatial dims),

```python
def eps_to_x0(x_t, eps_hat, a, b):   # invert x_t = a*x0 + b*eps
    return (x_t - b * eps_hat) / a   # (B, C, H, W)  estimate of the clean data

def eps_to_score(eps_hat, b):        # score = -eps / beta
    return -eps_hat / b              # (B, C, H, W)  direction uphill toward real data
```

We have the forward draw and the training loop; the third and final piece is sampling, the reverse process those conversions were built for. Start from pure noise and repeatedly subtract a little of the predicted noise, nudging $x_t$ to a slightly cleaner $x_{t-1}$ until you land back on a sample:

```python
@torch.no_grad()
def sample(model, shape, betas):
    # betas: (T,)  per-step variances; alpha_t = 1 - beta_t, alpha_bar_t = cumprod(alpha)
    alpha = 1.0 - betas                              # (T,)
    alpha_bar = torch.cumprod(alpha, dim=0)          # (T,)
    x = torch.randn(shape)                           # (B, C, H, W)  pure noise x_T
    for t in reversed(range(len(betas))):            # walk t = T-1, ..., 0
        eps_hat = model(x, t)                         # (B, C, H, W)  predicted noise in x_t
        mean = (x - betas[t] / (1 - alpha_bar[t]).sqrt() * eps_hat) / alpha[t].sqrt()
        z = torch.randn_like(x) if t > 0 else 0.0     # inject fresh noise except on the last step
        x = mean + betas[t].sqrt() * z                # (B, C, H, W)  one ancestral step -> x_{t-1}
    return x                                          # (B, C, H, W)  a fresh sample x_0
```

Each step is just the forward relations from above run backwards, with a dash of fresh noise to keep the chain stochastic. Drop that noise term and you are instead integrating the *probability-flow ODE*, the deterministic path that shares the same marginals, which is precisely the bridge to flow matching in the next section.

If you want the careful version of all this, the SDE that the forward process secretly is, the reverse-time SDE, and the probability-flow ODE that shares its marginals, I worked through it in [my earlier post on diffusion probabilities]({% post_url 2025-05-03-probabilities-diffusion %}). For this post the picture above is enough: **diffusion learns a vector field that pushes noise back toward data, one small step at a time.**

## Flow matching

Diffusion as described above is phrased in terms of stochastic noising and denoising, lots of random steps. Flow matching ([Lipman et al., 2022](#ref-fm)) takes the same goal, turn noise into data, and asks a cleaner question: forget the randomness, what if we just learn the *velocity* that transports a noise sample to a data sample along a smooth path?

One heads-up on notation before we start. In the diffusion section above $x_0$ was the *clean data* and time ran toward noise. Flow matching conventionally runs the other way, so for this section $t = 0$ is *noise* and $t = 1$ is *data*. So I’ll write $x_0 \sim \mathcal{N}(0, I)$ for the noise end and $x_1$ for the data end.

Picture a noise point and a data point as the two ends of a path. The simplest path you could draw is a straight line,

$$x_t = (1 - t)\, x_0 + t\, x_1, \quad x_0 \sim \mathcal{N}(0, I), \quad x_1 \sim p_{\text{data}}, \quad t \in [0, 1],$$

with noise at $t = 0$ and data at $t = 1$. If you’re walking along that line, your velocity is just the constant

$$\frac{d x_t}{dt} = x_1 - x_0.$$

![Diffusion and flow matching side by side. Top: DDPM adds noise then learns to denoise. Bottom: flow matching learns a velocity field $v_\theta(x_t, t)$ along a straight path between noise at $t=0$ and data at $t=1$; it is trained to match the conditional target velocity $x_1 - x_0$ on the linear interpolant $x_t = (1-t)x_0 + t x_1$, and sampled by integrating the ODE $dx/dt = v_\theta(x_t, t)$ from $t=0$ to $t=1$. One slip to ignore: the figure's training box writes the endpoints the other way round ($x_0 \sim p_{\text{data}}$, $x_1 \sim \mathcal{N}(0,I)$) — read $x_0$ as the noise end and $x_1$ as the data end, as everywhere else in the figure and the text.](/assets/images/diffusion-llm/flowmatching.png)

That’s the whole trick. We define a target velocity field $u_t(x_t) = x_1 - x_0$ that, at each point and time, says “which direction and how fast should I move to get from noise to data,” and we train a network $v_\theta(x_t, t)$ to regress onto it with a plain mean-squared-error loss,

$$\mathbb{E}_{t,\, x_0,\, x_1}\left[\big\lVert v_\theta(x_t, t) - u_t(x_t) \big\rVert^2\right].$$

Concretely, a single training step looks like this:

1. **Sample the two endpoints.** Draw a real data point $x_1 \sim p_{\text{data}}$ and a noise point $x_0 \sim \mathcal{N}(0, I)$. These are the two ends of one path.
2. **Pick a time on the path.** Draw $t \sim U[0, 1]$, where $t = 0$ is the noise end and $t = 1$ is the data end.
3. **Interpolate.** Place a point on the straight line between them, $x_t = (1 - t)\,x_0 + t\,x_1$. This, and *only* this, is what the network gets to see.
4. **Form the target velocity.** The direction from noise to data is constant along the line, so $u_t = x_1 - x_0$ (this is literally $\tfrac{d x_t}{dt}$).
5. **Predict and compare.** Run the network for its guess $v_\theta(x_t, t)$ and nudge it toward the target with the squared error $\lVert v_\theta(x_t, t) - u_t \rVert^2$.
6. **Repeat** over many independent draws of $(x_0, x_1, t)$.

Visually, one such path is just a straight shot from the noise end to the data end, and step 3 drops the network somewhere along it:

```
t:      0.0         0.25         0.5         0.75         1.0
        x0 ●────────────●────────────●────────────●────────────● x1
      (noise)                       x_t                       (data)
                            velocity u_t = x1 - x0  ───────────▶
```

At this point a natural objection is: isn't this just linear interpolation from noise to data? Yes and no, and the difference is the whole reason flow matching works. *Yes*, in that each individual training pair $(x_0, x_1)$ is joined by a straight line and the target velocity along it is the constant $x_1 - x_0$, a perfectly linear flow. *No*, in that the model never sees a matched pair. At training time it is handed a single point $x_t$, drawn from one $(x_0, x_1)$ pair, and asked for the velocity *there*, with no clue which endpoints produced it. Many different noise→data pairs pass through the neighbourhood of any given $x_t$, each tugging in a different straight-line direction, so the best the network can do is predict their *average*. Averaging a sheaf of crossing straight lines yields a smooth, curved field, the marginal velocity, and that curvature is exactly what lets a single model carry the whole noise cloud onto the whole data distribution.

![Flow matching as transport from a noise distribution (left) to a data distribution (right). Each faint line is one sampled (noise, data) pair joined by a straight conditional path, and the two bold lines are individual examples that happen to cross. Because many such paths pass through any given region, the velocity the network learns there is the average over all of them, so the marginal field it integrates at sampling time is smooth and curved even though every individual training target is a straight-line velocity.](/assets/images/diffusion-llm/flowmatching-illustration.png)

This is also what makes the objective tractable. That marginal velocity is an intractable average over all the paths through $x_t$, but we never have to form it: regressing against the *conditional* velocity $x_1 - x_0$ for a single sampled pair gives the same gradient in expectation as regressing against the intractable marginal field. So training is just: sample noise $x_0$, sample data $x_1$, pick a time $t \sim U[0, 1]$, interpolate to get $x_t$, and ask the network to predict $x_1 - x_0$. In code it is as short as it sounds:

```python
def flow_matching_loss(model, x1, eps):
    # x1: (B, ...) a data sample;  eps: (B, ...) ~ N(0, I) noise  (the two path endpoints)
    t = torch.rand(x1.shape[0])                  # (B,)  flow time ~ U[0,1]: 0 = noise, 1 = data
    tv = t.view(-1, *([1] * (x1.dim() - 1)))     # (B, 1, ...)  broadcast t over the feature dims
    x_t = (1 - tv) * eps + tv * x1               # (B, ...)  a point on the straight noise->data path
    v_target = x1 - eps                           # (B, ...)  the constant conditional velocity
    v_pred = model(x_t, t)                        # (B, ...)  the network's velocity at x_t
    return F.mse_loss(v_pred, v_target)           # scalar L2
```

Once trained, generation is an ODE rather than a noisy walk: start from a pure-noise sample $x_0 \sim \mathcal{N}(0, I)$ and integrate the learned velocity forward from $t = 0$ to $t = 1$,

$$\frac{d x_t}{dt} = v_\theta(x_t, t),$$

with any ODE solver you like (Euler, Heun, a dedicated DPM-solver). The trajectory ends at a data sample $x_1$. Because the paths are close to straight, you can often get away with far fewer steps than classical diffusion needs.

It’s worth saying clearly that diffusion and flow matching are not rival ideas, they’re the same family seen from two angles. Both learn a vector field that connects a simple noise distribution to the data distribution; diffusion arrives at it through a stochastic process and a score, flow matching arrives at it through a deterministic path and a velocity. In fact the probability-flow ODE of a diffusion model *is* a flow, and the standard diffusion schedule is just one particular choice of path. I’ll lean on the flow-matching framing for most of this post because it’s the cleaner mental model, and it’s the one most of the recent diffusion-for-language work is built on.

With that groundwork in place, let’s look at what actually changes when the data isn’t pixels but discrete tokens.

## The problem with discrete data

Everything above, diffusion and flow matching alike, leans on one move that quietly does a lot of work: treating the data as a point in a continuous vector space, so that "a little noisier" (diffusion) or "a step along a path" (flow matching) is even a well defined thing. That makes sense for an image, where a pixel is a real number and you can always nudge it by a fraction. It makes no sense for a token. A token is an index into a vocabulary, token 4123 is not "close to" token 4124 in any meaningful way, and there is no such thing as $0.3$ of the word "cat". The whole continuous machinery, the interpolation $\alpha(t) x_0 + \beta(t)\varepsilon$, the score $\nabla_x \log p(x_t)$, the velocity field, assumes a space where you can take small steps in any direction. A vocabulary has no such geometry.

So if we want to keep the diffusion recipe, the part we have to redesign is the forward process. We need a way to "corrupt" a sequence of tokens gradually, in a way that has a known, easy-to-sample forward direction and a learnable reverse. There are two answers that have stuck, and they correspond to two different ways of asking "what is the discrete version of noise?"

The first says: noise is *forgetting*. Replace tokens, one by one, with a special `[MASK]` symbol until the whole sequence is masked, then learn to fill them back in. This is masked or absorbing-state diffusion, first formalized for discrete state spaces by D3PM ([Austin et al., 2021](#ref-d3pm)), and the cleanest modern version is LLaDA ([Nie et al., 2025](#ref-llada)). The second says: keep the continuous-time, score-based view but rebuild it directly on the discrete state space, modelling the *ratios* between neighbouring states instead of a gradient. This is Score Entropy Discrete Diffusion, SEDD ([Lou et al., 2024](#ref-sedd)). I implemented both, plus a plain autoregressive baseline so we have something honest to compare against, and the rest of the post walks through the implementation and what the experiments actually showed.

A quick note on the setup before the code. Everything below is one small project: a single ~124M parameter transformer backbone shared across all three models, the GPT-2 byte-level BPE tokenizer, and TinyStories ([Eldan and Li, 2023](#ref-tinystories)) as the dataset, which is a corpus of very simple synthetic children's stories. I know Karapathy recommends Tiny Shakespare but I find it hard to qualitatively judge the model's coherence. For the three model variants, they use the same backbone, same tokenizer, same data, same optimizer. The only things that change between the three are the attention mask and the training objective. I do this so comparison is apples to apples.

## LLaDA: noise is masking

LLaDA uses masked diffusion which is the same idea as BERT ([Devlin et al., 2019](#ref-bert)). Pick a noise level $t \in (0, 1)$, and mask each token independently with probability $p_{\text{mask}}(t)$. At $t$ near 0 almost nothing is masked, at $t$ near 1 almost everything is. The reverse process is a network that looks at the partially masked sequence and predicts the original tokens at the masked positions. Train that, and to generate you start from an all-`[MASK]` sequence and unmask your way to a real one.

### The forward process: masking on a schedule

The forward process is about as simple as code gets. Draw one masking probability per sequence, then flip a coin per token:

```python
def _forward_process(self, input_ids):
    """Return (noisy_ids, masked_bool, p_mask) for the LLaDA forward step."""
    b, length = input_ids.shape
    t = torch.rand(b, device=input_ids.device)
    p_mask = (1 - self.eps) * t + self.eps
    p_mask = p_mask[:, None].expand(b, length)
    masked = torch.rand((b, length), device=input_ids.device) < p_mask
    noisy = torch.where(masked, self.mask_id, input_ids)
    return noisy, masked, p_mask
```

Because the coin flips are independent, the forward marginal factorizes over positions, $q(x_t \mid x_0) = \prod_i q(x_t^i \mid x_0^i)$, with each token either copied through or replaced by `[MASK]` with probability $p_{\text{mask}}(t)$. This is the *same* absorbing forward process we will meet again in SEDD; the only difference is the schedule, the rule for how fast $p_{\text{mask}}(t)$ ramps from roughly 0 (nothing masked) at $t = 0$ to 1 (everything masked) at $t = 1$. Here it is linear, $p_{\text{mask}}(t) = (1-\epsilon)t + \epsilon \approx t$, where the small floor $\epsilon$ just keeps the probability above zero (the loss below divides by it). SEDD's absorbing graph uses a different curve for the same job; we will see its exact form later. Same coin, different dial.

### Deriving the loss from the variational bound

The loss is cross-entropy on the masked positions only, which again is exactly the BERT masked-language-modelling loss. BERT picks a fixed fraction of positions to mask (about 15%) and averages the cross-entropy over them,

$$\mathcal{L}_{\text{BERT}} = \mathbb{E}\!\left[\frac{1}{|\mathcal{M}|}\sum_{i \in \mathcal{M}} -\log p_\theta\big(x_0^i \mid x_t\big)\right],$$

where $\mathcal{M}$ is the set of masked positions. LLaDA's loss looks almost identical, with two changes. The masking level $t$ is drawn fresh from $U[0,1]$ every step, so the model is trained across the whole range from lightly masked to almost fully masked, and each masked token's cross-entropy is divided by $p_{\text{mask}}(t)$,

$$\mathcal{L}_{\text{LLaDA}} = \mathbb{E}_{t \sim U[0,1]}\;\mathbb{E}_{x_t \sim q(\cdot\mid x_0)}\!\left[\frac{1}{p_{\text{mask}}(t)}\sum_{i:\, x_t^i = \texttt{[MASK]}} -\log p_\theta\big(x_0^i \mid x_t\big)\right].$$

That $1/p_{\text{mask}}(t)$ factor is the one twist that turns BERT into a proper diffusion bound. It is not a heuristic, it falls straight out of the likelihood bound, and it is worth seeing why.

Earlier we trained continuous diffusion with a plain MSE on the noise and never said where that loss came from. It is really a simplified, reweighted form of a deeper objective: a variational lower bound (an ELBO) on the data log-likelihood $\log p_\theta(x_0)$. For LLaDA it is cleanest to start from that bound directly, written as the usual sum of per-step KL terms,

$$-\log p_\theta(x_0) \le \mathbb{E}_q\Big[\underbrace{D_{\mathrm{KL}}\big(q(x_T\mid x_0)\,\|\,p(x_T)\big)}_{\text{prior}} + \sum_{t=2}^{T} D_{\mathrm{KL}}\big(q(x_{t-1}\mid x_t, x_0)\,\|\,p_\theta(x_{t-1}\mid x_t)\big) \;-\; \log p_\theta(x_0\mid x_1)\Big].$$

Two facts make this collapse to something simple. First, the forward process masks each position independently, so the whole bound factorizes over positions and we can derive it for a single token and sum at the end. Second, masking is *absorbing*, which makes the per-step posterior almost trivial. Work in discrete time with $T$ steps and let $\bar\alpha_t$ be the probability a token is still unmasked at step $t$ (so $\bar\alpha_0 = 1$, decreasing toward $\bar\alpha_T \approx 0$, with $p_{\text{mask}} = 1 - \bar\alpha_t$).

**The forward posterior.** A single token is in one of two states: its true value $x_0$, or $\texttt{[MASK]}$. Since a masked token stays masked, the posterior $q(x_{t-1}\mid x_t, x_0)$ is trivial in one case and a two-way split in the other:

- if $x_t$ is *unmasked*, it must have been unmasked at $t-1$ too, so $x_{t-1} = x_0$ with certainty;
- if $x_t = \texttt{[MASK]}$, then $x_{t-1}$ was either already masked or still held the true token and got masked on this step. Bayes gives the split

$$q(x_{t-1} = x_0 \mid x_t = \texttt{m}, x_0) = \frac{\bar\alpha_{t-1} - \bar\alpha_t}{1 - \bar\alpha_t}, \qquad q(x_{t-1} = \texttt{m} \mid x_t = \texttt{m}, x_0) = \frac{1 - \bar\alpha_{t-1}}{1 - \bar\alpha_t}.$$

**The reverse model.** We parametrize $p_\theta$ by plugging the network's predicted distribution over the clean token, $p_\theta(x_0 \mid x_t)$, into that *same* posterior shape. So it reproduces $q$'s "stay masked" probability $\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}$ exactly, and spreads the remaining reveal mass $\frac{\bar\alpha_{t-1}-\bar\alpha_t}{1-\bar\alpha_t}$ over token values according to $p_\theta(\cdot \mid x_t)$ instead of dumping it all on the true $x_0$.

**One KL term.** Unmasked positions contribute nothing: there $q$ and $p_\theta$ are both the point mass on $x_0$, so the KL is zero. For a masked position the two distributions agree on the mask atom and differ only on the reveal branch, and the KL of two distributions sharing an atom keeps only that branch,

$$D_{\mathrm{KL}}\big(q \,\|\, p_\theta\big) = \frac{\bar\alpha_{t-1} - \bar\alpha_t}{1 - \bar\alpha_t}\,\big(-\log p_\theta(x_0 \mid x_t)\big).$$

Each step's KL is just a *scaled cross-entropy* of the true token under the denoiser.

**Sum the steps.** Now take the expectation over the forward process. A token is masked at step $t$ with probability $1 - \bar\alpha_t$, and the KL above is conditioned on exactly that event, so the $1 - \bar\alpha_t$ cancels,

$$\mathbb{E}_{x_t}\big[\text{KL}_t\big] = (\bar\alpha_{t-1} - \bar\alpha_t)\,\mathbb{E}\big[-\log p_\theta(x_0 \mid x_t) \,\big|\, \text{masked}\big].$$

The sum $\sum_t (\bar\alpha_{t-1} - \bar\alpha_t)(\cdots)$ is a Riemann sum of $-\bar\alpha_t'$; letting $T \to \infty$ and writing the continuous survival as $\alpha_t$ turns it into an integral. Re-expressing the per-token conditional expectation as an unconditional sum over the random masked set puts back a factor $\frac{1}{1-\alpha_t}$ (since a position is in that set with probability $1-\alpha_t$), and summing over all positions gives the bound

$$-\log p_\theta(x_0) \;\le\; \mathbb{E}_{t\sim U[0,1]}\left[\frac{-\alpha_t'}{1-\alpha_t}\; \mathbb{E}_{x_t \sim q(\cdot\mid x_0)} \sum_{i:\, x_t^i = \texttt{[MASK]}} \big(-\log p_\theta(x_0^i \mid x_t)\big)\right].$$

(The prior term vanishes because at $t = 1$ both $q$ and the prior put all their mass on the fully-masked sequence, and the reconstruction term $-\log p_\theta(x_0 \mid x_1)$ folds into the same integral.) Only the masked positions appear, because the unmasked ones carried zero KL, and the weight $\frac{-\alpha_t'}{1-\alpha_t}$ is exactly what fell out: $-\alpha_t'$ is the rate at which tokens are freshly masked at time $t$, and $\frac{1}{1-\alpha_t}$ undoes the probability that the position we are summing over was masked at all. This matches the absorbing-state result of D3PM ([Austin et al., 2021](#ref-d3pm)) and the LLaDA ([Nie et al., 2025](#ref-llada)) objective.

Now specialize to LLaDA's linear schedule $p_{\text{mask}}(t) = t$, i.e. $\alpha_t = 1 - t$ and $\alpha_t' = -1$. The weight becomes

$$\frac{-\alpha_t'}{1-\alpha_t} = \frac{1}{t} = \frac{1}{p_{\text{mask}}(t)},$$

so the bound is exactly a $1/p_{\text{mask}}$-weighted sum of cross-entropies over the masked positions, with $t$ drawn uniformly. That is precisely the one line `F.cross_entropy(...) / p_mask[masked]` below:

```python
def training_loss(self, model, input_ids):
    b, length = input_ids.shape
    noisy, masked, p_mask = self._forward_process(input_ids)
    logits = model(noisy)
    token_loss = (
        F.cross_entropy(logits[masked], input_ids[masked], reduction="none") / p_mask[masked]
    )
    return token_loss.sum() / (b * length)
```

If you squint, this is the discrete echo of the continuous picture. The all-`[MASK]` state plays the role of pure noise, the clean sequence plays the role of $x_0$, and "predict the original token given a partially masked sequence" is the discrete denoiser. The $1/p_{\text{mask}}$ weighting we just derived is the discrete analog of the schedule-dependent weighting in the continuous ELBO.

There is a second way to read this objective that I find clarifying. Each training step shows the model an arbitrary subset of visible tokens and asks it to predict an arbitrary subset of hidden ones, across every noise level. That is an *any-order* generalization of next-token prediction: where an autoregressive model only ever predicts token $i$ from tokens $< i$, LLaDA learns to predict any masked position from whatever else happens to be visible. The strict left-to-right factorization is just one of the many orderings it is implicitly trained on, which is exactly why it can later infill from both sides (the subject of one of the experiments).

### Sampling: predict, commit, repeat

Sampling is where it gets interesting, and where the differences from autoregressive generation start to show. You begin with every position masked and run a fixed number of reverse steps. At each step you run the model once over the whole sequence, get a predicted token and a confidence for every masked position, and you *commit* the highest-confidence ones, leaving the rest masked for later steps. This is "low-confidence remasking": fill in what you are sure about first, and let those committed tokens inform the harder positions on the next pass.

```python
@torch.no_grad()
def generate(self, model, *, num_samples, seq_len, num_steps, device, temperature=1.0):
    x = torch.full((num_samples, seq_len), self.mask_id, dtype=torch.long, device=device)
    for step in range(num_steps):
        mask_index = x == self.mask_id
        if not mask_index.any():
            break
        logits = model(x)[..., : self.data_vocab_size]  # never emit [MASK]
        if temperature > 0:
            gumbel = -torch.log(-torch.log(torch.rand_like(logits.float()) + 1e-10) + 1e-10)
            x0 = (logits.float() / temperature + gumbel).argmax(dim=-1)
        else:
            x0 = logits.argmax(dim=-1)
        probs = F.softmax(logits.float(), dim=-1)
        confidence = probs.gather(-1, x0[..., None]).squeeze(-1)
        confidence = torch.where(mask_index, confidence, torch.full_like(confidence, -1.0))

        # Reveal a fraction of the masked tokens this step (linear schedule).
        for row in range(num_samples):
            num_masked = int(mask_index[row].sum().item())
            if num_masked == 0:
                continue
            reveal = max(1, num_masked // (num_steps - step))
            topk = torch.topk(confidence[row], k=reveal).indices
            x[row, topk] = x0[row, topk]
    return x
```

Two things to flag here, because both bit me later. First, `num_steps` is a genuine knob. With few steps you commit many tokens per pass and decode fast but sloppily, with many steps you commit a few at a time and decode slowly but more carefully. There is no analog of this dial in autoregressive decoding, and it turns out to be the whole story of one of the experiments. Second, that `temperature` defaulting to a positive value matters a lot. My first version used greedy argmax everywhere, and greedy LLaDA collapses into repeating the same few high-frequency tokens, because from an all-`[MASK]` start the most confident prediction at most positions is "." or "the". The Gumbel-max sampling restores diversity while keeping the confidence-based *ordering* of which positions to reveal.

## SEDD: noise is a jump on a graph

The second approach keeps more of the continuous theory and pays for it in complexity. Where LLaDA throws away the SDE machinery and just masks, SEDD rebuilds the whole score-based framework directly on the discrete state space. It is the more principled of the two, and also the one that gave me the most trouble, so it is worth slowing down.

### The forward process as a continuous-time Markov chain

In continuous space the forward process was an SDE drifting data into a Gaussian. The discrete analog is a *continuous-time Markov chain* (CTMC) over the vocabulary. Instead of a density $p_t(x)$ we track a probability *vector* $p_t \in \mathbb{R}^{V}$ over the $V$ tokens (per position), and instead of an SDE it evolves by a linear ODE,

$$\frac{d p_t}{dt} = \sigma(t)\, Q\, p_t,$$

where $Q \in \mathbb{R}^{V \times V}$ is a fixed *rate matrix* and $\sigma(t)$ is a scalar noise-rate schedule. A rate matrix just collects the instantaneous jump rates: the off-diagonal $Q_{yx} \ge 0$ is the rate of hopping from token $x$ to token $y$, and the diagonal is fixed to $Q_{xx} = -\sum_{y \ne x} Q_{yx}$ so each column sums to zero, which is exactly the condition that keeps $p_t$ normalized. Over a tiny step this is just $p_{t+\Delta t} \approx (I + \sigma(t)\,\Delta t\, Q)\, p_t$: with probability set by the rates you jump, otherwise you stay.

Because $Q$ is fixed and only scaled in time, the chain has a closed-form transition matrix. Writing the *total* (integrated) noise $\bar\sigma(t) = \int_0^t \sigma(s)\,ds$, the conditional of a noised token given the clean one is a matrix exponential,

$$p_{t\mid 0}(\cdot \mid x_0) = \exp\!\big(\bar\sigma(t)\, Q\big)_{:,\,x_0}.$$

This is the discrete echo of "$x_t = x_0 + \sigma\varepsilon$ is known in closed form": we can sample a noised token directly without simulating the chain step by step. In the code this is exactly the split between `total_noise(t)` $= \bar\sigma(t)$ and `rate_noise(t)` $= \sigma(t)$.

### The absorbing graph, and why it is just LLaDA with a schedule

SEDD works for any $Q$, but the one I used (and the one that matters for language) is the *absorbing* graph: every token jumps to a single `[MASK]` state at rate 1, and `[MASK]` never leaves. For that $Q$ the matrix exponential collapses to a one-liner, a token survives unchanged with probability $e^{-\bar\sigma(t)}$ and is otherwise sent to `[MASK]`,

$$p_{t\mid 0}(\text{stay} \mid x_0) = e^{-\bar\sigma(t)}, \qquad p_{t\mid 0}(\texttt{[MASK]} \mid x_0) = 1 - e^{-\bar\sigma(t)}.$$

That is worth pausing on: it is *the same forward process as LLaDA*, independent per-token masking, with the masking probability pinned to a specific schedule $p_{\text{mask}}(t) = 1 - e^{-\bar\sigma(t)}$. `sample_transition` is precisely this coin flip. So the two methods share a forward process and differ entirely in what the network predicts and how the loss is built, which is the whole reason this section is heavier.

### The concrete score and the reverse chain

In continuous diffusion, sampling backward needed the score $\nabla_x \log p_t(x)$. There is no gradient on a vocabulary, but there is a clean replacement. Anderson's reverse-time SDE result ([Anderson, 1982](#ref-anderson)) has a discrete counterpart: the reverse of a CTMC is *also* a CTMC, with rate matrix

$$\bar Q_t(y, x) = \frac{p_t(y)}{p_t(x)}\, Q_t(x, y), \qquad y \ne x.$$

So the only thing we need to run generation backward is, for each state $x$ and each neighbour $y$, the ratio of marginals $\frac{p_t(y)}{p_t(x)}$. SEDD calls the collection of these ratios the *concrete score*, and the network's job is to estimate it,

$$s_\theta(x, t)_y \;\approx\; \frac{p_t(y)}{p_t(x)}.$$

It plays exactly the role $\nabla_x \log p$ did in the continuous case (it is a kind of finite-difference of $\log p$ along the edges of the graph), but it lives on a discrete graph instead of a tangent space.

### Score entropy: a loss you can actually compute

Now the catch, the same one that makes ordinary score matching awkward: the target $\frac{p_t(y)}{p_t(x)}$ involves the intractable marginals $p_t$. The fix mirrors *denoising* score matching exactly. First, the population objective, the *score entropy*, is a Bregman divergence that is minimized precisely when $s_\theta$ hits the true ratio,

$$\mathcal{L}_{\text{SE}} = \mathbb{E}_{x \sim p_t}\sum_{y \ne x} Q_t(x,y)\left[ s_\theta(x)_y - \frac{p_t(y)}{p_t(x)}\log s_\theta(x)_y + K\!\Big(\tfrac{p_t(y)}{p_t(x)}\Big)\right],$$

with $K(a) = a(\log a - 1)$ a normalizer that depends only on the (unknown) target and so drops out of the gradient w.r.t. $\theta$. This still contains the intractable ratio. The trick is to condition on the clean token $x_0$: averaging over $x_0 \sim p_0$ and $x_t \sim p_{t\mid 0}(\cdot\mid x_0)$ turns the marginal ratio into the *conditional* ratio, which we just saw is a closed-form matrix exponential,

$$\mathcal{L}_{\text{DSE}} = \mathbb{E}_{x_0,\; x_t \sim p_{t\mid 0}(\cdot\mid x_0)}\sum_{y \ne x_t} Q_t(x_t, y)\left[ s_\theta(x_t)_y - \frac{p_{t\mid 0}(y\mid x_0)}{p_{t\mid 0}(x_t\mid x_0)}\log s_\theta(x_t)_y + K(\cdot)\right].$$

This *denoising score entropy* has the same minimizer as the intractable version (the conditional ratios average back to the marginal one), but every term is now computable. The backbone emits raw logits, which an output transform turns into the log concrete score $\log s_\theta(x_t, t)$ at noise level $\sigma$; the training loss is this denoising score entropy, summed over the sequence and weighted by the time schedule:

```python
def _per_sequence_loss(self, model, input_ids):
    """Denoising-weighted score entropy summed over the sequence, (B,)."""
    b = input_ids.shape[0]
    t = (1 - self.sampling_eps) * torch.rand(b, device=input_ids.device) + self.sampling_eps
    sigma = self.noise.total_noise(t)
    dsigma = self.noise.rate_noise(t)
    perturbed = self.graph.sample_transition(input_ids, sigma[:, None])
    log_score = self._log_score(model, perturbed, sigma)
    loss = self.graph.score_entropy(log_score, sigma[:, None], perturbed, input_ids)
    return (dsigma[:, None] * loss).sum(dim=-1)
```

It is worth reading that code straight off the math above. `sample_transition(input_ids, sigma)` draws $x_t \sim p_{t\mid 0}(\cdot\mid x_0)$ using the closed-form survival probability $e^{-\bar\sigma}$; `_log_score` is $\log s_\theta$; `graph.score_entropy(...)` assembles the Bregman terms with the *conditional* ratio $p_{t\mid 0}(y\mid x_0)/p_{t\mid 0}(x_t\mid x_0)$ as the target (the part that is tractable only because we conditioned on $x_0$); and the `dsigma` factor is the per-time weight $\sigma(t)$ from the continuous-time likelihood bound. Every expensive piece, scores over the whole vocabulary, the matrix-exponential transition, the per-edge loss, is the cost of staying faithful to the continuous theory, and it is why SEDD's gradient is so much noisier per step than LLaDA's plain cross-entropy.

Notice that the model now conditions on $\sigma$, the continuous noise level. That conditioning has to go somewhere, and the standard answer borrowed from the image-diffusion world (DiT) is adaLN-Zero: regress a shift, scale, and gate for each LayerNorm from an embedding of $\sigma$, and initialize those gates to zero so the network starts as a clean unmodulated transformer. That detail is going to matter in the debugging story below, so hold onto it.

Sampling is a predictor-corrector loop over a linear schedule of noise levels, starting from a fully-absorbed (all-`[MASK]`) sequence and stepping the score backward, with a final denoising step to clean up any residual masks. It is more involved than LLaDA's reveal loop, but conceptually it is the same arc: start from noise, walk back to data.

## One backbone, three modes, and the autoregressive baseline

All three models share a single pre-norm transformer. The only structural switches are whether attention is causal and whether the block LayerNorms are modulated by $\sigma$. The attention module captures the first switch in one line, using PyTorch's fused attention with its `is_causal` flag:

```python
class _SelfAttention(nn.Module):
    def __init__(self, dim, n_heads, dropout, causal=False):
        ...
        self.causal = causal

    def forward(self, x):
        b, length, dim = x.shape
        qkv = self.qkv(x).reshape(b, length, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        out = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        return self.proj(out.transpose(1, 2).reshape(b, length, dim))
```

LLaDA and SEDD run with `causal=False`, full bidirectional attention, because a diffusion model is allowed to look at the whole (partially noised) sequence at once. The autoregressive baseline runs with `causal=True`, so each position only sees itself and the past. That single flag is the entire architectural difference between "diffusion language model" and "GPT".

The baseline objective is the one everyone already knows, next-token cross-entropy, and I want it in the post because it is the thing the diffusion numbers have to be measured against:

```python
class ARObjective(DiffusionObjective):
    def _shifted_ce(self, model, input_ids):
        logits = model(input_ids)[:, :-1, :]   # predict t+1 from <= t
        targets = input_ids[:, 1:]
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

    def training_loss(self, model, input_ids):
        return self._shifted_ce(model, input_ids)

    @torch.no_grad()
    def nll_bound(self, model, input_ids):
        # Exact per-token NLL, so exp(.) is the true perplexity (not a bound).
        return self._shifted_ce(model, input_ids)
```

There is one asymmetry worth stating loudly, because it shapes how you have to read every number later. The AR model's `nll_bound` is *exact*: it is the real per-token negative log-likelihood, so `exp(nll)` is the true perplexity. Both diffusion models' `nll_bound` is a variational *upper bound*, an expectation over random masking or noise levels. They are not the same kind of number, and comparing "AR perplexity 4.4" against "LLaDA perplexity 21" as if AR were five times better is simply wrong. More on this when we get to the experiments.

## A debugging story: the silently broken LLaDA

Here is the part I most wanted to write down, because it is the kind of bug that does not crash, does not warn, and quietly makes your results meaningless.

When I first trained LLaDA on TinyStories, the loss dropped for about twenty-five steps, from roughly 11 down to 5.9, and then sat there. Dead flat. For thousands of steps. Validation perplexity parked itself at about 340 and would not move, and every sample the model produced was punctuation and a couple of stop words, things like "." and "the" and "They" over and over. The training did not error. The metrics did not look insane. It just was not learning anything past the first few dozen steps.

The number 5.9 turned out to be the tell. That is roughly the unigram entropy of the data, the loss you get from predicting the marginal token frequencies and nothing context-dependent.

Where does 5.9 come from? Cross-entropy in nats is $\mathbb{E}_{x \sim p}[-\log q(x)]$, the loss of a model that assigns probability $q(x)$ to the true token $x$. If the model ignores context entirely and predicts the *same* distribution $q$ at every position, the best it can do is set $q$ equal to the data's marginal token distribution $p$, and the loss bottoms out at the *unigram entropy*,

$$H = -\sum_{v} p(v)\,\log p(v),$$

the entropy of the token-frequency table. For the TinyStories tokens that comes out to about 5.9 nats. Two sanity checks bracket this number. At initialization the network is essentially uniform over the roughly 50k GPT-2 vocabulary, so the loss should start near $\log(50257) \approx 10.8$, which is right where it began (~11). And a loss of 5.9 nats is a perplexity of $e^{5.9} \approx 365$, matching the validation perplexity that parked itself around 340. Both numbers say the same thing: the model had collapsed onto the marginal token frequencies and stopped using context.

The model had learned which tokens are common and then stopped. So the question was not "why is the loss high" but "why has the body of the transformer stopped contributing at all".

The way to localize this kind of thing is a single-batch overfit test. Take one fixed batch and try to drive the loss to zero on it. A 124M parameter transformer should crush a single batch trivially. Mine could not, it stalled at the same unigram floor. That rules out the data pipeline and most hyperparameter explanations, and points at something structural. Then I instrumented the gradient norms per parameter group, and that was the smoking gun: at step 0 everything looked healthy, but within about 25 steps the gradients flowing into the transformer blocks had collapsed to around 0.004 while the embedding and output head kept a healthy gradient around 0.5 to 0.9. The body was receiving no learning signal. Only the marginal predictor, the embedding tied to the output head, was still moving, which is exactly how you end up stuck at the unigram distribution.

The root cause was an initialization detail. The standard GPT-2 init ([Radford et al., 2019](#ref-gpt2)) does something easy to forget: it down-scales the *output* projections of each block (the attention output projection and the MLP's second linear) by $1/\sqrt{2 N}$ where $N$ is the number of layers. The reason is that those projections write directly into the residual stream, and with $N$ layers all adding in at full scale, the residual stream gets dominated by accumulated block outputs (or, at init, by the input embedding relative to the under-trained blocks), the final LayerNorm washes out the block contributions, and their gradients vanish. My backbone initialized every linear with the same `std=0.02` and skipped that rescaling. The fix is four lines:

```python
# GPT-2 residual-projection scaling: down-scale each block's attention
# and MLP output projections by 1/sqrt(2 * n_layers). Without it the
# residual stream is dominated by the input embedding, the block
# gradients vanish within a few dozen steps, and the model collapses
# to a unigram predictor (verified by single-batch overfit).
residual_std = 0.02 / math.sqrt(2 * cfg.n_layers)
for block in self.blocks:
    nn.init.normal_(block.attn.proj.weight, mean=0.0, std=residual_std)
    nn.init.normal_(block.mlp.fc2.weight, mean=0.0, std=residual_std)
```

The detail I find genuinely interesting is *why only LLaDA was affected*. SEDD uses adaLN-Zero, and those zero-initialized gates mean every SEDD block starts as the exact identity. The unscaled output projections are never exposed at initialization, the residual stream is clean, and SEDD trains fine without the fix. LLaDA is the plain pre-norm path with no gating, so it adds those badly-scaled projections into the residual stream from step 0, and it is the only one of the three that hits the pathology. One missing line in a shared backbone, and only one of the three models silently breaks. That is the sort of thing that does not show up in a unit test.

With the fix in, LLaDA trains. The single batch overfits to near zero, and on the real data the plateau is gone.

## Do they actually learn?

Same backbone, same recipe (batch 32, sequence length 256, learning rate 3e-4, 8000 steps), measured on the TinyStories validation split:

| Variant | val ppl (start to end) | What the number means |
| --- | --- | --- |
| AR (GPT) | 9.6 to **4.4** | exact perplexity |
| LLaDA | 307.7 to **21.2** | NLL upper bound |
| SEDD | 344.6 to **20.0** | NLL upper bound (6000 steps, effective batch 128) |

![Validation perplexity versus training step for AR, LLaDA, and SEDD on TinyStories, on a log scale. AR sits far below the two diffusion models, but the AR curve is an exact perplexity while the diffusion curves are NLL upper bounds, so the gap is not a like-for-like quality gap.](/assets/images/diffusion-llm/convergence.png)

By a few thousand steps all three produce coherent TinyStories text. Here are unedited samples, one per model, after training:

> **AR (step 8000):** "Lily and Ben are friends. They like to play in the park. One day, they see a big pond with ducks. Lily wants to feed the ducks some bread."
>
> **LLaDA (step 4000):** "Ben are friends. They play in the park. They see the slide and slide. They are happy."
>
> **SEDD (step 6000):** "scary voice and grinned her fur it sounded nice times, Lily said. "Okay, but you can be dirty," Tim was."

None of these will win a prize, but they are clearly English, clearly on-topic, and clearly past the unigram floor. You can also see the quality ordering that the perplexity table only hinted at: the AR sample is the most fluent and varied, LLaDA is a touch more repetitive ("slide and slide", "They ... They ... They"), and SEDD is visibly the roughest of the three, with real named characters and quoted dialogue but shakier grammar. That ordering fits its convergence story: SEDD needed both a larger effective batch (128, via gradient accumulation) and more steps to get here, because its score-entropy objective is much noisier per step than LLaDA's token cross-entropy. For reference, this SEDD sample is from the local checkpoint that finished at validation perplexity 19.9, generated with 128 reverse steps.

It is worth seeing the failure mode too, because it is the qualitative signature of the gradient-collapse bug from earlier. Before the init fix, every LLaDA sample looked roughly like this:

> **LLaDA, pre-fix:** ". . the . . the They . . the . ."

That string is just the unigram distribution talking: the most frequent tokens, in no particular order, with no context. The jump from that to the "Ben are friends" sample above is the entire payoff of those four lines of initialization.

Now, the obvious temptation is to read that table as "AR is five times better than the diffusion models". Resist it. The AR column is an exact likelihood and the diffusion columns are upper bounds on a strictly harder per-token task (predict tokens under random masking or noise, not predict the next token given the true prefix). They are not on the same axis. Which is exactly why, to compare these models fairly, I had to stop looking at perplexity and start measuring things that mean the same thing for all three.

## Three experiments that actually compare apples to apples

The trick that makes a fair comparison possible is to score everything with one neutral judge. For the quality experiments I use a frozen, fully-trained AR model purely as a scorer, and ask: how likely is this generated text under the judge? That number, generative perplexity, means the same thing whether the text came from AR, LLaDA, or SEDD. It sidesteps the upper-bound problem entirely.

### Experiment 1: quality versus inference compute

This is the experiment that shows what the diffusion step-count knob actually buys you. I sweep LLaDA's number of reverse steps, generate samples at each setting, and score them with the frozen AR judge (lower generative perplexity is better). I also report a repetition rate, the fraction of repeated 4-grams, because generative perplexity alone can be gamed by degenerate repetitive text that the judge happens to find predictable. The unit of compute is NFE, the number of network forward passes: for LLaDA that is the step count, for AR it is the sequence length (256), since AR emits one token per forward pass.

| sampler | NFE | gen ppl (lower better) | repeat-4gram |
| --- | --- | --- | --- |
| real held-out text | n/a | 4.18 | 0.034 |
| LLaDA @ 8 steps | 8 | 66.5 | 0.061 |
| LLaDA @ 32 steps | 32 | 15.4 | 0.306 |
| LLaDA @ 128 steps | 128 | 11.1 | 0.468 |
| LLaDA @ 256 steps | 256 | 9.9 | 0.499 |
| LLaDA @ 512 steps | 512 | 9.8 | 0.490 |
| AR (GPT) | 256 | 6.6 | 0.018 |

![Two panels for experiment E1. Left: generative perplexity drops as LLaDA spends more reverse steps, saturating around 256 steps, with the AR point sitting well below the whole LLaDA sweep. Right: the repeated-4gram rate climbs toward 0.5 as steps increase, while AR and real text stay near 0.02 to 0.03.](/assets/images/diffusion-llm/e1_quality_compute.png)

There are two lessons here and they pull in opposite directions. The first is that the diffusion knob is real: more reverse steps genuinely buy better samples, with the gain saturating around 256 steps. In the cheap regime, LLaDA at 8 steps produces a full 256-token passage in 8 forward passes, which is 32 times fewer sequential steps than AR. That is the headline advantage of diffusion decoding. The second lesson is the cautionary one: read the repetition column. As the step count climbs, generative perplexity keeps dropping, but the repetition rate climbs to nearly 0.5, meaning half the 4-grams are repeats, against 0.034 for real text. So the falling perplexity is partly the confidence-based sampler converging onto repetitive, templated text that the judge finds easy, not genuinely better writing. And at matched compute (NFE 256) the AR model is both more fluent (perplexity 6.6) and far more diverse (repetition 0.018). At this scale, for raw quality per unit of compute, autoregressive still wins.

### Experiment 2: latency versus sequence length

This one isolates wall-clock cost, and it is where diffusion's parallelism shows up. An AR model emits one token per forward pass, so it needs `seq_len` *sequential* network evaluations that cannot be parallelized across positions. A diffusion model decodes every position in parallel on each reverse step, so its sequential depth is the step count, independent of how long the sequence is. Timing `generate` across sequence lengths on a single GPU:

| seq_len | AR (ms/seq) | LLaDA @ 32 (ms/seq) | LLaDA @ 128 (ms/seq) |
| --- | --- | --- | --- |
| 128 | 50 | 33 | 131 |
| 256 | 104 | 50 | 200 |
| 512 | 306 | 93 | 372 |
| 1024 | 1090 | 183 | 733 |

![Two panels for experiment E2. Left: AR latency per sequence grows super-linearly with sequence length while LLaDA at a fixed 32 steps stays nearly flat, crossing below AR by length 512. Right: LLaDA at 32 steps gains throughput as length grows while AR throughput falls.](/assets/images/diffusion-llm/e2_latency.png)

The shape is the whole point. AR latency grows super-linearly with length (50 to 1090 ms as length goes 128 to 1024), while LLaDA at a fixed step budget grows roughly linearly and its throughput actually rises as the fixed steps amortize over more tokens. At length 1024, LLaDA at 32 steps is about six times faster per sequence than AR. There is an honest caveat to attach: my AR sampler has no key-value cache, so it recomputes the full context every token and its absolute cost is pessimistic (a production decoder with a KV cache would be much faster). But the conclusion that survives the caveat is the scaling shape, not the absolute numbers. AR's sequential depth is `seq_len` and is irreducible, because each token genuinely needs the previous one, whereas diffusion's sequential depth is a tunable constant. A KV cache speeds up each AR step but does not change the fact that you need `seq_len` of them in sequence.

### Experiment 3: infilling, the thing AR cannot do

The first two experiments are about cost and quality on the standard left-to-right task, where AR is strong. This one is about a capability AR structurally lacks. Because diffusion attends bidirectionally, it can fill a gap in the *middle* of a sequence using context from *both sides*. A causal AR model can only ever condition on the left.

To measure this without confounding it with raw model quality (AR is the stronger model overall, so it would win a naive comparison for the wrong reason), I run the *same* LLaDA model twice: once filling a masked middle span with the suffix visible, and once with the suffix also masked so it only sees the prefix. The difference between those two is purely the value of right context, on identical weights. The core of the conditional fill is just LLaDA's reveal loop restricted to the gap, clamping the known tokens:

```python
x = x_true.clone()
x[:, a:b] = mask_id            # mask the gap [a, b)
if not use_suffix:
    x[:, b:] = mask_id          # ablation: hide the suffix too
# ... then run the same confidence-remasking loop, but only ever
# reveal positions inside [a, b); the prefix (and suffix, if visible)
# stay clamped to their true tokens the whole time.
```

I score the result two ways under the frozen AR judge: the perplexity of the whole reconstruction, and a "boundary" NLL, the likelihood of the *true* suffix given the prefix and the generated fill. A fill that was conditioned on the suffix should lead naturally into it, lowering that boundary number.

| condition | context | recon ppl | boundary NLL (lower better) |
| --- | --- | --- | --- |
| AR | left-only (causal) | **4.49** | 3.742 |
| LLaDA | left-only (suffix masked) | 6.87 | 3.773 |
| LLaDA | bidirectional (suffix visible) | 6.52 | **2.589** |

![Bar chart of boundary NLL for experiment E3. AR left-only and LLaDA left-only are essentially tied around 3.74 to 3.77 nats, while the same LLaDA with the suffix visible drops to 2.589 nats, the only bar that uses right context.](/assets/images/diffusion-llm/e3_infill.png)

The result is clean. Giving the same LLaDA the suffix drops its boundary NLL from 3.77 to 2.59, a 1.18 nat improvement, purely from being allowed to see what comes next. The generated fill genuinely coheres with the continuation. And look at the AR row: it is the more fluent model in absolute terms (reconstruction perplexity 4.49, the best of the three), but its boundary NLL (3.74) is essentially tied with LLaDA's left-only condition, because AR also cannot see the suffix. The bidirectional row is a setting AR cannot reproduce at all. Qualitatively, when the suffix is about climbing an icy hill, bidirectional LLaDA steers its fill toward that ending, while AR wanders off into a locally fluent but unrelated story, because it has no way of knowing how the passage is supposed to end.

## So, should you use diffusion for language?

After building all three and running the experiments, my honest summary is that at this scale it is a tradeoff, not a verdict.

| | Autoregressive | Diffusion (LLaDA / SEDD) |
| --- | --- | --- |
| Likelihood | exact | upper bound only |
| Quality per unit compute | best at this scale | worse, slower to converge |
| Training stability | robust | finicky (the init bug, SEDD's noisy loss) |
| Sequential decode steps | `seq_len`, irreducible | a tunable constant |
| Quality vs latency knob | none | yes, the step count |
| Bidirectional context / infilling | no | yes, natively |

Autoregressive models win the things that made them dominant: exact likelihood, the best quality for a given compute budget, and dead-simple, stable training. Diffusion language models win on the axes that perplexity never captured, a tunable knob that trades inference compute for quality, parallel decoding that decouples latency from sequence length, and the ability to condition on both sides of a gap. None of those advantages shows up if you only look at a perplexity table, which is exactly why the comparison has to be designed around metrics that mean the same thing for every model.

Whether that tradeoff tips toward diffusion at the scale of something like Gemini Diffusion is a different question than the one a single-GPU TinyStories project can answer. But the mechanisms are the same ones we just walked through, and now at least we have measured, rather than asserted, where each approach actually earns its keep.

If you want to poke at the numbers, the implementation is the small project described throughout: a shared backbone, the LLaDA and SEDD objectives, an AR baseline, and three eval scripts (`compare_decoding`, `latency_eval`, `infill_eval`) that produce the tables above.

## References

1. <a id="ref-ddpm"></a>J. Ho, A. Jain, and P. Abbeel. "Denoising Diffusion Probabilistic Models." *NeurIPS*, 2020. arXiv:2006.11239. [[link]](https://arxiv.org/abs/2006.11239)
2. <a id="ref-fm"></a>Y. Lipman, R. T. Q. Chen, H. Ben-Hamu, M. Nickel, and M. Le. "Flow Matching for Generative Modeling." *ICLR*, 2023. arXiv:2210.02747. [[link]](https://arxiv.org/abs/2210.02747)
3. <a id="ref-llada"></a>S. Nie, F. Zhu, Z. You, X. Zhang, J. Ou, J. Hu, J. Zhou, Y. Lin, J.-R. Wen, and C. Li. "Large Language Diffusion Models." 2025. arXiv:2502.09992. [[link]](https://arxiv.org/abs/2502.09992). The LLaDA recipe: masked diffusion with the $1/p_{\text{mask}}$ weighted loss and confidence-based remasking sampler used here.
4. <a id="ref-sedd"></a>A. Lou, C. Meng, and S. Ermon. "Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution." *ICML*, 2024. arXiv:2310.16834. [[link]](https://arxiv.org/abs/2310.16834). Score Entropy Discrete Diffusion, the score-ratio view and the absorbing-state predictor-corrector sampler.
5. <a id="ref-d3pm"></a>J. Austin, D. Johnson, J. Ho, D. Tarlow, and R. van den Berg. "Structured Denoising Diffusion Models in Discrete State-Spaces." *NeurIPS*, 2021. arXiv:2107.03006. [[link]](https://arxiv.org/abs/2107.03006). The earlier framework that formalized discrete-state forward processes, including the absorbing (masking) kernel.
6. <a id="ref-bert"></a>J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." *NAACL*, 2019. arXiv:1810.04805. [[link]](https://arxiv.org/abs/1810.04805). Masked language modelling, which the LLaDA forward process is a noise-scheduled generalization of.
7. <a id="ref-gpt2"></a>A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, and I. Sutskever. "Language Models are Unsupervised Multitask Learners." 2019. [[link]](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf). Source of the $1/\sqrt{2N}$ residual-projection init that the debugging story turned on.
8. <a id="ref-tinystories"></a>R. Eldan and Y. Li. "TinyStories: How Small Can Language Models Be and Still Speak Coherent English?" 2023. arXiv:2305.07759. [[link]](https://arxiv.org/abs/2305.07759). The synthetic dataset that makes all of this trainable on a single GPU.
9. <a id="ref-anderson"></a>B. D. O. Anderson. "Reverse-time diffusion equation models." *Stochastic Processes and their Applications*, 12(3):313–326, 1982. [[link]](https://doi.org/10.1016/0304-4149(82)90051-5). The reverse-time result whose discrete (continuous-time Markov chain) analog underlies the SEDD reverse process.

See you on the next blog post!

