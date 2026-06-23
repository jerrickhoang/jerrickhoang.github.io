---
layout: post
title: "Estimating Exact Sample Probabilities in Diffusion Models"
date: 2025-05-03
description: "How to compute the exact log-probability of a sample under a diffusion model: the variance-exploding SDE, the probability-flow ODE, the instantaneous change-of-variables (trace) identity, and Hutchinson's estimator, with from-scratch PyTorch."
category: "Machine Learning"
tags: ["diffusion", "ddpm", "flow-matching", "edm", "generative-models", "probability"]
---

A diffusion model gives you a way to *sample*, but it doesn’t obviously give you a way to ask “how likely is this particular sample?”. It turns out you can recover the exact log-probability of any point, and it’s genuinely useful: MotionDiffuser ([Jiang et al., 2023](#ref-motiondiffuser)) uses it to rank and filter generated trajectories by likelihood, and the same machinery underlies likelihood evaluation and out-of-distribution detection. This post derives that probability from the ground up and then implements it.

DDPM ([Ho et al., 2020](#ref-ddpm)), EDM ([Karras et al., 2022](#ref-edm)), and flow matching ([Lipman et al., 2022](#ref-fm)) are all members of the same diffusion / score-based family: DDPM is the original discrete-time formulation, while EDM and flow matching are continuous-time frameworks that turn out to be more convenient for the likelihood computation we’re after. The derivations here follow the EDM paper, extended to conditional flow matching.

## Some Math

Both EDM and flow matching define a fixed schedule $(\alpha(t), \beta(t))$ such that,

$$z(t) = \alpha(t)\, z(0) + \beta(t)\, \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I),$$

with the clean data at $t = 0$ and (scaled) noise at the final time. In EDM, $\alpha(t) = 1$ and $\beta(t) = \sigma(t)$ where $\sigma(t)$ is a noise schedule. In flow matching, $\alpha(t) = 1 - t$ and $\beta(t) = t$.

Let’s write the schedule equation for EDM again,

$$z = x_0 + \sigma(t)\, \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I).$$

EDM is built on a *variance-exploding* SDE, where the variance of $z$ grows over time. Recall that $W_t$ is a Wiener process (standard Brownian motion):

- $W_0 = 0$
- independent Gaussian increments,

$$W_{t+\Delta t} - W_t \sim \mathcal{N}(0,\ \Delta t),$$

- almost surely continuous but nowhere differentiable.

The SDE we want is,

$$dz = g(t)\, dW_t,$$

where $g(t)$ is the diffusion coefficient (note there is no drift term, which is exactly what “variance exploding” buys us). Why this form? The key facts about the Wiener increment are,

$$\mathbb{E}[dW_t] = 0, \quad \mathrm{Var}(dW_t) = \Delta t,$$

so over an infinitesimally small $\Delta t$,

$$z_{t+\Delta t} - z_t = g(t)\left(W_{t+\Delta t} - W_t\right) \quad \text{with} \quad W_{t+\Delta t} - W_t \sim \mathcal{N}(0, \Delta t).$$

Intuitively, at each instant we add mean-zero Gaussian noise of variance $g^2(t)\, dt$. Integrating from $0$ to $t$ (with $z(0) = 0$) gives,

$$z(t) = \int_0^t g(s)\, dW_s.$$

Applying Itô’s isometry,

$$\mathbb{E}\!\left[\left(\int_0^t g(s)\, dW_s\right)^2\right] = \mathbb{E}\!\left[\int_0^t g(s)^2\, ds\right].$$

Since the schedule $g(s)$ is deterministic (independent of $W$), the right-hand side needs no expectation, and because the Itô integral is mean-zero we have $\mathrm{Var}[z(t)] = \mathbb{E}[z(t)^2]$, so this simplifies to,

$$\mathrm{Var}[z(t)] = \mathbb{E}[z(t)^2] = \int_0^t g(s)^2\, ds.$$

Matching this against the EDM schedule $z = x_0 + \sigma(t)\,\varepsilon$, we want $\mathrm{Var}[z(t)] = \sigma(t)^2$, which gives,

$$\int_0^t g(s)^2\, ds = \sigma(t)^2 \quad \implies \quad g(t) = \sqrt{\frac{d}{dt}\left[\sigma(t)^2\right]}.$$

In other words,

$$dz = \sqrt{\frac{d}{dt}\left[\sigma(t)^2\right]}\ dW_t.$$

Now the bridge to a deterministic ODE. The reverse-time SDE result of Anderson ([Anderson, 1982](#ref-anderson)), made into the *probability-flow ODE* by Song et al. ([Song et al., 2021](#ref-song)), says that any Itô SDE,

$$dz = \mu(z, t)\, dt + g(t)\, dW_t,$$

with drift $\mu$, admits a purely deterministic ODE that has the **same time-marginal densities**,

$$\frac{dz}{dt} = \mu(z, t) - \tfrac{1}{2} g(t)^2 \nabla_z \log p(z, t).$$

Our EDM SDE has no drift ($\mu = 0$), so plugging in $g(t)^2 = \tfrac{d}{dt}[\sigma(t)^2]$ gives,

$$\frac{dz}{dt} = -\tfrac{1}{2}\frac{d}{dt}\left[\sigma(t)^2\right] \nabla_z \log p\big(z(t); \sigma(t)\big),$$

which simplifies to,

$$\frac{dz}{dt} = -\sigma(t)\, \dot{\sigma}(t)\, \nabla_z \log p(z; \sigma).$$

From here it’s convenient to give the right-hand side of this ODE its own name, the **vector field** $f(z, t)$ (not to be confused with the SDE drift $\mu$ above, which was zero for us). Both EDM and flow matching ultimately learn this vector field,

$$\frac{dz}{dt} = f\big(z(t), t\big),$$

where $t \in [0, T]$ denotes “continuous” time, $z(0)$ is the data latent, and $z(T)$ is pure noise (for flow matching $T = 1$). The two frameworks differ only in what $f$ is:

- In EDM,

$$f(z, t) = -\dot{\sigma}(t)\, \sigma(t)\, \nabla_z \log p(z; \sigma(t)).$$

- In flow matching,

$$f(z, t) = v_\theta(z, t),$$

the velocity field learned directly.

Now recall Liouville’s theorem, relating the change in density to net inflows / outflows. For a deterministic flow,

$$\frac{dz}{dt} = f\big(z(t), t\big),$$

the density $p(z, t)$ of points transported by the flow must satisfy the continuity equation,

$$\frac{\partial}{\partial t} p(z, t) + \nabla_z \cdot \big(p(z, t)\, f(z, t)\big) = 0.$$

Intuitively this just means “mass is neither created nor destroyed”:

- $\partial_t p(z, t)$ is how fast the density at point $z$ rises or falls over time;
- $p\, f$ is the flux, how much “stuff” moves through each point per unit time;
- $\nabla \cdot (p\, f)$ is the net outflow from an infinitesimal volume: positive divergence (more flux out than in) means the density inside must decrease, negative divergence means it increases.

So Liouville’s equation is just a conservation law,

$$\underbrace{\frac{\partial p}{\partial t}}_{\text{local change}} + \underbrace{\nabla \cdot (p\, f)}_{\text{net outflow}} = 0 \quad \iff \quad \partial_t p = -\nabla \cdot (p\, f).$$

Now follow a single particle $z(t)$ along the flow and ask how the density *it* experiences changes as it moves. By the chain rule (the material derivative),

$$\frac{d}{dt} p\big(z(t), t\big) = \underbrace{\frac{\partial p}{\partial t}}_{\text{local time-change}} + \underbrace{\frac{dz}{dt} \cdot \nabla_z p}_{\text{advection by the flow}}.$$

Using $\frac{dz}{dt} = f(z, t)$ we have,

$$\frac{d}{dt} p\big(z(t), t\big) = \partial_t p + f(z, t) \cdot \nabla_z p.$$

Substituting the continuity law $\partial_t p = -\nabla \cdot (p\, f) = -\left[p\, (\nabla \cdot f) + f \cdot \nabla p\right]$, the two advection terms cancel,

$$\frac{d}{dt} p = -p\, (\nabla \cdot f) \quad \implies \quad \frac{d}{dt} \log p = \frac{1}{p}\frac{dp}{dt} = -\nabla \cdot f.$$

By the definition of divergence, $\nabla \cdot f$ is the sum of the diagonal Jacobian entries, i.e. the trace of the Jacobian, so

$$\frac{d}{dt} \log p\big(z(t)\big) = -\mathrm{Tr}\big(\partial_z f(z(t), t)\big).$$

This is the *instantaneous change-of-variables* formula, equation (11) in the MotionDiffuser paper ([Jiang et al., 2023](#ref-motiondiffuser)), and the same equation applies to both the flow-matching and EDM formulations.

## Point Probability Estimation

For a particle following the path $z(t)$, with $z(0)$ the real data and $z(T)$ pure noise, we integrate the relation above from $0$ to $T$ to get,

$$\log p\big(z(0)\big) = \log p\big(z(T)\big) - \int_0^T \mathrm{Tr}\big(\partial_z f(z(t), t)\big)\, dt.$$

Where,

- $\log p(z(T))$ is known in closed form: for EDM it is $\mathcal{N}(0, \sigma_{\max}^2 I)$ (the terminal, maximum noise level), and for flow matching it is $\mathcal{N}(0, I)$;
- the integral of traces captures how the model’s vector field reshapes probability mass as the particle moves back toward the data at $t = 0$.

So to score a real sample $z(0)$: run the ODE out to $z(T)$, evaluate the (Gaussian) prior log-density there, and subtract the accumulated trace of the Jacobian along the path.

## Some Code

The one expensive piece above is $\mathrm{Tr}(\partial_z f)$. Computed exactly it costs one Jacobian-vector product per input dimension, i.e. $O(n)$ backward passes for an $n$-dimensional $z$, which is hopeless for high-dimensional latents. Hutchinson’s trace estimator ([Hutchinson, 1990](#ref-hutchinson); applied to continuous-time log-likelihoods in FFJORD, [Grathwohl et al., 2019](#ref-ffjord)) gives an unbiased estimate at the cost of just a few Jacobian-vector products.

The estimator rests on one fact: for any random vector $v$ with $\mathbb{E}[v v^T] = I$,

$$\mathbb{E}\!\left[v^T A\, v\right] = \mathrm{Tr}(A).$$

So we sample a batch of random vectors $v$, compute $v^T (A v)$, and average to get an unbiased estimate of $\mathrm{Tr}(A)$. The easiest way to get vectors satisfying $\mathbb{E}[v v^T] = I$ is to draw from a Rademacher distribution,

- $v_i = +1$ with probability $0.5$ and $v_i = -1$ with probability $0.5$.

In our case $A = \partial_z f$ is the Jacobian of the vector field, and $A v$ is exactly a vector-Jacobian product, which autograd gives us for free. We never have to materialize the full Jacobian,

```python
import math
import torch

def hutchinson_trace(f, z, t, n_samples=1):
    """
    Unbiased estimate of Tr(∂f/∂z) at (z, t) via Hutchinson's estimator.
    `f(z, t)` returns the ODE vector field, same shape as `z`; the leading
    dimension of `z` is the batch, so we return one trace per batch item.
    """
    z = z.detach().requires_grad_(True)
    fz = f(z, t)
    trace = torch.zeros(z.shape[0], device=z.device)
    for _ in range(n_samples):
        # Rademacher probe vector v (E[v vᵀ] = I)
        v = torch.randint(0, 2, z.shape, device=z.device, dtype=z.dtype) * 2 - 1
        # vector-Jacobian product: vᵀ (∂f/∂z)
        vjp = torch.autograd.grad(fz, z, grad_outputs=v, retain_graph=True)[0]
        # vᵀ (∂f/∂z) v, summed over all non-batch dims
        trace += (vjp * v).flatten(1).sum(dim=1)
    return trace / n_samples
```

With the trace in hand, point-probability estimation is just integrating the ODE from the data end to the noise end while accumulating the trace, then adding the prior log-density,

```python
def gaussian_logp(z, sigma=1.0):
    """log N(z; 0, sigma^2 I), one value per batch item."""
    flat = z.flatten(1)
    n    = flat.shape[1]
    sq   = (flat ** 2).sum(dim=1)
    return -0.5 * (sq / sigma**2 + n * math.log(2 * math.pi * sigma**2))

def log_prob(f, z0, T=1.0, n_steps=100, prior_logp=gaussian_logp, n_hutch=1):
    """
    Estimate log p(z0) by integrating  d/dt log p = -Tr(∂f/∂z)  along the ODE
    from t = 0 (data) to t = T (noise), then adding the prior log-density at z(T):

        log p(z0) = log p(z(T)) - ∫₀ᵀ Tr(∂f/∂z) dt
    """
    dt          = T / n_steps
    z           = z0.clone()
    trace_integ = torch.zeros(z.shape[0], device=z.device)

    t = 0.0
    for _ in range(n_steps):
        trace_integ += hutchinson_trace(f, z, t, n_samples=n_hutch) * dt
        with torch.no_grad():
            z = z + f(z, t) * dt          # forward Euler step along the ODE
        t += dt

    return prior_logp(z) - trace_integ
```

It’s worth sanity-checking the trace estimator on a case where we know the answer. For a *linear* field $f(z) = z A^T$ the Jacobian is exactly $A$ (independent of $z$), so the estimate should converge to $\mathrm{Tr}(A)$,

```python
A     = torch.randn(8, 8)
f_lin = lambda z, t: z @ A.T
z     = torch.randn(4096, 8)

est = hutchinson_trace(f_lin, z, 0.0, n_samples=4).mean()
print(f"Hutchinson estimate: {est:.3f}   exact Tr(A): {torch.trace(A):.3f}")
# Hutchinson estimate: -1.812   exact Tr(A): -1.794
```

The two agree up to the estimator’s variance, which shrinks as we average more probe vectors (and over the batch). Swapping `f_lin` for a trained EDM or flow-matching vector field is then all it takes to score real samples.

## Wrapping up

We started from the variance-exploding SDE behind EDM, converted it into the deterministic probability-flow ODE that shares the same marginals, and used the continuity equation to turn “how does the density change along the flow” into the clean trace identity $\tfrac{d}{dt}\log p = -\mathrm{Tr}(\partial_z f)$. Integrating that along the ODE, with a Gaussian prior at the noise end and Hutchinson’s estimator for the trace, gives the exact log-probability of any sample, for both EDM and flow matching. The payoff is concrete: a single scalar per sample that you can use to rank generations, evaluate likelihoods, or flag out-of-distribution inputs.

Two practical caveats. The log-probability is only as exact as your ODE solver, the forward-Euler loop above is written for clarity, so in practice use a higher-order solver (e.g. Heun, as EDM and MotionDiffuser do) and enough steps. And Hutchinson trades exactness for speed: it’s unbiased but noisy, and averaging more probe vectors reduces the variance at linear cost.

## References

1. <a id="ref-ddpm"></a>J. Ho, A. Jain, and P. Abbeel. "Denoising Diffusion Probabilistic Models." *NeurIPS*, 2020. arXiv:2006.11239. [[link]](https://arxiv.org/abs/2006.11239)
2. <a id="ref-edm"></a>T. Karras, M. Aittala, T. Aila, and S. Laine. "Elucidating the Design Space of Diffusion-Based Generative Models." *NeurIPS*, 2022. arXiv:2206.00364. [[link]](https://arxiv.org/abs/2206.00364) — The EDM formulation and noise-schedule conventions these derivations extend.
3. <a id="ref-fm"></a>Y. Lipman, R. T. Q. Chen, H. Ben-Hamu, M. Nickel, and M. Le. "Flow Matching for Generative Modeling." *ICLR*, 2023. arXiv:2210.02747. [[link]](https://arxiv.org/abs/2210.02747)
4. <a id="ref-song"></a>Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, and B. Poole. "Score-Based Generative Modeling through Stochastic Differential Equations." *ICLR*, 2021. arXiv:2011.13456. [[link]](https://arxiv.org/abs/2011.13456) — Introduces the probability-flow ODE whose deterministic dynamics are used throughout.
5. <a id="ref-anderson"></a>B. D. O. Anderson. "Reverse-time diffusion equation models." *Stochastic Processes and their Applications*, 12(3):313–326, 1982. [[link]](https://doi.org/10.1016/0304-4149(82)90051-5) — The reverse-time SDE result underlying the deterministic ODE.
6. <a id="ref-motiondiffuser"></a>C. Jiang, A. Cornman, C. Park, B. Sapp, Y. Zhou, and D. Anguelov. "MotionDiffuser: Controllable Multi-Agent Motion Prediction using Diffusion." *CVPR*, 2023. arXiv:2306.03083. [[link]](https://arxiv.org/abs/2306.03083) — Equation (11) (the exact log-probability integral) and the use of Hutchinson's estimator referenced here.
7. <a id="ref-hutchinson"></a>M. F. Hutchinson. "A stochastic estimator of the trace of the influence matrix for Laplacian smoothing splines." *Communications in Statistics — Simulation and Computation*, 19(2):433–450, 1990. [[link]](https://doi.org/10.1080/03610919008812866)
8. <a id="ref-ffjord"></a>W. Grathwohl, R. T. Q. Chen, J. Bettencourt, I. Sutskever, and D. Duvenaud. "FFJORD: Free-Form Continuous Dynamics for Scalable Reversible Generative Models." *ICLR*, 2019. arXiv:1810.01367. [[link]](https://arxiv.org/abs/1810.01367) — Where the Hutchinson trace estimator is applied to continuous-time log-likelihoods.
