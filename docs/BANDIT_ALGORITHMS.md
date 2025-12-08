# Bandit Algorithm Formulations

**Last Updated**: 2025-12-07

Mathematical formulations and implementation details for all multi-armed bandit algorithms used in conduit-benchmark.

---

## Table of Contents

1. [Overview](#overview)
2. [Multi-Objective Reward Function](#multi-objective-reward-function)
3. [Non-Contextual Algorithms](#non-contextual-algorithms)
   - [Thompson Sampling](#1-thompson-sampling)
   - [UCB1](#2-ucb1-upper-confidence-bound)
   - [Epsilon-Greedy](#3-epsilon-greedy)
4. [Contextual Algorithms](#contextual-algorithms)
   - [LinUCB](#4-linucb-linear-upper-confidence-bound)
   - [Contextual Thompson Sampling](#5-contextual-thompson-sampling)
   - [Dueling Bandit](#6-dueling-bandit)
5. [Hybrid Algorithms](#hybrid-algorithms)
   - [Hybrid Thompson-LinUCB](#7-hybrid-thompson-linucb)
   - [Hybrid UCB1-LinUCB](#8-hybrid-ucb1-linucb)
6. [Baseline Algorithms](#baseline-algorithms)
7. [Algorithm Comparison](#algorithm-comparison)
8. [References](#references)

---

## Overview

We evaluate 11 multi-armed bandit algorithms for LLM routing, spanning three categories:

| Category | Algorithms | Key Property |
|----------|------------|--------------|
| **Non-Contextual** | Thompson, UCB1, Epsilon-Greedy | Ignore query features |
| **Contextual** | LinUCB, Contextual Thompson, Dueling | Use query embeddings |
| **Hybrid** | Thompson+LinUCB, UCB1+LinUCB | Warm-start then contextual |
| **Baseline** | Random, Always-Best, Always-Cheapest | Reference bounds |

### Notation

| Symbol | Meaning |
|--------|---------|
| $K$ | Number of arms (models) |
| $t$ | Time step (query index) |
| $a_t$ | Arm selected at time $t$ |
| $r_t$ | Reward received at time $t$ |
| $x_t \in \mathbb{R}^d$ | Context feature vector at time $t$ |
| $n_a$ | Number of times arm $a$ has been pulled |
| $\bar{r}_a$ | Empirical mean reward for arm $a$ |

---

## Multi-Objective Reward Function

All learning algorithms optimize a **composite reward** balancing quality, cost, and latency:

$$r = w_q \cdot q - w_c \cdot c - w_l \cdot l$$

Where:
- $q \in [0, 1]$: Quality score (higher is better)
- $c \in [0, 1]$: Normalized cost (0 = cheapest, 1 = most expensive)
- $l \in [0, 1]$: Normalized latency (0 = fastest, 1 = slowest)

**Default weights** (from `conduit.yaml`):
- $w_q = 0.70$ (quality)
- $w_c = 0.20$ (cost)
- $w_l = 0.10$ (latency)

**Normalization**:
- Cost: $c = \frac{\text{cost} - \text{min\_cost}}{\text{max\_cost} - \text{min\_cost}}$
- Latency: $l = \frac{\text{latency} - \text{min\_latency}}{\text{max\_latency} - \text{min\_latency}}$

---

## Non-Contextual Algorithms

These algorithms learn model quality without considering query features.

### 1. Thompson Sampling

**Model**: Beta-Bernoulli conjugate prior

**Prior**: Each arm $a$ has Beta distribution parameters $(\alpha_a, \beta_a)$, initialized to $(1, 1)$.

**Selection** (at time $t$):
1. For each arm $a$, sample $\theta_a \sim \text{Beta}(\alpha_a, \beta_a)$
2. Select arm $a_t = \arg\max_a \theta_a$

**Update** (after observing reward $r_t$ for arm $a_t$):
$$\alpha_{a_t} \leftarrow \alpha_{a_t} + r_t$$
$$\beta_{a_t} \leftarrow \beta_{a_t} + (1 - r_t)$$

**Expected Value**:
$$\mathbb{E}[\theta_a] = \frac{\alpha_a}{\alpha_a + \beta_a}$$

**Variance**:
$$\text{Var}[\theta_a] = \frac{\alpha_a \beta_a}{(\alpha_a + \beta_a)^2 (\alpha_a + \beta_a + 1)}$$

**Hyperparameters**:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `prior_alpha` | 1.0 | Prior successes |
| `prior_beta` | 1.0 | Prior failures |
| `window_size` | 0 | Sliding window (0 = unlimited) |

**Regret Bound**: $O(\sqrt{KT \log T})$ (optimal for stochastic bandits)

**Implementation**: `conduit/engines/bandits/thompson_sampling.py`

---

### 2. UCB1 (Upper Confidence Bound)

**Model**: Optimistic upper confidence bound on mean reward

**Selection** (at time $t$):
$$a_t = \arg\max_a \left[ \bar{r}_a + c \sqrt{\frac{\ln t}{n_a}} \right]$$

Where:
- $\bar{r}_a$: Empirical mean reward for arm $a$
- $n_a$: Number of pulls for arm $a$
- $c$: Exploration parameter (default: $\sqrt{2}$)
- $t$: Total pulls across all arms

**Components**:
- **Exploitation**: $\bar{r}_a$ (select high-reward arms)
- **Exploration**: $c \sqrt{\frac{\ln t}{n_a}}$ (bonus for under-explored arms)

**Update** (after observing reward $r_t$ for arm $a_t$):
$$\bar{r}_{a_t} \leftarrow \bar{r}_{a_t} + \frac{1}{n_{a_t}}(r_t - \bar{r}_{a_t})$$
$$n_{a_t} \leftarrow n_{a_t} + 1$$

**Initialization**: Pull each arm once before UCB selection.

**Hyperparameters**:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `c` | $\sqrt{2} \approx 1.414$ | Exploration parameter |
| `window_size` | 0 | Sliding window (0 = unlimited) |

**Regret Bound**: $O(\sqrt{KT \log T})$ (optimal)

**Implementation**: `conduit/engines/bandits/ucb.py`

---

### 3. Epsilon-Greedy

**Model**: Simple exploration-exploitation with decaying exploration rate

**Selection** (at time $t$):
$$a_t = \begin{cases}
\text{uniform random arm} & \text{with probability } \varepsilon_t \\
\arg\max_a \bar{r}_a & \text{with probability } 1 - \varepsilon_t
\end{cases}$$

**Epsilon Decay**:
$$\varepsilon_t = \max(\varepsilon_{\min}, \varepsilon_{t-1} \cdot \gamma)$$

Where $\gamma$ is the decay rate.

**Hyperparameters**:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `epsilon` | 0.1 | Initial exploration rate |
| `decay` | 1.0 | Multiplicative decay per step |
| `min_epsilon` | 0.01 | Minimum exploration floor |
| `window_size` | 0 | Sliding window (0 = unlimited) |

**Regret Bound**: $O(K \log T / \Delta)$ where $\Delta$ is the suboptimality gap (suboptimal)

**Implementation**: `conduit/engines/bandits/epsilon_greedy.py`

---

## Contextual Algorithms

These algorithms use query features (embeddings) to make context-dependent decisions.

### 4. LinUCB (Linear Upper Confidence Bound)

**Model**: Ridge regression with optimistic exploration

**Assumption**: Expected reward is linear in context features:
$$\mathbb{E}[r | x, a] = \theta_a^\top x$$

**Per-Arm State**:
- $A_a \in \mathbb{R}^{d \times d}$: Design matrix (initialized to $I_d$)
- $b_a \in \mathbb{R}^d$: Response vector (initialized to $\mathbf{0}$)
- $A_a^{-1}$: Cached inverse (updated incrementally)

**Ridge Regression Estimate**:
$$\hat{\theta}_a = A_a^{-1} b_a$$

**Selection** (at time $t$ with context $x_t$):
$$a_t = \arg\max_a \left[ \hat{\theta}_a^\top x_t + \alpha \sqrt{x_t^\top A_a^{-1} x_t} \right]$$

Where:
- $\hat{\theta}_a^\top x_t$: Predicted reward (exploitation)
- $\alpha \sqrt{x_t^\top A_a^{-1} x_t}$: Confidence width (exploration)

**Update** (after observing reward $r_t$ for arm $a_t$):
$$A_{a_t} \leftarrow A_{a_t} + x_t x_t^\top$$
$$b_{a_t} \leftarrow b_{a_t} + r_t x_t$$

**Incremental Inverse Update** (Sherman-Morrison formula):
$$A^{-1}_{\text{new}} = A^{-1} - \frac{A^{-1} x x^\top A^{-1}}{1 + x^\top A^{-1} x}$$

**Hyperparameters**:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha` | 1.0 | Exploration parameter |
| `feature_dim` | 1538 | Context dimension (embedding + metadata) |
| `window_size` | 0 | Sliding window (0 = unlimited) |

**Regret Bound**: $O(d\sqrt{T \log T})$ (optimal for linear bandits)

**Convergence**: Requires $O(d)$ to $O(10d)$ samples for stable estimates

**Implementation**: `conduit/engines/bandits/linucb.py`

---

### 5. Contextual Thompson Sampling

**Model**: Bayesian linear regression with posterior sampling

**Prior**: $\theta_a \sim \mathcal{N}(\mathbf{0}, \lambda^{-1} I_d)$

**Posterior** (after $n$ observations):
$$\theta_a | \mathcal{D} \sim \mathcal{N}(\mu_a, \Sigma_a)$$

Where:
$$\Sigma_a = \left( \lambda I_d + \sum_{i: a_i = a} x_i x_i^\top \right)^{-1}$$
$$\mu_a = \Sigma_a \left( \sum_{i: a_i = a} r_i x_i \right)$$

**Selection** (at time $t$ with context $x_t$):
1. For each arm $a$, sample $\tilde{\theta}_a \sim \mathcal{N}(\mu_a, \Sigma_a)$
2. Select $a_t = \arg\max_a \tilde{\theta}_a^\top x_t$

**Sampling** (via Cholesky decomposition):
$$\tilde{\theta}_a = \mu_a + L_a z, \quad z \sim \mathcal{N}(\mathbf{0}, I_d)$$

Where $L_a L_a^\top = \Sigma_a$ (Cholesky factor).

**Hyperparameters**:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `lambda_reg` | 1.0 | Prior precision / regularization |
| `feature_dim` | 1538 | Context dimension |
| `window_size` | 0 | Sliding window (0 = unlimited) |

**Regret Bound**: $O(d\sqrt{T} \log T)$ (optimal)

**Implementation**: `conduit/engines/bandits/contextual_thompson_sampling.py`

---

### 6. Dueling Bandit

**Model**: Pairwise preference learning (FGTS.CDB - Fast Gradient Thompson Sampling)

**Preference Model**: Learn preference weights $w_a \in \mathbb{R}^d$ for each arm.

**Preference Score**:
$$s_a(x) = w_a^\top x$$

**Selection** (at time $t$ with context $x_t$):
1. Compute noisy scores: $\tilde{s}_a = w_a^\top x_t + \sigma \cdot z_a$, where $z_a \sim \mathcal{N}(0, 1)$
2. Select top 2 arms: $(a_1, a_2) = \text{argtop2}_a \tilde{s}_a$
3. Execute both and observe preference $p \in [-1, 1]$

**Update** (gradient descent on preference):
$$w_{a_1} \leftarrow w_{a_1} + \eta \cdot p \cdot c \cdot x_t$$
$$w_{a_2} \leftarrow w_{a_2} - \eta \cdot p \cdot c \cdot x_t$$

Where:
- $p$: Preference score ($+1$ = $a_1$ better, $-1$ = $a_2$ better)
- $c$: Confidence in preference judgment
- $\eta$: Learning rate

**Hyperparameters**:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `exploration_weight` | 0.1 | Thompson noise std dev ($\sigma$) |
| `learning_rate` | 0.01 | Gradient step size ($\eta$) |
| `feature_dim` | 1538 | Context dimension |

**Use Case**: When explicit quality scores are unavailable but pairwise comparisons are possible.

**Implementation**: `conduit/engines/bandits/dueling.py`

---

## Hybrid Algorithms

Hybrid algorithms combine fast non-contextual warm-start with contextual learning.

### 7. Hybrid Thompson-LinUCB

**Phase 1** (Warm-start, $t < T_{\text{switch}}$):
- Use Thompson Sampling (non-contextual)
- Fast initial exploration without feature computation

**Phase 2** (Contextual, $t \geq T_{\text{switch}}$):
- Convert Thompson state to LinUCB
- Use LinUCB with contextual features

**State Conversion** (Thompson → LinUCB):
$$\bar{r}_a = \frac{\alpha_a}{\alpha_a + \beta_a}$$
$$n_a = \alpha_a + \beta_a - 2$$

LinUCB initialized with empirical means from Thompson phase.

**Hyperparameters**:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `switch_threshold` | 2000 | Queries before switching |
| `ucb1_c` | 1.5 | Phase 1 exploration |
| `linucb_alpha` | 1.0 | Phase 2 exploration |

**Implementation**: `conduit/engines/bandits/hybrid_bandits.py`

---

### 8. Hybrid UCB1-LinUCB

**Phase 1** (Warm-start, $t < T_{\text{switch}}$):
- Use UCB1 (non-contextual)
- Deterministic exploration based on confidence bounds

**Phase 2** (Contextual, $t \geq T_{\text{switch}}$):
- Convert UCB1 state to LinUCB
- Use LinUCB with contextual features

**State Conversion** (UCB1 → LinUCB):
- Transfer $\bar{r}_a$ and $n_a$ directly
- Initialize LinUCB design matrices with identity

**Hyperparameters**: Same as Hybrid Thompson-LinUCB

**Implementation**: `conduit/engines/bandits/hybrid_bandits.py`

---

## Baseline Algorithms

### Random
- **Selection**: $a_t \sim \text{Uniform}(\{1, \ldots, K\})$
- **Use**: Lower bound on performance

### Always-Best
- **Selection**: $a_t = \arg\max_a \mathbb{E}[q_a]$ (highest expected quality)
- **Use**: Upper bound on quality, ignores cost

### Always-Cheapest
- **Selection**: $a_t = \arg\min_a \text{cost}_a$ (lowest cost model)
- **Use**: Upper bound on cost efficiency, ignores quality

### Oracle (excluded from presets)
- **Selection**: Execute all $K$ arms, select best for this query
- **Use**: Theoretical upper bound (perfect hindsight)
- **Cost**: $K \times$ standard cost (runs all models)

**Implementation**: `conduit/engines/bandits/baselines.py`

---

## Algorithm Comparison

| Algorithm | Contextual | Regret Bound | Convergence | Best For |
|-----------|------------|--------------|-------------|----------|
| Thompson | No | $O(\sqrt{KT \log T})$ | Fast | General use |
| UCB1 | No | $O(\sqrt{KT \log T})$ | Fast | Deterministic exploration |
| Epsilon-Greedy | No | $O(K \log T / \Delta)$ | Medium | Simplicity |
| LinUCB | Yes | $O(d\sqrt{T \log T})$ | $O(d)$ samples | Query-dependent routing |
| Contextual TS | Yes | $O(d\sqrt{T} \log T)$ | $O(d)$ samples | Uncertainty quantification |
| Dueling | Yes | - | Medium | Pairwise feedback |
| Hybrid | Yes | Combined | Fast then $O(d)$ | Production systems |

### Selection Guidance

- **Few queries (<500)**: Thompson Sampling or UCB1
- **Many queries (>2000)**: LinUCB or Contextual Thompson
- **Query diversity matters**: Contextual algorithms
- **Production deployment**: Hybrid Thompson-LinUCB
- **Pairwise feedback only**: Dueling Bandit

---

## References

1. **Thompson Sampling**: Thompson, W.R. (1933). "On the Likelihood that One Unknown Probability Exceeds Another in View of the Evidence of Two Samples." *Biometrika*.

2. **UCB1**: Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002). "Finite-time Analysis of the Multiarmed Bandit Problem." *Machine Learning*.

3. **LinUCB**: Li, L., Chu, W., Langford, J., & Schapire, R.E. (2010). "A Contextual-Bandit Approach to Personalized News Article Recommendation." *WWW*.

4. **Contextual Thompson Sampling**: Agrawal, S. & Goyal, N. (2013). "Thompson Sampling for Contextual Bandits with Linear Payoffs." *ICML*.

5. **Dueling Bandits**: Yue, Y. & Joachims, T. (2009). "Interactively Optimizing Information Retrieval Systems as a Dueling Bandits Problem." *ICML*.

6. **Hybrid Bandits**: Bouneffouf, D. et al. (2019). "A Survey on Practical Applications of Multi-Armed and Contextual Bandits." *arXiv*.

---

## Appendix: Code References

| Algorithm | Source File |
|-----------|-------------|
| Thompson Sampling | `conduit/engines/bandits/thompson_sampling.py` |
| UCB1 | `conduit/engines/bandits/ucb.py` |
| Epsilon-Greedy | `conduit/engines/bandits/epsilon_greedy.py` |
| LinUCB | `conduit/engines/bandits/linucb.py` |
| Contextual Thompson | `conduit/engines/bandits/contextual_thompson_sampling.py` |
| Dueling Bandit | `conduit/engines/bandits/dueling.py` |
| Hybrid Algorithms | `conduit/engines/bandits/hybrid_bandits.py` |
| Baselines | `conduit/engines/bandits/baselines.py` |
| State Conversion | `conduit/engines/bandits/state_conversion.py` |
