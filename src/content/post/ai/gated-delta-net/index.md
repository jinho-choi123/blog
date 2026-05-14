---
title: "GatedDeltaNet: Improving Mamba2 with Delta Rule"
description: "This post introduces GatedDeltaNet, an improved Mamba2 with the delta rule."
publishDate: "03 May 2026"
updatedDate: "03 May 2026"
tags: ["AI"]
---


## Prerequisites

Please read the following posts first:
- [Mathematical Formulation of Linear Attention](../linear-attention/)
- [DeltaNet: Improved Linear Attention with Delta Rule](../delta-net/)

## Summary

GatedDeltaNet unifies two complementary linear attention mechanisms: DeltaNet's delta rule for precise key-value association updates and Mamba2's scalar gating for bulk memory decay. The gated delta rule can be interpreted as online SGD with weight decay, where the gate parameter $\alpha_t$ controls the spectrum between global memory reset ($\alpha_t \to 0$) and pure delta rule ($\alpha_t \to 1$). Through a modified WY representation, GatedDeltaNet retains the same efficient chunked parallel form as DeltaNet with $O(LCd + Ld^2)$ complexity.

## Terminology

- $q_t \in \mathbb{R}^{1 \times d}$: the query vector at time $t$
- $k_t \in \mathbb{R}^{1 \times d}$: the key vector at time $t$
- $v_t \in \mathbb{R}^{1 \times d}$: the value vector at time $t$
- $o_t \in \mathbb{R}^{1 \times d}$: the output vector at time $t$
- $L$: the sequence length
- $d$: the dimension of the query, key, and value vectors
- $C$: the chunk size
- $S_{t} \in \mathbb{R}^{d \times d}$: the state matrix at time $t$
- $\beta_t \in \mathbb{R}$: the learning rate for delta rule at time $t$
- $\alpha_t \in \mathbb{R}$: the forgetting gate at time $t$, controlling memory decay
- $I$: the identity matrix
- $\odot$: element-wise multiplication operator

## Evolution of Linear Attention

Before diving into GatedDeltaNet, let's understand where it sits in the evolution of linear attention mechanisms. Each generation adds a new capability to the state update rule.

### 1st Generation: Additive (Linear Attention)

$$
S_t = S_{t-1} + k_t^T v_t
$$

The state matrix accumulates all key-value associations via simple addition. There is no mechanism to forget or modify past associations. As the sequence grows, the state accumulates noise from all previous tokens equally.

> **Limitation:** No selective forgetting. Old, irrelevant associations persist forever, causing interference in long contexts.

### 2nd Generation: Gated Additive (RetNet, Mamba2, GLA)

$$
S_t = G_t \odot S_{t-1} + k_t^T v_t
$$

A forgetting gate $G_t$ is introduced to decay the entire state before adding new associations. This enables bulk forgetting — the model can uniformly reduce the influence of all past information.

> **Limitation:** The gate applies uniformly to all stored associations. The model cannot selectively modify a specific key-value pair while preserving others.

### 3rd Generation: Delta Rule (DeltaNet)

$$
S_t = (I - \beta_t k_t^T k_t) S_{t-1} + \beta_t k_t^T v_t
$$

The delta rule treats the state as a regression model and performs per-step gradient descent to correct specific associations. This enables precise, targeted updates to individual key-value pairs.

> **Limitation:** No bulk forgetting. While individual associations can be corrected, the model cannot globally decay its memory. Long contexts still cause interference from accumulated associations.

### Bridging the Gap: GatedDeltaNet

GatedDeltaNet combines the strengths of both the 2nd and 3rd generations:

$$
S_t = \alpha_t (I - \beta_t k_t^T k_t) S_{t-1} + \beta_t k_t^T v_t
$$

The gate $\alpha_t$ provides bulk forgetting (2nd gen capability), while the delta rule provides precise association updates (3rd gen capability). The model can now do both simultaneously.

## Deriving the Gated Delta Rule

### Recap: DeltaNet's Delta Rule

Recall from the [DeltaNet post](../delta-net/) that the delta rule updates the state matrix by minimizing the MSE loss for the current key-value pair:

$$
\begin{aligned}
L_t(S) &= \frac{1}{2} \| v_t - k_t S_{t-1} \|^2 \\
\frac{\partial L_t}{\partial S} &= -k_t^T(v_t - k_t S_{t-1}) \\
S_t &= S_{t-1} - \beta_t \frac{\partial L_t}{\partial S} \\
&= S_{t-1} + \beta_t k_t^T(v_t - k_t S_{t-1}) \\
&= (I - \beta_t k_t^T k_t) S_{t-1} + \beta_t k_t^T v_t
\end{aligned}
$$

This is online SGD on the state matrix — each token performs one gradient step to bring $k_t S$ closer to $v_t$.

### Adding Weight Decay: The Gated Delta Rule

In standard optimization, there are two ways to add weight decay:

**L2 Regularization** — add $\frac{\lambda}{2}\|W\|^2$ to the loss:
$$
W_{\text{new}} = W_{\text{old}} - \eta \nabla (L + \frac{\lambda}{2}\|W\|^2)
$$

**Decoupled Weight Decay** — decay first, then take gradient step on the decayed weights:
$$
\begin{aligned}
\tilde{W} &= \alpha W_{\text{old}} \quad \text{(decay first)} \\
W_{\text{new}} &= \tilde{W} - \eta \nabla L(\tilde{W}) \quad \text{(gradient on decayed weights)}
\end{aligned}
$$

The key difference: in decoupled weight decay, the gradient is computed at the **decayed** point $\tilde{W}$, not the original $W_{\text{old}}$. GatedDeltaNet uses the decoupled form.

> **Note:** Standard AdamW evaluates the gradient at the original $W$ ($W_{\text{new}} = W_{\text{old}} - \eta(\nabla L + \lambda W_{\text{old}})$). GatedDeltaNet's "decay-then-gradient-on-decayed" is a stricter form of decoupling: applying standard AdamW directly would yield the L2-form transition $(\alpha_t I - \beta_t k_t^T k_t)$ instead.

#### Derivation: Two-Step Process

**Step A — Weight decay (bulk forgetting):**

Decay the entire state uniformly before the gradient step:
$$
\tilde{S}_{t-1} = \alpha_t S_{t-1}
$$

**Step B — Delta rule on the decayed state:**

The loss is now evaluated at the decayed state $\tilde{S}_{t-1}$, not the original $S_{t-1}$:
$$
\begin{aligned}
L_t(\tilde{S}) &= \frac{1}{2} \| v_t - k_t \tilde{S}_{t-1} \|^2 = \frac{1}{2} \| v_t - k_t \alpha_t S_{t-1} \|^2 \\
\nabla_{\tilde{S}} L_t &= -k_t^T(v_t - k_t \tilde{S}_{t-1})
\end{aligned}
$$

Apply the gradient step:
$$
\begin{aligned}
S_t &= \tilde{S}_{t-1} + \beta_t k_t^T(v_t - k_t \tilde{S}_{t-1}) \\
&= \alpha_t S_{t-1} + \beta_t k_t^T v_t - \beta_t k_t^T k_t \cdot \alpha_t S_{t-1} \\
&= \alpha_t S_{t-1} - \alpha_t \beta_t k_t^T k_t S_{t-1} + \beta_t k_t^T v_t \\
&= \alpha_t(I - \beta_t k_t^T k_t) S_{t-1} + \beta_t k_t^T v_t
\end{aligned}
$$

$$
S_t = \alpha_t (I - \beta_t k_t^T k_t) S_{t-1} + \beta_t k_t^T v_t
\tag{Eq. 1}
$$

#### Why Decoupled Weight Decay, Not L2 Regularization?

If we had used L2 regularization instead (adding $\frac{\lambda}{2}\|S\|^2$ to the loss), the update would be:

$$
S_t = (\alpha_t I - \beta_t k_t^T k_t) S_{t-1} + \beta_t k_t^T v_t
$$

The transition matrix here is $(\alpha_t I - \beta_t k_t^T k_t)$ — the scalar $\alpha_t$ does **not** factor out cleanly. This would break the WY representation trick that enables efficient chunked computation (discussed in [Chunked Parallel Form](#chunked-parallel-form-of-gateddeltanet)).

With decoupled weight decay, the transition matrix is $\alpha_t(I - \beta_t k_t^T k_t)$ — the scalar $\alpha_t$ factors out as a clean multiplier, preserving the WY representation:

| Approach | Transition Matrix | $\alpha_t$ Factors Out? | WY Representation |
|----------|-------------------|------------------------|-------------------|
| L2 Regularization | $(\alpha_t I - \beta_t k_t^T k_t)$ | No | Breaks |
| Decoupled Weight Decay | $\alpha_t(I - \beta_t k_t^T k_t)$ | Yes | Works |

> **Interpretation as SGD + Decoupled Weight Decay:**
> - **Weight decay step** $\tilde{S}_{t-1} = \alpha_t S_{t-1}$: uniformly shrinks all stored associations (bulk forgetting)
> - **Gradient step** $\beta_t k_t^T(v_t - k_t \tilde{S}_{t-1})$: corrects the specific association for $k_t$ at the decayed state

### Limiting Behaviors

The gate $\alpha_t$ creates a spectrum between known architectures:

| Condition | Behavior | Equivalent Model |
|-----------|----------|-----------------|
| $\alpha_t = 1$ | No weight decay | DeltaNet |
| $\beta_t = 0$ | No gradient step, pure decay | Mamba2-style gating |
| $\alpha_t = 0$ | Complete memory reset | Hard reset (only current token matters) |

The output is computed as before:
$$
o_t = q_t S_t
$$

## Chunked Parallel Form of GatedDeltaNet

### Recap: DeltaNet's Chunked Form

Recall from the [DeltaNet post](../delta-net/#final-form-of-deltanet) that DeltaNet uses the WY representation and UT transform to derive an efficient chunked parallel form:

$$
\begin{aligned}
&\text{Auxiliary matrices (UT transform)} \\
A_{[t]}^{0:C} &= \text{tril}(-\text{diag}(\beta_{[t]}^{0:C})K_{[t]}^{0:C} {K_{[t]}^{0:C}}^T, -1)\\
W_{[t]}^{0:C} &= (I - A_{[t]}^{0:C})^{-1} \text{diag}(\beta_{[t]}^{0:C})K_{[t]}^{0:C} \\
U_{[t]}^{0:C} &= (I - A_{[t]}^{0:C})^{-1} \text{diag}(\beta_{[t]}^{0:C})V_{[t]}^{0:C} \\ \\

&\text{State update and output} \\
S_{[t]}^{0:C} &= S_{[t-1]}^C + {K_{[t]}^{0:C}}^T \underset{\text{Delta corrected Value}}{\underline{(U_{[t]}^{0:C} - W_{[t]}^{0:C} S_{[t-1]}^C)}} \\
O_{[t]}^{0:C} &= Q_{[t]}^{0:C}S_{[t-1]}^C + (Q_{[t]}^{0:C}{K_{[t]}^{0:C}}^T \odot \text{Mask}) \underset{\text{Delta corrected Value}}{\underline{(U_{[t]}^{0:C} - W_{[t]}^{0:C} S_{[t-1]}^C)}}
\end{aligned}
$$

### The Challenge: How Do Gates Affect the Chunked Form?

In GatedDeltaNet, the recurrence is $(\text{Eq. 1})$:

$$
S_t = \alpha_t (I - \beta_t k_t^T k_t) S_{t-1} + \beta_t k_t^T v_t
$$

A naive approach would be to unroll this within a chunk and try to incorporate the gate factors $\alpha_i$ into the WY representation. However, this leads to gate-contaminated $U$ vectors, gate-weighted causal masks ($\Gamma$ instead of binary $M$), and a messy derivation. There is a much cleaner approach.

### Key Insight: The Degatified State

Define a "degatified" state by dividing out the cumulative gate product:

$$
\tilde{S}_{[t]}^r = \frac{S_{[t]}^r}{\gamma_{[t]}^r} \quad \text{where} \quad \gamma_{[t]}^r = \prod_{i=1}^r \alpha_{[t]}^i
\tag{Eq. 2}
$$

Here $\gamma_{[t]}^r$ is the **chunk-local** cumulative gate product within chunk $[t]$, so $\gamma_{[t]}^0 = 1$ (empty product). Inter-chunk state transitions are handled separately via the $\gamma_{[t]}^C$ factor at chunk boundaries.

Let's derive the recurrence for $\tilde{S}_{[t]}$. Starting from GatedDeltaNet's recurrence within chunk $[t]$:

$$
\begin{aligned}
S_{[t]}^r &= \alpha_{[t]}^r(I - \beta_{[t]}^r {k_{[t]}^r}^T k_{[t]}^r) S_{[t]}^{r-1} + \beta_{[t]}^r {k_{[t]}^r}^T v_{[t]}^r \\
\gamma_{[t]}^r \tilde{S}_{[t]}^r &= \alpha_{[t]}^r(I - \beta_{[t]}^r {k_{[t]}^r}^T k_{[t]}^r) \gamma_{[t]}^{r-1} \tilde{S}_{[t]}^{r-1} + \beta_{[t]}^r {k_{[t]}^r}^T v_{[t]}^r \\
\gamma_{[t]}^r \tilde{S}_{[t]}^r &= \gamma_{[t]}^r(I - \beta_{[t]}^r {k_{[t]}^r}^T k_{[t]}^r) \tilde{S}_{[t]}^{r-1} + \beta_{[t]}^r {k_{[t]}^r}^T v_{[t]}^r \quad (\text{since } \alpha_{[t]}^r \gamma_{[t]}^{r-1} = \gamma_{[t]}^r) \\
\tilde{S}_{[t]}^r &= (I - \beta_{[t]}^r {k_{[t]}^r}^T k_{[t]}^r) \tilde{S}_{[t]}^{r-1} + \frac{\beta_{[t]}^r}{\gamma_{[t]}^r} {k_{[t]}^r}^T v_{[t]}^r \\
\tilde{S}_{[t]}^r &= (I - \beta_{[t]}^r {k_{[t]}^r}^T k_{[t]}^r) \tilde{S}_{[t]}^{r-1} + \beta_{[t]}^r {k_{[t]}^r}^T \tilde{v}_{[t]}^r
\tag{Eq. 3}
\end{aligned}
$$

where $\tilde{v}_{[t]}^r = v_{[t]}^r / \gamma_{[t]}^r$ is the gate-rescaled value.

> **This is a standard DeltaNet recurrence!** The degatified state $\tilde{S}_{[t]}$ follows exactly the same update rule as DeltaNet, just with rescaled values $\tilde{v}_{[t]}^r = v_{[t]}^r / \gamma_{[t]}^r$. This means we can directly apply DeltaNet's chunked parallel form.

### Applying DeltaNet's Chunked Form to the Degatified State

Since $\tilde{S}_{[t]}$ follows a DeltaNet recurrence with values $\tilde{V}_{[t]}$ (rows $\tilde{v}_{[t]}^j = v_{[t]}^j/\gamma_{[t]}^j$), we apply the standard chunked form:

$$
\begin{aligned}
&\text{Auxiliary matrices} \\
A_{[t]}^{0:C} &= \text{tril}(-\text{diag}(\beta_{[t]}^{0:C})K_{[t]}^{0:C} {K_{[t]}^{0:C}}^T, -1) \\
W_{[t]}^{0:C} &= (I - A_{[t]}^{0:C})^{-1} \text{diag}(\beta_{[t]}^{0:C})K_{[t]}^{0:C} \\
\tilde{U}_{[t]}^{0:C} &= (I - A_{[t]}^{0:C})^{-1} \text{diag}(\beta_{[t]}^{0:C})\tilde{V}_{[t]}^{0:C}
\end{aligned}
$$

The degatified state update at the end of chunk $[t]$ is:

$$
\tilde{S}_{[t]}^{0:C} = S_{[t-1]}^C + {K_{[t]}^{0:C}}^T(\tilde{U}_{[t]}^{0:C} - W_{[t]}^{0:C} S_{[t-1]}^C)
$$

Note that $\tilde{S}_{[t]}^0 = S_{[t-1]}^C$ since $\gamma_{[t]}^0 = 1$ (empty product).

Converting back to the real state:

$$
S_{[t]}^{0:C} = \gamma^C_{[t]} \left( S_{[t-1]}^C + {K_{[t]}^{0:C}}^T(\tilde{U}_{[t]}^{0:C} - W_{[t]}^{0:C} S_{[t-1]}^C) \right)
\tag{Eq. 4}
$$

### Chunked Output

The output at position $r$ in chunk $[t]$ is:

$$
o_{[t]}^r = q_{[t]}^r S_{[t]}^r = q_{[t]}^r \gamma^r_{[t]} \tilde{S}_{[t]}^r
$$

Using the DeltaNet chunked output for $\tilde{S}$:

$$
\begin{aligned}
o_{[t]}^r &= \gamma^r_{[t]} q_{[t]}^r \left(S_{[t-1]}^C + \sum_{j=1}^r {k_{[t]}^j}^T (\tilde{u}_{[t]}^j - w_{[t]}^j S_{[t-1]}^C) \right)
\end{aligned}
$$

Define gate-scaled queries $\overleftarrow{q}_{[t]}^r = \gamma^r_{[t]} q_{[t]}^r$. In matrix form:

$$
O_{[t]}^{0:C} = \overleftarrow{Q}_{[t]}^{0:C}S_{[t-1]}^C + (\overleftarrow{Q}_{[t]}^{0:C}{K_{[t]}^{0:C}}^T \odot \text{Mask}) (\tilde{U}_{[t]}^{0:C} - W_{[t]}^{0:C} S_{[t-1]}^C)
\tag{Eq. 5}
$$

### Comparison with DeltaNet's Chunked Form

| Aspect | DeltaNet | GatedDeltaNet |
|--------|----------|---------------|
| Values for UT transform | $V^{0:C}$ | $\tilde{V}^{0:C}$ (rows $v^j/\gamma^j$) |
| $W$ matrix | $(I-A)^{-1}\text{diag}(\beta)K$ | **Same** (unchanged) |
| $U$ / $\tilde{U}$ matrix | $(I-A)^{-1}\text{diag}(\beta)V$ | $(I-A)^{-1}\text{diag}(\beta)\tilde{V}$ |
| Causal mask | Binary $M$ | **Same** binary $M$ (unchanged) |
| Queries in output | $Q^{0:C}$ | $\overleftarrow{Q}^{0:C}$ (rows $\gamma^r q^r$) |
| Inter-chunk state | $S_{[t-1]}^C$ | $\gamma^C_{[t]} \cdot \tilde{S}^{0:C}_{[t]}$ |
| Complexity | $O(LCd + Ld^2)$ | $O(LCd + Ld^2)$ (same) |
| Additional compute | None | $O(LC)$ for cumulative gate products + value rescaling |

> The degatified trick reveals that GatedDeltaNet's chunked form is almost identical to DeltaNet's. The WY representation and UT transform machinery are completely reused — only the input values are rescaled by $1/\gamma^j$, queries are scaled by $\gamma^r$, and the final chunk state $S_{[t]}^{0:C}$ is multiplied by $\gamma^C_{[t]}$. The causal mask stays binary, not gate-weighted.
