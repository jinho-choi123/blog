---
title: "FLA's GatedDeltaNet: Gate-Aware Chunked Formulation"
description: "How Flash Linear Attention implements GatedDeltaNet's chunked parallel algorithm with numerical stability, and proof of equivalence with the degatified form."
publishDate: "04 May 2026"
tags: ["AI"]
---

## Prerequisites

This post assumes familiarity with:

- [GatedDeltaNet: Improving Mamba2 with Delta Rule](/post/ai/gated-delta-net/) — the degatified chunked form (Eq. 4-5)
- [DeltaNet: WY Representation and UT Transform](/post/ai/delta-net/) — the underlying machinery

We use the same notation established in those posts. In particular, recall the degatified chunked form:

$$
S_{[t]}^{0:C} = \gamma^C_{[t]} (S_{[t-1]}^C + {K_{[t]}^{0:C}}^T(\tilde{U}_{[t]}^{0:C} - W_{[t]}^{0:C} S_{[t-1]}^C))
\tag{Eq. 4}
$$

$$
O_{[t]}^{0:C} = \overleftarrow{Q}_{[t]}^{0:C} S_{[t-1]}^C + (\overleftarrow{Q}_{[t]}^{0:C}{K_{[t]}^{0:C}}^T \odot M)(\tilde{U}_{[t]}^{0:C} - W_{[t]}^{0:C} S_{[t-1]}^C)
\tag{Eq. 5}
$$

where $\tilde{V} = V/\gamma$, $\overleftarrow{Q}^r = \gamma^r q^r$, and $W$, $\tilde{U}$ are computed via the standard (gate-free) UT transform on rescaled values.

## FLA's Gate-Aware UT Transform

[FLA (Flash Linear Attention)](https://github.com/fla-org/flash-linear-attention) is a state-of-the-art efficient linear attention implementation that supports various linear attention variants, including GatedDeltaNet. FLA's internal formulation is mathematically equivalent to the degatified form derived above, but distributes the gate factors differently for better numerical stability.

Instead of removing the gate via degatification ($\tilde{V} = V / \gamma$), FLA bakes the gate ratios directly into the $A$ matrix:

$$
\begin{aligned}
&\text{Gate-aware auxiliary matrices} \\
\tilde{A}_{[t]}^{0:C} &= \text{tril}(-\text{diag}(\beta_{[t]}^{0:C})(\Gamma_{[t]} \odot K_{[t]}^{0:C} {K_{[t]}^{0:C}}^T), -1) \\
\tilde{W}_{[t]}^{0:C} &= (I - \tilde{A}_{[t]}^{0:C})^{-1} \text{diag}(\beta_{[t]}^{0:C} \gamma_{[t]}^{0:C})K_{[t]}^{0:C} \\
\tilde{U}_{[t]}^{0:C} &= (I - \tilde{A}_{[t]}^{0:C})^{-1} \text{diag}(\beta_{[t]}^{0:C})V_{[t]}^{0:C}
\end{aligned}
$$

where $\Gamma_{ij} = \gamma_{[t]}^i / \gamma_{[t]}^j$ for $i \geq j$ is the inter-position gate decay ratio (with $\Gamma_{ii} = 1$), computed stably in log space as $\exp(\text{cumsum}(\log\alpha)_i - \text{cumsum}(\log\alpha)_j)$. The strict-lower-triangular mask in $\tilde{A}$ is enforced by the explicit $\text{tril}(\cdot, -1)$ operator, so the diagonal value of $\Gamma$ is irrelevant there.

Note three key differences from the degatified form:

1. **$\tilde{A}$ contains $\Gamma$**: gate ratios are baked into the UT transform
2. **$\tilde{U}$ uses original $V$**: no division by $\gamma$ (avoids blow-up when $\gamma \to 0$)
3. **$\tilde{W}$ uses $\beta\gamma K$**: an extra $\gamma$ factor multiplies the keys

The gate-scaled variables use arrow notation from the paper:

$$
\begin{aligned}
\overleftarrow{q}_{[t]}^r &= \gamma_{[t]}^r \cdot q_{[t]}^r \quad \text{(query decayed to chunk start)} \\
\overrightarrow{k}_{[t]}^r &= \frac{\gamma_{[t]}^C}{\gamma_{[t]}^r} \cdot k_{[t]}^r \quad \text{(key decayed to chunk end)}
\end{aligned}
$$

State update and output:

$$
S_{[t]}^{0:C} = \gamma^C_{[t]} S_{[t-1]}^C + {\overrightarrow{K}_{[t]}^{0:C}}^T(\tilde{U}_{[t]}^{0:C} - \tilde{W}_{[t]}^{0:C} S_{[t-1]}^C)
\tag{Eq. 6}
$$

$$
O_{[t]}^{0:C} = \overleftarrow{Q}_{[t]}^{0:C}S_{[t-1]}^C + (Q_{[t]}^{0:C}{K_{[t]}^{0:C}}^T \odot \Gamma_{[t]}) (\tilde{U}_{[t]}^{0:C} - \tilde{W}_{[t]}^{0:C} S_{[t-1]}^C)
\tag{Eq. 7}
$$

where $\Gamma_{[t],rj} = \gamma_{[t]}^r / \gamma_{[t]}^j$ for $r \geq j$ is the gate-aware causal mask (replacing the binary mask $M$).

## Proof of Equivalence

We now prove that FLA's formulation (Eq. 6-7) is algebraically equivalent to the degatified form (Eq. 4-5).

### Step 1: The Key Identity

The gate-aware and standard $A$ matrices are related by a diagonal similarity transform:

$$
(I - \tilde{A}) = \text{diag}(\gamma)(I - A)\text{diag}(1/\gamma)
$$

**Proof.** For diagonal entries ($i = j$): both sides equal $1$. For off-diagonal entries ($i > j$):

$$
\begin{aligned}
\text{LHS}_{ij} &= (I - \tilde{A})_{ij} = \beta_i \frac{\gamma_i}{\gamma_j} k_i {k_j}^T \\
\text{RHS}_{ij} &= \gamma_i \cdot (I-A)_{ij} \cdot \frac{1}{\gamma_j} = \gamma_i \cdot \beta_i k_i {k_j}^T \cdot \frac{1}{\gamma_j} = \beta_i \frac{\gamma_i}{\gamma_j} k_i {k_j}^T \quad \checkmark
\end{aligned}
$$

Taking the inverse of both sides:

$$
\boxed{(I - \tilde{A})^{-1} = \text{diag}(\gamma)(I - A)^{-1}\text{diag}(1/\gamma)}
\tag{Eq. 8}
$$

### Step 2: Relating $\tilde{U}_{\text{FLA}}$ and $\tilde{U}_{\text{degat}}$

$$
\begin{aligned}
\tilde{U}_{\text{FLA}} &= (I-\tilde{A})^{-1}\text{diag}(\beta)V \\
&= \text{diag}(\gamma)(I-A)^{-1}\text{diag}(1/\gamma) \cdot \text{diag}(\beta) \cdot V \\
&= \text{diag}(\gamma)(I-A)^{-1}\text{diag}(\beta) \cdot \text{diag}(1/\gamma) V \\
&= \text{diag}(\gamma) \underbrace{(I-A)^{-1}\text{diag}(\beta)\tilde{V}}_{\tilde{U}_{\text{degat}}}
\end{aligned}
$$

Row-wise: $\tilde{u}_{\text{FLA}}^j = \gamma^j \cdot \tilde{u}_{\text{degat}}^j$

### Step 3: Relating $\tilde{W}_{\text{FLA}}$ and $W_{\text{degat}}$

$$
\begin{aligned}
\tilde{W}_{\text{FLA}} &= (I-\tilde{A})^{-1}\text{diag}(\beta\gamma)K \\
&= \text{diag}(\gamma)(I-A)^{-1}\text{diag}(1/\gamma) \cdot \text{diag}(\beta\gamma) \cdot K \\
&= \text{diag}(\gamma)(I-A)^{-1}\text{diag}(\beta) \cdot K \\
&= \text{diag}(\gamma) \underbrace{(I-A)^{-1}\text{diag}(\beta)K}_{W_{\text{degat}}}
\end{aligned}
$$

Row-wise: $\tilde{w}_{\text{FLA}}^j = \gamma^j \cdot w_{\text{degat}}^j$

> The $\gamma$ factors in $\tilde{U}_{\text{FLA}}$ and $\tilde{W}_{\text{FLA}}$ cancel with the $1/\gamma$ factors in $\overrightarrow{K}$ and $\Gamma$, yielding the same final result as the degatified form.

### Step 4: State Update Equivalence

The delta-corrected value in FLA is a $\gamma$-scaled version of the degatified one:

$$
\tilde{U}_{\text{FLA}} - \tilde{W}_{\text{FLA}} S_{[t-1]}^C = \text{diag}(\gamma)(\tilde{U}_{\text{degat}} - W_{\text{degat}} S_{[t-1]}^C)
$$

Substituting into FLA's state update (Eq. 6):

$$
\begin{aligned}
S_{[t]}^{0:C} &= \gamma^C S_{[t-1]}^C + \overrightarrow{K}^T \cdot \text{diag}(\gamma) \cdot (\tilde{U}_{\text{degat}} - W_{\text{degat}} S_{[t-1]}^C)
\end{aligned}
$$

Since $\overrightarrow{K}$ has rows $(\gamma^C/\gamma^j) k^j$, the product $\overrightarrow{K}^T \cdot \text{diag}(\gamma)$ has columns $(\gamma^C/\gamma^j) \cdot \gamma^j \cdot k^{jT} = \gamma^C k^{jT}$, giving:

$$
\overrightarrow{K}^T \cdot \text{diag}(\gamma) = \gamma^C K^T
$$

Therefore:

$$
S_{[t]}^{0:C} = \gamma^C S_{[t-1]}^C + \gamma^C K^T(\tilde{U}_{\text{degat}} - W_{\text{degat}} S_{[t-1]}^C) = \gamma^C(S_{[t-1]}^C + K^T(\tilde{U}_{\text{degat}} - W_{\text{degat}} S_{[t-1]}^C)) \quad \checkmark
$$

This is exactly Eq. 4.

### Step 5: Output Equivalence

Substituting into FLA's output (Eq. 7), the intra-chunk attention term becomes:

$$
(QK^T \odot \Gamma) \cdot \text{diag}(\gamma) \cdot (\tilde{U}_{\text{degat}} - W_{\text{degat}} S_{[t-1]}^C)
$$

For entry $(r, j)$: $(q^r {k^j}^T) \cdot (\gamma^r/\gamma^j) \cdot \gamma^j = \gamma^r (q^r {k^j}^T)$. In matrix form:

$$
(QK^T \odot \Gamma) \cdot \text{diag}(\gamma) = \overleftarrow{Q}{K}^T \odot M
$$

where the gate-weighted mask $\Gamma$ combined with $\text{diag}(\gamma)$ collapses to a binary mask $M$ times the gate-scaled query $\overleftarrow{Q}$. Therefore:

$$
O_{[t]}^{0:C} = \overleftarrow{Q}S_{[t-1]}^C + (\overleftarrow{Q}{K}^T \odot M)(\tilde{U}_{\text{degat}} - W_{\text{degat}} S_{[t-1]}^C) \quad \checkmark
$$

This is exactly Eq. 5.

## Why FLA Prefers This Form: Numerical Stability

Both forms compute the same result, but the intermediate values differ dramatically:

| | Degatified Form | FLA (Gate-Aware) |
| --- | --- | --- |
| $\tilde{V} = V/\gamma$ | $\gamma \to 0 \Rightarrow$ values blow up | Uses original $V$ |
| $(I-A)^{-1}$ entries | $O(1)$ | $O(1)$ (Γ provides damping) |
| Delta-corrected value | Can reach $10^{13}\text{--}10^{30}$ | Stays $O(1)$ |
| Final output | $(\text{tiny } \gamma) \times (\text{huge value})$ = cancellation | $O(1) \times O(1)$ = clean |
| fp32 with $C \geq 30$ | Fails | Passes |

The degatified form divides by $\gamma$ then multiplies back by $\gamma$, which is mathematically exact but numerically catastrophic when $\gamma$ is small. FLA avoids this by keeping $\gamma$ as ratios ($\gamma_i/\gamma_j \leq 1$) throughout, never dividing by a small number.
