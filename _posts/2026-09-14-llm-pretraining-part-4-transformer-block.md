---
title: "LLM Pretraining Part 4: Transformer Blocks, RMSNorm, and SwiGLU"
date: 2026-09-14 03:00:00 +0000
categories: [LLM, Pretraining]
tags: [transformer-block, rmsnorm, swiglu, gelu, residual-connection]
description: A practical guide to pre-norm transformer blocks, residual streams, RMSNorm, ReLU, GELU, SiLU, SwiGLU, tensor shapes, and interview reasoning.
math: true
---
## 1. Modern decoder-only transformer block

A common pre-normalization transformer block performs:

```text
X
→ RMSNorm
→ Causal multi-head attention
→ Residual addition
→ RMSNorm
→ SwiGLU feedforward network
→ Residual addition
```

Mathematically:

$$
X'=X+\operatorname{Attention}(\operatorname{RMSNorm}(X))
$$

$$
Y=X'+\operatorname{FFN}(\operatorname{RMSNorm}(X'))
$$

The input and output shapes remain `[B, T, d_model]`.

## 2. The residual stream

The main tensor flowing through the transformer is the **residual stream**:

$$
X\in\mathbb{R}^{B\times T\times d_{\text{model}}}
$$

Every sublayer reads from this stream, calculates an update, and adds that update back:

$$
X_{\text{new}}=X+\Delta X
$$

A useful mental model is:

> Each sublayer proposes an update to the existing token representation rather than reconstructing it from scratch.

For a residual addition to work, the two tensors must have the same shape:

```text
residual:         [B, T, d_model]
attention output: [B, T, d_model]
FFN output:       [B, T, d_model]
```

Residual connections also create a direct gradient path. If:

$$
Y=X+F(X)
$$

then:

$$
\frac{\partial Y}{\partial X}
=I+\frac{\partial F(X)}{\partial X}
$$

The identity term helps gradients propagate through deep networks.

## 3. LayerNorm and RMSNorm

LayerNorm centers and scales each token representation:

$$
\operatorname{LayerNorm}(x)
=
\gamma\odot\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta
$$

RMSNorm controls its magnitude without subtracting its mean:

$$
\operatorname{RMS}(x)
=
\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2+\epsilon}
$$

$$
\operatorname{RMSNorm}(x)
=
\gamma\odot\frac{x}{\operatorname{RMS}(x)}
$$

| Operation | LayerNorm | RMSNorm |
|---|---:|---:|
| Subtracts the mean | Yes | No |
| Divides by a scale statistic | Yes | Yes |
| Learnable scale | Yes | Yes |
| Learnable bias | Common | Often omitted |
| Computational work | More | Slightly less |

## 4. RMSNorm dimensions

Given:

```text
X: [B, T, d_model]
```

RMSNorm calculates the mean square over the final hidden dimension:

```python
rms = np.sqrt(
    np.mean(X**2, axis=-1, keepdims=True) + eps
)
```

The shapes are:

```text
X:     [B, T, d_model]
rms:   [B, T, 1]
gamma: [d_model]
```

Both `rms` and `gamma` broadcast against `X`, and the output remains `[B, T, d_model]`.

```python
def rms_norm(X, gamma, eps=1e-6):
    inverse_rms = 1.0 / np.sqrt(
        np.mean(X**2, axis=-1, keepdims=True) + eps
    )
    return X * inverse_rms * gamma
```

## 5. Pre-norm versus post-norm

The original Transformer used post-normalization:

$$
Y=\operatorname{Norm}(X+F(X))
$$

```python
Y = norm(X + sublayer(X))
```

Many modern LLMs use pre-normalization:

$$
Y=X+F(\operatorname{Norm}(X))
$$

```python
Y = X + sublayer(norm(X))
```

In pre-norm, the sublayer receives normalized input, but the skip branch carries the original residual stream directly to the addition. This clean identity path generally improves optimization stability in deep transformers.

## 6. The feedforward network

Attention mixes information across token positions. The feedforward network transforms the features within each token independently.

A basic FFN is:

$$
\operatorname{FFN}(x)=\phi(xW_1)W_2
$$

Shape flow:

```text
Input:  [B, T, d_model]
Hidden: [B, T, d_ff]
Output: [B, T, d_model]
```

The same FFN weights are applied independently at every position:

```python
hidden = activation(X @ W1)
output = hidden @ W2
```

The expanded hidden dimension provides a larger intermediate feature space for nonlinear transformations.

## 7. ReLU

ReLU applies a hard threshold:

$$
\operatorname{ReLU}(x)=\max(0,x)
$$

```python
def relu(x):
    return np.maximum(0, x)
```

| $x$ | ReLU($x$) |
|---:|---:|
| -2 | 0 |
| -1 | 0 |
| 0 | 0 |
| 1 | 1 |
| 2 | 2 |

ReLU is simple and efficient, but it completely removes negative values. A unit that continually receives negative inputs can have zero output and zero gradient.

## 8. GELU

GELU smoothly scales inputs instead of using a hard cutoff:

$$
\operatorname{GELU}(x)
\approx
\frac{x}{2}\left[1+\tanh\left(
\sqrt{\frac{2}{\pi}}(x+0.044715x^3)
\right)\right]
$$

```python
def gelu(x):
    return 0.5 * x * (
        1.0
        + np.tanh(
            np.sqrt(2.0 / np.pi)
            * (x + 0.044715 * x**3)
        )
    )
```

| $x$ | ReLU($x$) | GELU($x$) |
|---:|---:|---:|
| -2 | 0 | -0.046 |
| -1 | 0 | -0.159 |
| 0 | 0 | 0 |
| 1 | 1 | 0.841 |
| 2 | 2 | 1.955 |

GELU is smooth around zero, permits small negative outputs, and approaches the identity function for large positive inputs.

## 9. SiLU

SwiGLU uses SiLU as its gate activation:

$$
\operatorname{SiLU}(x)=x\sigma(x),
\qquad
\sigma(x)=\frac{1}{1+e^{-x}}
$$

```python
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def silu(x):
    return x * sigmoid(x)
```

| $x$ | Sigmoid($x$) | SiLU($x$) |
|---:|---:|---:|
| -2 | 0.119 | -0.238 |
| -1 | 0.269 | -0.269 |
| 0 | 0.500 | 0 |
| 1 | 0.731 | 0.731 |
| 2 | 0.881 | 1.762 |

Like GELU, SiLU is smooth and permits small negative outputs.

## 10. SwiGLU

SwiGLU is a gated FFN rather than a single scalar activation. It uses two independently learned projections:

$$
g=xW_{\text{gate}},\qquad u=xW_{\text{up}}
$$

The gate and value branches are combined elementwise:

$$
h=\operatorname{SiLU}(g)\odot u
$$

Then the result is projected back to the model width:

$$
\operatorname{SwiGLU}(x)
=
[\operatorname{SiLU}(xW_{\text{gate}})
\odot(xW_{\text{up}})]W_{\text{down}}
$$

The projection shapes are:

```text
W_gate: [d_model, d_ff]
W_up:   [d_model, d_ff]
W_down: [d_ff, d_model]
```

```python
def swiglu(X, W_gate, W_up, W_down):
    gate = silu(X @ W_gate)  # [B, T, d_ff]
    value = X @ W_up         # [B, T, d_ff]
    hidden = gate * value    # elementwise multiplication
    return hidden @ W_down   # [B, T, d_model]
```

### Numerical example

Suppose the learned branches produce:

```python
gate_pre = np.array([-1.0, 0.0, 2.0])
value = np.array([3.0, -4.0, 0.5])
```

Then:

```text
SiLU(gate_pre) ≈ [-0.269, 0.000, 1.762]

hidden = SiLU(gate_pre) × value
       ≈ [-0.807, 0.000, 0.881]
```

The gate can suppress, preserve, scale, or change the sign of intermediate features.

## 11. Comparing ReLU, GELU, and SwiGLU

| Property | ReLU | GELU | SwiGLU |
|---|---|---|---|
| Basic form | $\max(0,x)$ | $x\Phi(x)$ | $\operatorname{SiLU}(g)\odot u$ |
| Input branches | One | One | Two |
| Smooth | No | Yes | Yes |
| Negative outputs possible | No | Yes | Yes |
| Learned gating | No | No | Yes |
| FFN projections | Usually 2 | Usually 2 | Usually 3 |

For the same `d_ff`, SwiGLU has three FFN matrices rather than two:

$$
N_{\text{SwiGLU}}=3d_{\text{model}}d_{\text{ff}}
$$

$$
N_{\text{standard}}=2d_{\text{model}}d_{\text{ff}}
$$

Architectures often choose a smaller SwiGLU hidden dimension to keep parameter and compute budgets comparable.

## 12. Complete pre-norm block

```python
def transformer_block(X, attention, ffn, norm1, norm2):
    X = X + attention(norm1(X))
    X = X + ffn(norm2(X))
    return X
```

The components have complementary roles:

| Component | Primary role |
|---|---|
| Attention | Moves information between token positions |
| FFN/SwiGLU | Transforms features independently at each position |
| RMSNorm | Controls the scale of sublayer inputs |
| Residual connection | Preserves information and gradient flow |

A useful summary is:

```text
Attention: Which other tokens should I read?
FFN:       How should I transform what I now know?
```

## 13. Interview check and reviewed answers

Assume:

```text
X.shape = [4, 256, 1024]
d_ff = 2816
```

### 1. Over which axis does RMSNorm calculate the mean square?

The hidden dimension is axis 2, normally written as `axis=-1`:

```python
mean_square = np.mean(X**2, axis=-1, keepdims=True)
```

### 2. What is the RMS shape with `keepdims=True`?

```text
[4, 256, 1]
```

It broadcasts across the 1024 features.

### 3. What are the SwiGLU matrix shapes?

```text
W_gate: [1024, 2816]
W_up:   [1024, 2816]
W_down: [2816, 1024]
```

### 4. What are the gate and value branch shapes?

Both are:

```text
[4, 256, 2816]
```

Applying SiLU does not change the gate branch's shape.

### 5. Which SwiGLU operation is elementwise?

The Hadamard product between the activated gate and value branches:

```python
hidden = silu(X @ W_gate) * (X @ W_up)
```

The projections use matrix multiplication; `gate * value` is elementwise multiplication.

### 6. Why must the final output be `[4, 256, 1024]`?

It must match the residual stream so the addition is valid:

```text
residual:       [4, 256, 1024]
SwiGLU output:  [4, 256, 1024]
```

### 7. Does the pre-norm residual branch pass through RMSNorm?

No. The sublayer branch receives `RMSNorm(X)`, while the residual skip branch carries the original `X` directly to the addition.

## Key takeaway

A modern transformer block alternates between cross-token communication through attention and per-token feature transformation through a gated FFN. RMSNorm stabilizes sublayer inputs, while residual connections preserve a direct path for representations and gradients. SwiGLU improves the FFN's expressiveness by using one learned branch to gate another before projecting back to the residual width.
