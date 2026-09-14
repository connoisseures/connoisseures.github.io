---
title: "LLM Pretraining Part 3: Multi-Head Attention"
date: 2026-09-14 02:00:00 +0000
categories: [LLM, Pretraining]
tags: [multi-head-attention, tensor-shapes, causal-mask, numpy, transformer]
description: A practical guide to multi-head attention, head splitting and concatenation, output projection, parameter counts, complexity, and tensor-shape reasoning.
math: true
---
## 1. Why use multiple attention heads?

A single attention head produces one learned retrieval pattern:

$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}\left(
\frac{QK^\top}{\sqrt{d_k}}+M
\right)V
$$

However, a token may need to retrieve several kinds of information simultaneously. For example, different attention patterns may be useful for grammatical relationships, nearby context, phrase structure, or semantic relationships.

Multi-head attention gives the model multiple independently learned retrieval mechanisms. The roles of heads are not manually assigned; they emerge during training because every head has its own learned query, key, and value projections.

## 2. Splitting the model dimension

Suppose:

```text
d_model = 512
num_heads = 8
```

Normally:

$$
d_{\text{head}}=\frac{d_{\text{model}}}{H}
$$

Therefore:

```text
d_head = 512 / 8 = 64
```

The combined width remains unchanged:

$$
H\times d_{\text{head}}=8\times64=512=d_{\text{model}}
$$

Instead of one attention head operating on all 512 dimensions, eight heads operate on 64-dimensional query, key, and value vectors.

## 3. Linear projections

Given:

$$
X\in\mathbb{R}^{B\times T\times d_{\text{model}}}
$$

the implementation computes:

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V
$$

where:

$$
W_Q,W_K,W_V\in
\mathbb{R}^{d_{\text{model}}\times d_{\text{model}}}
$$

Thus:

$$
Q,K,V\in\mathbb{R}^{B\times T\times d_{\text{model}}}
$$

For example:

```python
X.shape  == [2, 100, 512]
Wq.shape == [512, 512]

Q = X @ Wq
Q.shape  == [2, 100, 512]
```

At this point, the heads are still packed inside the last dimension.

## 4. Reshaping into heads

First split the final dimension:

```text
[B, T, d_model] → [B, T, H, d_head]
```

Then transpose the head and sequence dimensions:

```text
[B, T, H, d_head] → [B, H, T, d_head]
```

In code:

```python
Q = Q.reshape(B, T, num_heads, head_dim)
Q = Q.transpose(0, 2, 1, 3)
```

The same transformation is applied to `K` and `V`.

Putting the head dimension before the sequence dimension makes the batched matrix multiplication straightforward:

```text
Q:  [B, H, T, d_head]
Kᵀ: [B, H, d_head, T]
```

Therefore:

```text
Q @ Kᵀ: [B, H, T, T]
```

Every head receives its own $T\times T$ attention matrix.

## 5. Per-head attention

For head $h$:

$$
\operatorname{head}_h
=
\operatorname{softmax}\left(
\frac{Q_hK_h^\top}{\sqrt{d_{\text{head}}}}+M
\right)V_h
$$

For all heads together:

| Tensor | Shape |
|---|---|
| $Q$, $K$, $V$ | `[B, H, T, d_head]` |
| Attention scores | `[B, H, T, T]` |
| Attention weights | `[B, H, T, T]` |
| Per-head outputs | `[B, H, T, d_head]` |

The causal mask commonly has shape `[T, T]` or `[1, 1, T, T]`. It broadcasts across both the batch and head dimensions.

## 6. Concatenating the heads

After attention:

```text
heads: [B, H, T, d_head]
```

Move the head dimension next to `d_head`:

```python
combined = heads.transpose(0, 2, 1, 3)
```

Now:

```text
combined: [B, T, H, d_head]
```

Merge the final two dimensions:

```python
combined = combined.reshape(B, T, d_model)
```

This works because:

$$
H\times d_{\text{head}}=d_{\text{model}}
$$

## 7. Output projection

After concatenation, multi-head attention applies a learned output projection:

$$
Y=operatorname{Concat}(operatorname{head}_1,\ldots,
\operatorname{head}_H)W_O
$$

where:

$$
W_O\in\mathbb{R}^{d_{\text{model}}\times d_{\text{model}}}
$$

This projection mixes information across heads. It also produces an output compatible with the residual stream:

$$
X_{\text{out}}=X+\operatorname{MultiHeadAttention}(X)
$$

Both tensors have shape `[B, T, d_model]`.

## 8. Complete tensor flow

Suppose:

```text
B = 2
T = 100
d_model = 512
H = 8
d_head = 64
```

| Operation | Shape |
|---|---|
| Input $X$ | `[2, 100, 512]` |
| Projected $Q$, $K$, $V$ | `[2, 100, 512]` each |
| Split into heads | `[2, 8, 100, 64]` each |
| Attention scores | `[2, 8, 100, 100]` |
| Attention weights | `[2, 8, 100, 100]` |
| Per-head results | `[2, 8, 100, 64]` |
| Transpose for concatenation | `[2, 100, 8, 64]` |
| Merge heads | `[2, 100, 512]` |
| Output projection | `[2, 100, 512]` |

## 9. Minimal NumPy implementation

```python
import numpy as np


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def multi_head_attention(X, Wq, Wk, Wv, Wo, num_heads):
    """
    X:          [B, T, d_model]
    Wq, Wk, Wv: [d_model, d_model]
    Wo:         [d_model, d_model]
    """
    B, T, d_model = X.shape

    if d_model % num_heads != 0:
        raise ValueError("d_model must be divisible by num_heads")

    head_dim = d_model // num_heads

    # Linear projections: [B, T, d_model]
    Q = X @ Wq
    K = X @ Wk
    V = X @ Wv

    # Split heads: [B, T, H, D]
    Q = Q.reshape(B, T, num_heads, head_dim)
    K = K.reshape(B, T, num_heads, head_dim)
    V = V.reshape(B, T, num_heads, head_dim)

    # [B, T, H, D] -> [B, H, T, D]
    Q = Q.transpose(0, 2, 1, 3)
    K = K.transpose(0, 2, 1, 3)
    V = V.transpose(0, 2, 1, 3)

    # Per-head attention scores: [B, H, T, T]
    scores = Q @ K.transpose(0, 1, 3, 2)
    scores = scores / np.sqrt(head_dim)

    # [T, T] broadcasts over batch and head dimensions.
    future_mask = np.triu(
        np.ones((T, T), dtype=bool),
        k=1,
    )
    scores = np.where(future_mask, -np.inf, scores)

    # Normalize over key positions.
    weights = softmax(scores, axis=-1)

    # Retrieve values: [B, H, T, D]
    heads = weights @ V

    # Concatenate heads: [B, T, H, D] -> [B, T, d_model]
    combined = heads.transpose(0, 2, 1, 3)
    combined = combined.reshape(B, T, d_model)

    # Mix information across heads.
    output = combined @ Wo

    return output, weights
```

## 10. Separate head matrices versus one large matrix

Mathematically, each head has its own projections:

$$
W_Q^{(h)},W_K^{(h)},W_V^{(h)}
$$

Efficient implementations pack the per-head matrices into three larger matrices:

$$
W_Q,W_K,W_V\in
\mathbb{R}^{d_{\text{model}}\times d_{\text{model}}}
$$

The large matrix multiplication is followed by a reshape. These descriptions are mathematically equivalent, but the packed form uses hardware more efficiently.

## 11. Parameter count

Ignoring biases, multi-head attention has four projection matrices:

$$
W_Q,\quad W_K,\quad W_V,\quad W_O
$$

Each has shape `[d_model, d_model]`, so the approximate parameter count is:

$$
4d_{\text{model}}^2
$$

For $d_{\text{model}}=512$:

$$
4\times512^2=1{,}048{,}576
$$

Changing the number of heads does not normally change this parameter count when `d_model` remains fixed. It changes how the same projected width is partitioned:

```text
8 heads  × 64 dimensions = 512
16 heads × 32 dimensions = 512
```

More heads provide more independently learned attention patterns, while every individual head has a smaller feature dimension.

## 12. Computational complexity

The attention-score computation costs approximately:

$$
O(BHT^2d_{\text{head}})
$$

Because $Hd_{\text{head}}=d_{\text{model}}$, this becomes:

$$
O(BT^2d_{\text{model}})
$$

Multi-head attention therefore retains the quadratic dependence on sequence length. Every head constructs a $T\times T$ score matrix.

## 13. Interview check and reviewed answers

Assume:

```text
X.shape = [3, 256, 768]
num_heads = 12
```

### 1. What is `head_dim`?

$$
d_{\text{head}}=768/12=64
$$

### 2. What is the shape of $Q$ immediately after projection?

```text
X:  [3, 256, 768]
WQ: [768, 768]
Q:  [3, 256, 768]
```

### 3. What is its shape after splitting and transposing?

```text
Before transpose: [3, 256, 12, 64]
After transpose:  [3, 12, 256, 64]
```

### 4. What is the attention-score shape?

```text
Q:  [3, 12, 256, 64]
Kᵀ: [3, 12, 64, 256]

Q @ Kᵀ → [3, 12, 256, 256]
```

Every batch item and head has its own $256\times256$ attention matrix.

### 5. What is the per-head output shape before concatenation?

```text
weights: [3, 12, 256, 256]
V:       [3, 12, 256, 64]

weights @ V → [3, 12, 256, 64]
```

### 6. What is the shape after concatenation?

First transpose:

```text
[3, 12, 256, 64] → [3, 256, 12, 64]
```

Then merge the final dimensions:

```text
[3, 256, 12 × 64] = [3, 256, 768]
```

### 7. What happens when the number of heads changes to 24?

With `d_model=768`:

$$
d_{\text{head}}=768/24=32
$$

The principal projection matrices remain `[768, 768]`, so the approximate parameter count stays:

$$
4\times768^2=2{,}359{,}296
$$

Only the partition changes:

```text
12 heads × 64 dimensions = 768
24 heads × 32 dimensions = 768
```

## Key takeaway

Multi-head attention projects the residual representation into queries, keys, and values; splits each projection into independently learned heads; calculates one causal attention matrix per head; concatenates the retrieved values; and applies an output projection to mix information across heads.

The central shape flow is:

```text
[B, T, d_model]
    → [B, H, T, d_head]
    → scores [B, H, T, T]
    → heads [B, H, T, d_head]
    → [B, T, d_model]
```
