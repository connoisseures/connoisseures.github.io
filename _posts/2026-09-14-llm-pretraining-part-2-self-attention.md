---
title: "LLM Pretraining Part 2: Self-Attention and Causal Masking"
date: 2026-09-14 01:00:00 +0000
categories: [LLM, Pretraining]
tags: [self-attention, query-key-value, causal-mask, softmax, numpy]
description: A detailed guide to query-key-value attention, score scaling, causal masking, stable softmax, tensor shapes, and a NumPy implementation.
math: true
---
## 1. Why does an LLM need attention?

Part 1 introduced the causal language-modeling objective:

$$
P(w_{t+1}\mid w_{\le t})
$$

To predict the next token, the model must identify which earlier tokens contain useful information. **Self-attention** is the mechanism that retrieves and combines that information.

For each token position $i$, attention produces a weighted combination of information from the available positions:

$$
a_i=\sum_{j\le i}\alpha_{ij}v_j
$$

where:

- $i$ is the current query position.
- $j$ is a context position.
- $\alpha_{ij}$ is the attention weight from position $i$ to position $j$.
- $v_j$ is the information retrieved from position $j$.

In a causal LLM, $j\le i$ because the model must not use future tokens.

For example:

```text
The animal didn't cross the road because it was tired.
```

When processing `it`, a head may retrieve information from `animal` to build a context-sensitive representation.

## 2. Queries, keys, and values

Given a token representation $x_i$, the model creates three vectors through learned linear projections:

$$
q_i=x_iW_Q,\qquad k_i=x_iW_K,\qquad v_i=x_iW_V
$$

| Vector | Role |
|---|---|
| Query $q_i$ | What information does the current position need? |
| Key $k_j$ | What kind of information does position $j$ contain? |
| Value $v_j$ | What information will position $j$ contribute? |

A useful information-retrieval analogy is:

```text
Query → search request
Key   → description used for matching
Value → content returned by the match
```

Queries and keys determine relevance. Values contain the information actually transferred.

## 3. Query–key matching

The relevance of position $j$ to position $i$ is measured with a dot product:

$$
q_i\cdot k_j
$$

A larger score represents a stronger match. For all positions simultaneously:

$$
S=QK^\top
$$

If:

$$
Q,K\in\mathbb{R}^{T\times d_k}
$$

then:

$$
[T,d_k][d_k,T]=[T,T]
$$

Every element $S_{ij}$ answers:

> How relevant is the key at position $j$ to the query at position $i$?

Therefore:

- Each row corresponds to one query position.
- Each column corresponds to one key position.

## 4. Why scale by $\sqrt{d_k}$?

Scaled dot-product attention uses:

$$
S=\frac{QK^\top}{\sqrt{d_k}}
$$

The dot product is a sum over the query/key feature dimension:

$$
q\cdot k=\sum_{r=1}^{d_k}q_rk_r
$$

If the components have approximately unit variance, the dot product's variance grows roughly with $d_k$, and its standard deviation grows with $\sqrt{d_k}$. Without scaling, large scores can push softmax into an almost one-hot distribution, producing extremely small gradients for most entries.

Dividing by $\sqrt{d_k}$ keeps score magnitudes more stable:

```python
scores = scores / np.sqrt(Q.shape[-1])
```

The divisor is the query/key head dimension, not the sequence length.

## 5. Causal masking

During training, all sequence positions are processed in parallel. The causal mask prevents a position from attending to future tokens and seeing the answer it should predict.

For a sequence of length four, the allowed pattern is:

$$
\begin{bmatrix}
1&0&0&0\\
1&1&0&0\\
1&1&1&0\\
1&1&1&1
\end{bmatrix}
$$

Future positions receive negative infinity before softmax:

$$
M_{ij}=
\begin{cases}
0,&j\le i\\
-\infty,&j>i
\end{cases}
$$

The masked scores are:

$$
S_{\text{masked}}=\frac{QK^\top}{\sqrt{d_k}}+M
$$

In NumPy:

```python
seq_len = Q.shape[0]

future_mask = np.triu(
    np.ones((seq_len, seq_len), dtype=bool),
    k=1,
)

scores[future_mask] = -np.inf
```

`k=1` selects positions strictly above the diagonal. The diagonal remains visible because the current input token is valid context for predicting the following token.

### Why not multiply masked scores by zero?

Softmax exponentiates every score. Setting a masked score to zero is insufficient because:

$$
e^0=1
$$

The supposedly masked position could still receive probability. Setting it to negative infinity works because:

$$
e^{-\infty}=0
$$

For example:

```python
# Incorrect: the masked score becomes 0 and may receive probability.
scores = np.array([-2.0, 0.0])

# Correct: the masked position receives exactly zero probability.
scores = np.array([-2.0, -np.inf])
```

## 6. Stable, row-wise softmax

Softmax converts each row of scores into attention weights:

$$
A_{ij}=\frac{\exp(S_{ij})}{\sum_m\exp(S_{im})}
$$

A numerically stable implementation subtracts the maximum from every row:

```python
scores = scores - np.max(scores, axis=-1, keepdims=True)

weights = np.exp(scores)
weights = weights / np.sum(weights, axis=-1, keepdims=True)
```

Subtracting the same constant from every entry does not change softmax:

$$
\operatorname{softmax}(z)=\operatorname{softmax}(z-c)
$$

It prevents overflow when computing `np.exp(scores)`.

Softmax operates over the **key-position dimension**. For scores shaped `[B, T, T]`, this is `axis=2`, equivalently `axis=-1`:

```python
weights = softmax(scores, axis=-1)
```

For each batch item and query position:

```python
weights.sum(axis=-1) == 1
```

## 7. Retrieving values

After calculating the weights, attention retrieves a weighted combination of values:

$$
H=AV
$$

If:

$$
A:[T,T],\qquad V:[T,d_v]
$$

then:

$$
H:[T,d_v]
$$

For position $i$:

$$
h_i=\sum_{j\le i}A_{ij}v_j
$$

The complete attention equation is:

$$
\boxed{
\operatorname{Attention}(Q,K,V)=
\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V
}
$$

A compact way to remember the operation is:

```text
match with QKᵀ → normalize with softmax → retrieve with AV
```

## 8. Batched tensor shapes

Given an input batch:

$$
X:[B,T,d_{\text{model}}]
$$

for one attention head:

$$
Q,K:[B,T,d_k],\qquad V:[B,T,d_v]
$$

The complete shape flow is:

| Tensor | Shape |
|---|---|
| $Q$ | `[B, T, d_k]` |
| $K^\top$ | `[B, d_k, T]` |
| Scores $QK^\top$ | `[B, T, T]` |
| Attention weights | `[B, T, T]` |
| $V$ | `[B, T, d_v]` |
| Output $AV$ | `[B, T, d_v]` |

In NumPy, the key transpose preserves the batch dimension and exchanges only the final two dimensions:

```python
scores = Q @ K.transpose(0, 2, 1)
```

## 9. Complete NumPy implementation

```python
import numpy as np


def causal_attention(Q, K, V):
    """
    Q: [B, T, d_k]
    K: [B, T, d_k]
    V: [B, T, d_v]

    Returns:
        output:  [B, T, d_v]
        weights: [B, T, T]
    """
    d_k = Q.shape[-1]
    seq_len = Q.shape[1]

    # Compare every query with every key.
    scores = Q @ K.transpose(0, 2, 1)
    scores = scores / np.sqrt(d_k)

    # [T, T] broadcasts across the batch dimension.
    future_mask = np.triu(
        np.ones((seq_len, seq_len), dtype=bool),
        k=1,
    )
    scores = np.where(future_mask, -np.inf, scores)

    # Stable softmax over key positions.
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights = weights / np.sum(weights, axis=-1, keepdims=True)

    # Retrieve a weighted combination of value vectors.
    output = weights @ V

    return output, weights
```

## 10. Connection to next-token prediction

Consider:

```text
Input:  So long and thanks for
Target: long and thanks for all
```

The representation at the final input position `for` may attend to all input tokens:

```text
So, long, and, thanks, for
```

It cannot attend to `all`, because `all` is the target and is not part of the input. After several transformer layers, the contextual representation at `for` is passed to the language-modeling head:

```python
logits = hidden_state @ output_projection
```

These logits represent:

$$
P(\text{next token}\mid\text{So long and thanks for})
$$

Cross-entropy compares this distribution with the correct target `all`.

## 11. Interview check and reviewed answers

Assume:

```python
Q.shape == (4, 128, 64)
K.shape == (4, 128, 64)
V.shape == (4, 128, 64)
```

### Question 1: What is the score shape?

Answer:

```python
(4, 128, 128)
```

Reason:

```text
Q:                  [4, 128, 64]
K.transpose(0,2,1): [4, 64, 128]
scores:             [4, 128, 128]
```

For each of the four sequences, every one of the 128 queries is compared with all 128 keys.

### Question 2: Why divide by $\sqrt{64}$ rather than $\sqrt{128}$?

The inner product is taken over the query/key feature dimension $d_k=64$. Its variance grows approximately with $d_k$, so dividing by $\sqrt{d_k}$ stabilizes the score scale before softmax. The sequence length `128` controls how many comparisons occur, but it is not the dimension being summed over.

### Question 3: Along which axis should softmax operate?

For scores shaped `[B, query_position, key_position]`, softmax operates over the key positions:

```python
weights = softmax(scores, axis=2)
```

Using `axis=-1` is equivalent and more robust:

```python
weights = softmax(scores, axis=-1)
```

### Question 4: Why use negative infinity instead of zero?

Setting a masked score to `-inf` makes its probability zero because $\exp(-\infty)=0$. Multiplying a score by zero is incorrect because $\exp(0)=1$, so the masked position could still receive probability.

## Key takeaway

Self-attention performs three fundamental operations:

1. **Match:** $QK^\top$ compares every query with every available key.
2. **Normalize:** causal masking and row-wise softmax create valid attention weights.
3. **Retrieve:** multiplying the weights by $V$ combines information from context positions.

For batched single-head attention, remember the central shape transformation:

```text
[B, T, d_k] @ [B, d_k, T] → [B, T, T]
[B, T, T] @ [B, T, d_v] → [B, T, d_v]
```
