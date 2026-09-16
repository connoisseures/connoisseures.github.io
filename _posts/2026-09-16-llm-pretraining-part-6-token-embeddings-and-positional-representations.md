---
title: "LLM Pretraining Part 6: Token Embeddings and Positional Representations"
date: 2026-09-16 00:00:00 +0000
categories: [LLM, Pretraining]
tags: [token-embeddings, positional-encoding, rope, alibi, weight-tying]
description: A shape-first guide to token embeddings, learned and sinusoidal positions, RoPE, ALiBi, KV-cache positions, and weight tying in decoder-only language models.
math: true
---

Parts 1–5 followed tensors through next-token prediction, attention, multi-head attention, transformer blocks, and efficient matrix implementation. This part explains how discrete token IDs become continuous vectors and how a Transformer represents token order.

## 1. Token embeddings

Suppose the tokenizer produces integer token IDs:

$$
\text{input\_ids}\in\mathbb{N}^{B\times T}
$$

For example:

```text
input_ids.shape = [2, 4]

[
  [15, 81, 29, 7],
  [42, 13,  9, 0]
]
```

The model stores an embedding table:

$$
E\in\mathbb{R}^{V\times d_{\text{model}}}
$$

where:

- $V$ is the vocabulary size.
- $d_{\text{model}}$ is the hidden dimension.

Embedding lookup selects one row for every token ID:

$$
X_{b,t}=E[\text{input\_ids}_{b,t}]
$$

The shape transformation is:

```text
input_ids:       [B, T]
embedding_table: [V, d_model]
token_vectors:   [B, T, d_model]
```

For example:

```text
input_ids:       [2, 4]
embedding_table: [50_000, 1_024]
token_vectors:   [2, 4, 1_024]
```

In PyTorch:

```python
import torch
import torch.nn as nn

embedding = nn.Embedding(
    num_embeddings=50_000,
    embedding_dim=1_024,
)

input_ids = torch.tensor([
    [15, 81, 29, 7],
    [42, 13, 9, 0],
])

x = embedding(input_ids)

print(x.shape)  # [2, 4, 1024]
```

The forward operation is a row lookup. During backpropagation, the selected rows receive gradients.

## 2. Embedding lookup as matrix multiplication

Conceptually, token ID $i$ can be represented by a one-hot vector:

$$
o_i\in\mathbb{R}^{V}
$$

Its embedding is:

$$
x_i=o_iE
$$

Shape flow:

$$
[1,V][V,d_{\text{model}}]=[1,d_{\text{model}}]
$$

Implementations do not normally construct the large one-hot tensor. They directly gather the corresponding rows from $E$.

## 3. Why embeddings alone are insufficient

Consider:

```text
The dog chased the cat
The cat chased the dog
```

These sentences contain the same set of tokens but have different meanings. Token embeddings identify the tokens, but they do not inherently identify where each token occurs.

Self-attention without positional information is permutation-equivariant: if the input tokens are reordered, the outputs are reordered in the corresponding way. The model therefore needs an additional representation of order.

## 4. Two families of positional representation

Position usually enters a Transformer in one of two places:

| Method family | Where position enters | Examples |
|---|---|---|
| Additive representation | Added to hidden states before Transformer layers | Learned absolute, sinusoidal |
| Attention-based representation | Changes Q/K geometry or attention logits | RoPE, relative bias, ALiBi |

For an additive method:

$$
X^{(0)}=E_{\text{token}}+E_{\text{position}}
$$

The shapes can be:

```text
token_embeddings:    [B, T, d_model]
position_embeddings: [1, T, d_model]
result:              [B, T, d_model]
```

The position tensor broadcasts across the batch dimension.

Methods such as RoPE do not add a position vector to the residual stream. Instead, they transform $Q$ and $K$ so that the attention dot product contains relative-position information.

## 5. Learned absolute positional embeddings

A learned position table has shape:

$$
P\in\mathbb{R}^{T_{\max}\times d_{\text{model}}}
$$

For a sequence of length $T$:

```python
positions = torch.arange(T, device=input_ids.device)

x = token_embedding(input_ids)       # [B, T, d_model]
p = position_embedding(positions)    # [T, d_model]
x = x + p                            # [B, T, d_model]
```

The `[T, d_model]` position tensor broadcasts over the batch dimension. It can also be represented explicitly as `[1, T, d_model]`.

Advantages:

- Simple to implement.
- Position representations are learned from data.
- Effective within the trained context length.

Limitations:

- The table has a fixed maximum length.
- Positions outside the table have no learned rows.
- Extrapolation beyond the training context is generally weak.

## 6. Sinusoidal positional encoding

The original Transformer used fixed sine and cosine functions:

$$
PE(pos,2i)=\sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

$$
PE(pos,2i+1)=\cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

Each dimension pair uses a different frequency. The resulting table has shape:

```text
[T, d_model]
```

It is added to the token embeddings:

$$
X=E_{\text{token}}+PE
$$

Sinusoidal encodings have no learned parameters and can be calculated for unseen positions. However, being mathematically computable at a longer position does not guarantee that a model trained only on shorter sequences will perform well there.

## 7. Rotary positional embeddings

Rotary positional embeddings, or RoPE, rotate each query and key according to its position. RoPE is normally applied after Q/K projection and head splitting:

```text
Q: [B, H, T, d_head]
K: [B, H, T, d_head]
```

RoPE is not normally applied to $V$.

### Rotation of dimension pairs

Adjacent feature dimensions are treated as two-dimensional coordinate pairs:

```text
(x_0, x_1), (x_2, x_3), ...
```

For position $m$, one pair is rotated as:

$$
\begin{bmatrix}
x'_{2i}\\
x'_{2i+1}
\end{bmatrix}
=
\begin{bmatrix}
\cos(m\theta_i) & -\sin(m\theta_i)\\
\sin(m\theta_i) & \cos(m\theta_i)
\end{bmatrix}
\begin{bmatrix}
x_{2i}\\
x_{2i+1}
\end{bmatrix}
$$

The shape does not change:

```text
[B, H, T, d_head] → [B, H, T, d_head]
```

Because dimensions are rotated in pairs, the standard formulation requires an even `d_head`.

### Why RoPE expresses relative position

Let $R_m$ rotate a query at position $m$, and let $R_n$ rotate a key at position $n$:

$$
q'_m=R_mq_m,\qquad k'_n=R_nk_n
$$

Their dot product is:

$$
(q'_m)^Tk'_n=q_m^TR_m^TR_nk_n
$$

Since:

$$
R_m^TR_n=R_{n-m}
$$

the Q–K interaction depends on the relative displacement $n-m$. RoPE uses absolute positions to construct rotations, while the attention interaction naturally exposes relative position.

## 8. Shape-aware RoPE implementation

```python
def apply_rope(x, cos, sin):
    """
    x:   [B, H, T, d_head]
    cos: [1, 1, T, d_head // 2]
    sin: [1, 1, T, d_head // 2]
    """
    x_even = x[..., 0::2]  # [B, H, T, d_head // 2]
    x_odd = x[..., 1::2]   # [B, H, T, d_head // 2]

    rotated_even = x_even * cos - x_odd * sin
    rotated_odd = x_even * sin + x_odd * cos

    return torch.stack(
        [rotated_even, rotated_odd],
        dim=-1,
    ).flatten(-2)
```

Usage:

```python
q = apply_rope(q, cos, sin)
k = apply_rope(k, cos, sin)

scores = q @ k.transpose(-2, -1)
scores = scores / math.sqrt(d_head)
```

Shape flow:

```text
q:            [B, H, T, d_head]
k:            [B, H, T, d_head]
k.transpose:  [B, H, d_head, T]
scores:       [B, H, T, T]
```

RoPE changes the values inside $Q$ and $K$, not their shapes.

## 9. RoPE and the KV cache

During autoregressive decoding, cached keys have already been rotated using their original positions.

Suppose a cache contains tokens at positions `0` through `T - 1`. The next token is at zero-based position `T`:

```text
new Q before RoPE: [B, H, 1, d_head]
new K before RoPE: [B, H, 1, d_head]
```

Rotate both with position `T`:

```text
rotated new Q: [B, H, 1, d_head]
rotated new K: [B, H, 1, d_head]
```

Then append the new key:

```text
cached K before: [B, H, T,     d_head]
cached K after:  [B, H, T + 1, d_head]
```

A common bug is applying position zero to every newly decoded token. The new position must be derived from the cache length or supplied through correct `position_ids`.

## 10. ALiBi and relative attention biases

ALiBi adds a distance-dependent bias directly to attention scores:

$$
S=\frac{QK^T}{\sqrt{d_{\text{head}}}}
$$

Example shapes:

```text
scores: [B, H, T, T]
bias:   [1, H, T, T]
```

Then:

$$
S'=S+\text{bias}
$$

Each head typically uses a different slope, and distant keys receive increasingly negative biases. Unlike learned absolute positions, ALiBi does not add vectors to hidden states or rotate Q/K. It directly changes the attention logits according to relative distance.

## 11. Weight tying

The input embedding table can also be reused as the output vocabulary projection.

Input embedding table:

$$
E\in\mathbb{R}^{V\times d_{\text{model}}}
$$

Final hidden states:

$$
H\in\mathbb{R}^{B\times T\times d_{\text{model}}}
$$

Vocabulary logits:

$$
\text{logits}=HE^T
$$

Shape flow:

```text
hidden_states: [B, T, d_model]
E.T:           [d_model, V]
logits:        [B, T, V]
```

In PyTorch:

```python
self.token_embedding = nn.Embedding(vocab_size, d_model)
self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

self.lm_head.weight = self.token_embedding.weight
```

The weights are shared, but the operations are different:

- Input: gather rows from $E$.
- Output: multiply hidden states by $E^T$.

Weight tying reduces the parameter count and creates a shared geometry between representing input tokens and predicting output tokens.

## 12. Positional-method comparison

| Method | Where applied | Learned parameters | Relative-position behavior | Typical limitation |
|---|---|---:|---|---|
| Learned absolute | Added to embeddings | Yes | Indirect | Fixed learned position table |
| Sinusoidal | Added to embeddings | No | Can be derived from frequencies | Long-context generalization is not guaranteed |
| RoPE | Rotates Q and K | Usually no positional table | Natural in Q–K dot products | Requires careful long-context scaling and cache positions |
| ALiBi | Added to attention logits | Usually fixed slopes | Direct distance bias | Less expressive than a learned positional mechanism |

## 13. Interview check and reviewed answers

Assume:

```text
B = 2
T = 512
V = 50,000
d_model = 1,024
H = 16
d_head = 64
```

### 1. Token embedding-table shape

```text
[50000, 1024]
```

Correct. Each of the 50,000 vocabulary entries has one 1,024-dimensional embedding.

### 2. Output shape after embedding `[2, 512]` token IDs

```text
[2, 512, 1024]
```

Correct. Embedding lookup appends the model dimension without changing batch or sequence length.

### 3. Learned position-table shape for a 4,096-token context

```text
[4096, 1024]
```

Correct.

### 4. Position-tensor shape that broadcasts over the batch

```text
[1, 512, 1024]
```

Correct. `[512, 1024]` also broadcasts correctly against `[2, 512, 1024]`.

### 5. Query shape after splitting into heads

```text
[2, 16, 512, 64]
```

Correct.

### 6. Does RoPE change the shape of Q?

```text
No
```

Correct. It rotates pairs of values while preserving all tensor dimensions.

### 7. Why apply RoPE to Q and K but not normally to V?

Q and K determine attention relationships through their dot products. V contains the information aggregated using the resulting attention weights. Position therefore needs to affect the Q–K interaction; standard RoPE does not need to rotate V.

### 8. Weight-tied output projection

The embedding table is:

```text
E:   [50000, 1024]
E.T: [1024, 50000]
```

The matrix operation is:

```text
[2, 512, 1024] @ [1024, 50000]
    → [2, 512, 50000]
```

```python
logits = hidden_states @ embedding.weight.T
```

The original answer identified the correct shape of $E^T$ but needed to include the complete multiplication.

### 9. Position of the next token with 300 cached tokens

If the cached tokens occupy zero-based positions `0` through `299`, the next token is the 301st token but has position ID:

```text
300
```

Using `301` would introduce an off-by-one error.

### 10. Correct description of RoPE

```text
B. RoPE rotates query and key dimension pairs based on position.
```

Correct. RoPE does not replace causal masking and does not increase `d_head`.

## 14. Interview-ready summary

Token embeddings map integer token IDs from `[B, T]` to continuous vectors `[B, T, d_model]` by gathering rows from a `[V, d_model]` table. Because self-attention alone does not encode order, positional information must be introduced. Learned absolute and sinusoidal representations are added to token embeddings, whereas RoPE rotates queries and keys so that their dot product depends on relative position. RoPE preserves Q/K shapes and works naturally with KV caching as long as every cached token is rotated using its correct absolute position. With weight tying, the input embedding matrix is reused in transposed form to project final hidden states to `[B, T, V]` vocabulary logits.

## Key takeaway

Token embeddings represent **what** each token is. Positional representations provide information about **where** it occurs. Additive methods modify the residual-stream input, while RoPE and ALiBi change the attention calculation. Correct shape handling, cache position IDs, and the orientation of the tied embedding matrix are the main implementation details to verify.
