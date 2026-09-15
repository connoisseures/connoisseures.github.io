---
title: "LLM Pretraining Part 5: Efficient Matrix Implementation"
date: 2026-09-15 00:00:00 +0000
categories: [LLM, Pretraining]
tags: [fused-qkv, tensor-shapes, broadcasting, swiglu, flashattention]
description: A practical guide to vectorized transformer operations, fused QKV and SwiGLU projections, mask broadcasting, reshape versus transpose, and efficient attention.
math: true
---
## 1. Vectorizing transformer operations

Transformer equations are often written one token or one attention head at a time, but production implementations avoid Python loops over batches, tokens, and heads. Instead, they use large batched matrix operations that run efficiently on GPUs.

A slow conceptual implementation is:

```python
for b in range(B):
    for t in range(T):
        Q[b, t] = X[b, t] @ Wq
```

The vectorized equivalent is:

```python
Q = X @ Wq
```

Shape flow:

```text
X:  [B, T, d_model]
Wq: [d_model, d_model]
Q:  [B, T, d_model]
```

The same weight matrix is applied to every token, while optimized matrix-multiplication kernels process the work together.

## 2. Fused QKV projection

A direct attention implementation uses three projections:

```python
Q = X @ Wq
K = X @ Wk
V = X @ Wv
```

The weights can instead be concatenated along the output dimension:

$$
W_{\text{QKV}}=[W_Q\;W_K\;W_V]
$$

If each individual matrix is `[d_model, d_model]`, the fused matrix is:

```text
W_qkv: [d_model, 3*d_model]
```

All projections are then computed with one matrix multiplication:

```python
qkv = X @ W_qkv                       # [B, T, 3*d_model]
Q, K, V = np.split(qkv, 3, axis=-1)  # each [B, T, d_model]
```

Fusion does not substantially reduce the arithmetic count. Its main benefits are fewer kernel launches and fewer reads of `X` from memory.

## 3. Reshaping fused QKV into heads

Assume:

```text
d_model = H * d_head
qkv: [B, T, 3*d_model]
```

Expose the QKV, head, and head-feature dimensions:

```python
qkv = qkv.reshape(B, T, 3, H, d_head)
```

Shape:

```text
[B, T, 3, H, d_head]
```

Move the QKV and head axes forward:

```python
qkv = qkv.transpose(2, 0, 3, 1, 4)
```

Shape:

```text
[3, B, H, T, d_head]
```

Finally:

```python
Q, K, V = qkv
```

Each tensor has shape:

```text
[B, H, T, d_head]
```

## 4. Reshape versus transpose

These operations have different meanings.

`reshape` changes how dimensions are grouped:

```python
Q = Q.reshape(B, T, H, d_head)
```

It divides `d_model` into `H * d_head` while following the existing element order.

`transpose` changes the logical order of axes:

```python
Q = Q.transpose(0, 2, 1, 3)
```

It performs:

```text
[B, T, H, d_head] → [B, H, T, d_head]
```

Replacing that transpose with:

```python
Q.reshape(B, H, T, d_head)
```

is generally wrong. It may return the expected shape while assigning elements to incorrect token-head pairs.

For example:

```python
x = np.array([
    [1, 2, 3],
    [4, 5, 6],
])
```

Transpose:

```text
[[1, 4],
 [2, 5],
 [3, 6]]
```

Reshape to the same shape:

```text
[[1, 2],
 [3, 4],
 [5, 6]]
```

The shapes match, but their meanings and values do not.

## 5. Views, strides, and contiguity

A transpose often creates a view with a different stride pattern instead of copying data. After a transpose, a tensor may no longer be contiguous.

In PyTorch:

```python
combined = heads.transpose(1, 2)
combined = combined.contiguous().view(B, T, d_model)
```

Alternatively:

```python
combined = heads.transpose(1, 2).reshape(B, T, d_model)
```

Important distinctions:

- `transpose` permutes logical axes and usually changes strides.
- `view` requires a compatible memory layout.
- `reshape` returns a view when possible and may allocate a copy otherwise.
- It is not generally correct to say that reshape never changes strides.

## 6. Batched score calculation

With:

```text
Q: [B, H, T, d_head]
K: [B, H, T, d_head]
```

calculate:

```python
scores = Q @ K.transpose(0, 1, 3, 2)
```

Shape flow:

```text
Q:  [B, H, T, d_head]
Kᵀ: [B, H, d_head, T]
scores: [B, H, T, T]
```

The batch and head dimensions are handled as batch dimensions by matrix multiplication.

In PyTorch, negative indices make the intended matrix transpose clearer:

```python
scores = Q @ K.transpose(-2, -1)
```

## 7. Equivalent `einsum` expressions

Attention scores can also be written as:

```python
scores = np.einsum("bhtd,bhsd->bhts", Q, K)
```

where:

```text
b: batch
h: head
t: query position
s: key position
d: head feature
```

The repeated feature dimension `d` is summed over:

$$
S_{b,h,t,s}=\sum_d Q_{b,h,t,d}K_{b,h,s,d}
$$

Value retrieval can be written as:

```python
heads = np.einsum("bhts,bhsd->bhtd", weights, V)
```

which sums over key positions `s`.

## 8. Broadcasting the causal mask

Attention scores have shape:

```text
[B, H, T, T]
```

A causal mask can have shape `[T, T]`; NumPy aligns it with the last two dimensions and broadcasts it over batch and heads. An explicit representation is:

```python
causal_mask = causal_mask[None, None, :, :]
```

```text
scores:      [B, H, T, T]
causal_mask: [1, 1, T, T]
```

Dimensions of size one are broadcast without explicitly repeating the mask.

## 9. Padding key mask

A padding mask commonly begins with:

```text
padding_mask: [B, T]
```

To hide padded **key positions** while broadcasting across heads and queries:

```python
key_mask = padding_mask[:, None, None, :]
```

Shape:

```text
[B, 1, 1, T]
```

Broadcasting:

```text
scores:   [B, H, T, T]
key_mask: [B, 1, 1, T]
result:   [B, H, T, T]
```

The causal and padding masks serve different purposes:

| Mask | Shape | Purpose |
|---|---|---|
| Causal mask | `[1, 1, T, T]` | Hide future key positions |
| Padding key mask | `[B, 1, 1, T]` | Hide padded keys in each sequence |

## 10. Padding query mask

Key masking prevents valid queries from reading padding, but it does not automatically remove outputs at padded query positions.

To zero padded per-head query outputs:

```python
query_mask = padding_mask[:, None, :, None]
heads = heads * query_mask
```

Shapes:

```text
heads:      [B, H, T, d_head]
query_mask: [B, 1, T, 1]
```

The mask broadcasts over heads and feature dimensions.

After concatenation, an equivalent mask is:

```python
output = output * padding_mask[:, :, None]
```

Key masking controls what information can be read; query masking controls whether an output is retained.

## 11. Fused SwiGLU projection

SwiGLU uses two input projections:

```python
gate = X @ W_gate
value = X @ W_up
```

They can be fused:

$$
W_{\text{gate-up}}=[W_{\text{gate}}\;W_{\text{up}}]
$$

Shape:

```text
W_gate_up: [d_model, 2*d_ff]
```

Implementation:

```python
gate_up = X @ W_gate_up
gate_pre, value = np.split(gate_up, 2, axis=-1)

hidden = silu(gate_pre) * value
output = hidden @ W_down
```

Shape flow:

```text
X:        [B, T, d_model]
gate_up:  [B, T, 2*d_ff]
gate_pre: [B, T, d_ff]
value:    [B, T, d_ff]
hidden:   [B, T, d_ff]
output:   [B, T, d_model]
```

## 12. Complete vectorized transformer block

```python
def transformer_block(
    X,
    W_qkv,
    W_o,
    W_gate_up,
    W_down,
    norm1_scale,
    norm2_scale,
    num_heads,
    padding_mask=None,
):
    B, T, d_model = X.shape
    d_head = d_model // num_heads

    # Attention sublayer
    attn_input = rms_norm(X, norm1_scale)

    qkv = attn_input @ W_qkv
    qkv = qkv.reshape(B, T, 3, num_heads, d_head)
    qkv = qkv.transpose(2, 0, 3, 1, 4)
    Q, K, V = qkv

    scores = Q @ K.transpose(0, 1, 3, 2)
    scores = scores / np.sqrt(d_head)

    future = np.triu(np.ones((T, T), dtype=bool), k=1)
    scores = np.where(future, -np.inf, scores)

    if padding_mask is not None:
        key_mask = padding_mask[:, None, None, :]
        scores = np.where(key_mask, scores, -np.inf)

    weights = softmax(scores, axis=-1)
    heads = weights @ V

    if padding_mask is not None:
        query_mask = padding_mask[:, None, :, None]
        heads = heads * query_mask

    attn_output = heads.transpose(0, 2, 1, 3)
    attn_output = attn_output.reshape(B, T, d_model)
    attn_output = attn_output @ W_o

    X = X + attn_output

    # SwiGLU sublayer
    ffn_input = rms_norm(X, norm2_scale)

    gate_up = ffn_input @ W_gate_up
    gate_pre, value = np.split(gate_up, 2, axis=-1)

    hidden = silu(gate_pre) * value
    ffn_output = hidden @ W_down

    return X + ffn_output
```

## 13. FlashAttention perspective

The implementation above materializes the complete score and probability tensors:

```text
scores:  [B, H, T, T]
weights: [B, H, T, T]
```

For long sequences, these tensors consume substantial memory. FlashAttention computes attention in tiles and avoids storing the entire matrices in high-bandwidth memory.

The mathematical result remains the same. Its main advantages come from:

- Tiling the computation
- Reducing GPU memory reads and writes
- Fusing operations
- Maintaining online softmax statistics

FlashAttention does not remove standard attention's exact quadratic arithmetic complexity, but it substantially improves memory efficiency and practical runtime.

## 14. Interview check and reviewed answers

Assume:

```text
X:       [2, 512, 1024]
H:       16
d_head:  64
d_ff:    2816
```

### 1. Shape of `W_qkv`

```text
[1024, 3*1024] = [1024, 3072]
```

### 2. Shape of `qkv = X @ W_qkv`

```text
[2, 512, 3*1024] = [2, 512, 3072]
```

### 3. Shape after exposing QKV and heads

```text
[2, 512, 3, 16, 64]
```

### 4. Shapes after transpose and unpacking

After:

```text
[2, 512, 3, 16, 64]
→ [3, 2, 16, 512, 64]
```

each of `Q`, `K`, and `V` has shape:

```text
[2, 16, 512, 64]
```

### 5. Padding key-mask shape

Correct shape:

```text
[2, 1, 1, 512]
```

`[1, 1, 512, 512]` would instead resemble an explicit causal mask. The padding key mask must vary by batch and key position while broadcasting over heads and query positions.

### 6. Padding query-mask shape

```text
[2, 1, 512, 1]
```

It broadcasts over 16 heads and 64 features.

### 7. Fused `W_gate_up` shape

Each individual projection is `[1024, 2816]`, but the fused matrix contains both:

```text
W_gate_up: [1024, 2*2816]
           [1024, 5632]
```

The output `[2, 512, 5632]` is split into two tensors of `[2, 512, 2816]`.

### 8. Why transpose cannot be replaced by reshape

Reshape regroups elements according to their existing linear order; transpose explicitly swaps logical axes. Replacing transpose with reshape can produce the expected shape while silently assigning values to the wrong token-head pairs. Reshape may also change strides or create a copy, so “reshape does not change stride” is not generally correct.

## Key takeaway

Efficient transformer code expresses token, head, and batch operations as large batched matrix multiplications. Fused projections improve hardware utilization, broadcasting avoids unnecessary mask copies, and correct use of reshape and transpose preserves semantic tensor dimensions. Specialized kernels such as FlashAttention further improve performance by reducing memory traffic without changing the attention result.
