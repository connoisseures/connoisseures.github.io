---
title: "LLM Coding Interview: Debugging Causal Self-Attention"
date: 2026-09-12 00:00:00 +0000
categories: [Interview Preparation, LLM]
tags: [attention, causal-mask, padding-mask, numpy, debugging]
description: A step-by-step debugging exercise covering attention scaling, causal masking, stable softmax, broadcasting, and padded query outputs.
---

## Interview question

The following NumPy function is intended to compute single-head causal self-attention:

```python
import numpy as np


def causal_attention(Q, K, V):
    """
    Q, K, V: shape (seq_len, head_dim)
    Returns: shape (seq_len, head_dim)
    """
    scores = Q @ K.T
    scores = scores / Q.shape[0]

    seq_len = Q.shape[0]
    causal_mask = np.triu(np.ones((seq_len, seq_len)), k=1)
    scores = scores * causal_mask

    probabilities = np.exp(scores)
    probabilities = probabilities / probabilities.sum(axis=0, keepdims=True)

    output = probabilities @ V
    return output
```

The function runs without an exception, but its output is incorrect.

1. Identify every bug.
2. Explain why each bug matters.
3. Write a corrected implementation using numerically stable softmax.

## Bug 1: Scaling by sequence length

The original implementation divides the attention scores by `Q.shape[0]`, which is the sequence length. Scaled dot-product attention divides by the square root of the key/query head dimension:

```python
scores = scores / np.sqrt(Q.shape[1])
```

For query and key vectors of dimension \(d_k\), their dot product sums \(d_k\) terms. Its variance therefore grows with \(d_k\). Dividing by \(\sqrt{d_k}\) keeps the logits at a reasonable scale and prevents softmax from becoming excessively sharp.

## Bug 2: Multiplying by the causal mask

In the original code, future positions contain `1` and allowed positions contain `0`:

```python
causal_mask = np.triu(np.ones((seq_len, seq_len)), k=1)
scores = scores * causal_mask
```

This keeps future scores and replaces allowed scores with zero—the reverse of the desired behavior. Moreover, setting a masked score to zero does not block it because `exp(0) = 1`, so it can still receive positive softmax probability.

Masked logits should be replaced with negative infinity before softmax:

```python
future_mask = np.triu(
    np.ones((seq_len, seq_len), dtype=bool),
    k=1,
)
scores = np.where(future_mask, -np.inf, scores)
```

The diagonal remains available, so token \(i\) can attend to itself and all earlier tokens, but not to tokens after \(i\).

## Bug 3: Numerically unstable softmax

Computing `np.exp(scores)` directly can overflow when a logit is large. Subtracting the maximum value does not change the resulting softmax distribution and keeps the exponentials in a safe range:

```python
shifted_scores = scores - np.max(scores, axis=-1, keepdims=True)
exp_scores = np.exp(shifted_scores)
```

The `keepdims=True` argument is important. Without it, a row maximum array with shape `(L,)` is aligned with the last axis of the `(L, L)` score matrix, causing subtraction by column rather than explicitly preserving one maximum per query row.

## Bug 4: Normalizing along the wrong axis

Each query must produce a probability distribution over all key positions. Keys occupy the last dimension of the score matrix, so softmax must normalize along `axis=-1` (equivalent to `axis=1` for the two-dimensional example):

```python
probabilities = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
```

The original `axis=0` normalizes across query positions instead.

## Corrected single-head implementation

```python
import numpy as np


def causal_attention(Q, K, V):
    """
    Q, K, V: (L, D)
    Returns: (L, D)
    """
    seq_len, head_dim = Q.shape

    scores = (Q @ K.T) / np.sqrt(head_dim)

    future_mask = np.triu(
        np.ones((seq_len, seq_len), dtype=bool),
        k=1,
    )
    scores = np.where(future_mask, -np.inf, scores)

    shifted_scores = scores - np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(shifted_scores)
    probabilities = exp_scores / exp_scores.sum(axis=-1, keepdims=True)

    return probabilities @ V
```

## Follow-up: Batched multi-head attention with padding

Suppose the tensors have these shapes:

```text
Q, K, V:      (B, H, L, D)
padding_mask: (B, L)
```

`padding_mask[b, i]` is `True` when position `i` contains a real token and `False` when it is padding.

### Mask padded keys

The final score dimension represents key positions. Reshape the padding mask to `(B, 1, 1, L)` so NumPy broadcasts it across heads and query positions:

```python
key_mask = padding_mask[:, None, None, :]
```

This prevents real queries from attending to padded keys.

### Zero padded query outputs

Masking keys does not automatically zero the output produced for a padded query. The query position is the second-to-last output dimension, so reshape the same padding mask to `(B, 1, L, 1)` and apply it after attention:

```python
query_mask = padding_mask[:, None, :, None]
output = output * query_mask
```

Applying this after softmax avoids intentionally turning padded query rows into rows containing only `-inf`, which would make an ordinary softmax produce `NaN`.

## Robust batched implementation

The following implementation safely handles fully masked rows, including those that can occur with left padding:

```python
import numpy as np


def masked_causal_attention(Q, K, V, padding_mask):
    """
    Q, K, V:      (B, H, L, D)
    padding_mask: (B, L), True for real tokens

    Returns:
        output: (B, H, L, D)
    """
    _, _, seq_len, head_dim = Q.shape

    # (B, H, L, L)
    scores = Q @ K.transpose(0, 1, 3, 2)
    scores = scores / np.sqrt(head_dim)

    # (1, 1, L, L): True for the current and earlier positions.
    causal_mask = np.tril(
        np.ones((seq_len, seq_len), dtype=bool)
    )[None, None, :, :]

    # (B, 1, 1, L): True for valid key positions.
    key_mask = padding_mask[:, None, None, :].astype(bool)
    allowed_mask = causal_mask & key_mask

    masked_scores = np.where(allowed_mask, scores, -np.inf)

    # Stable masked softmax. Replace a non-finite maximum for a fully
    # masked row before exponentiation, then explicitly produce zero.
    row_max = np.max(masked_scores, axis=-1, keepdims=True)
    row_max = np.where(np.isfinite(row_max), row_max, 0.0)

    exp_scores = np.where(
        allowed_mask,
        np.exp(masked_scores - row_max),
        0.0,
    )
    denominator = exp_scores.sum(axis=-1, keepdims=True)

    probabilities = np.divide(
        exp_scores,
        denominator,
        out=np.zeros_like(exp_scores),
        where=denominator > 0,
    )

    # (B, H, L, D)
    output = probabilities @ V

    # (B, 1, L, 1): zero all padded query outputs.
    query_mask = padding_mask[:, None, :, None].astype(output.dtype)
    output = output * query_mask

    return output
```

## Interview takeaways

- Scale attention logits by \(\sqrt{d_k}\), not by sequence length.
- Use negative infinity for disallowed logits; multiplying scores by zero is not a valid softmax mask.
- Normalize each query over key positions—the last score dimension.
- Keep the reduced dimension when implementing stable softmax.
- A key padding mask has shape `(B, 1, 1, L)`.
- A query padding mask has shape `(B, 1, L, 1)`.
- Fully masked rows require a safe masked-softmax implementation to avoid `NaN` values.
