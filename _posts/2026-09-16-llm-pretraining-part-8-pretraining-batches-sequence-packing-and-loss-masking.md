---
title: "LLM Pretraining Part 8: Pretraining Batches, Sequence Packing, and Loss Masking"
date: 2026-09-16 00:00:00 +0000
categories: [LLM, Pretraining]
tags: [batching, sequence-packing, attention-mask, loss-mask, gradient-accumulation]
description: A shape-first guide to building pretraining batches, padding and packing sequences, isolating document boundaries, masking shifted targets, and normalizing loss over valid tokens.
math: true
---

Parts 1–7 built the complete model path from token IDs through embeddings, Transformer blocks, vocabulary logits, and next-token loss. This part explains how raw documents become efficient and correctly masked training batches.

## 1. From documents to a token stream

Suppose a dataset contains:

```text
Document A: "The cat slept."
Document B: "Transformers use attention."
Document C: "Speech models process audio."
```

After tokenization and adding an end-of-sequence token:

```text
Document A: [14, 81, 39, 2]
Document B: [55, 17, 92, 41, 2]
Document C: [73, 28, 64, 11, 2]
```

Here, token ID `2` represents `<EOS>`.

One simple pretraining pipeline concatenates the documents:

```text
[14, 81, 39, 2, 55, 17, 92, 41, 2, 73, 28, 64, 11, 2]
```

The stream is divided into fixed-length sequences. For sequence length `T = 6`:

```text
Sequence 1: [14, 81, 39, 2, 55, 17]
Sequence 2: [92, 41, 2, 73, 28, 64]
```

This approach uses nearly every token and avoids extensive padding.

## 2. Creating inputs and labels

A convenient implementation first creates chunks of length `T + 1`:

```text
chunk = [14, 81, 39, 2, 55, 17, 92]
```

Then:

```text
input_ids = [14, 81, 39, 2, 55, 17]
labels    = [81, 39, 2, 55, 17, 92]
```

Shapes:

```text
chunk:     [B, T + 1]
input_ids: [B, T]
labels:    [B, T]
```

```python
input_ids = chunks[:, :-1]
labels = chunks[:, 1:]
```

The shift creates one next-token target for every input position.

## 3. Padding batches

Documents and sequences often have different lengths:

```text
Sequence A: [10, 20, 30, 40, 50]
Sequence B: [60, 70, 80]
```

After padding:

```text
input_ids:
[
  [10, 20, 30, 40, 50],
  [60, 70, 80,  0,  0],
]
```

The padding-validity mask is:

```text
attention_mask:
[
  [1, 1, 1, 1, 1],
  [1, 1, 1, 0, 0],
]
```

Shapes:

```text
input_ids:      [B, T]
attention_mask: [B, T]
```

The padding mask has two potential roles:

1. Prevent attention from reading padded key positions.
2. Prevent padded target positions from contributing to the loss.

These roles are related but implemented in different places.

## 4. Attention masking versus loss masking

The attention key mask operates inside attention:

```python
key_mask = attention_mask[:, None, None, :]
```

Shapes:

```text
attention scores: [B, H, T, T]
key mask:         [B, 1, 1, T]
```

The loss mask operates on targets:

```python
labels = labels.masked_fill(
    target_valid == 0,
    -100,
)
```

Then:

```python
loss = F.cross_entropy(
    logits.reshape(-1, V),
    labels.reshape(-1),
    ignore_index=-100,
)
```

A position can be hidden from attention, ignored by the loss, or both. For normal padded language-model training, padding should generally be masked from attention and ignored by the loss.

## 5. Why padding wastes computation

Suppose `T = 1024`, but a sequence contains only 300 valid tokens. Its utilization is:

$$
\frac{300}{1024}\approx29.3\%
$$

The remaining 724 positions still occupy tensor space and may participate in matrix operations even though their losses are ignored.

For a batch:

$$
\text{utilization}=\frac{\text{number of valid tokens}}{B\times T}
$$

Example:

```text
B = 4
T = 1024
valid tokens = 3200
```

Then:

$$
\frac{3200}{4096}=78.125\%
$$

Better batching and sequence packing increase utilization.

## 6. Length bucketing

Length bucketing groups examples with similar lengths.

Instead of batching:

```text
lengths = [100, 1000, 150, 900]
```

form:

```text
short batch: [100, 150]
long batch:  [900, 1000]
```

Advantages:

- Simple to implement.
- Compatible with ordinary `[B, T]` padding masks.
- Reduces wasted computation.

Limitations:

- Does not eliminate padding.
- Requires careful shuffling to avoid correlated batches.
- May produce batches with different valid-token counts.

## 7. Sequence packing

Sequence packing places multiple shorter examples into one fixed-length row.

Suppose:

```text
T = 8

Document A: [11, 12, 13, 2]
Document B: [21, 22, 2]
```

A packed row is:

```text
tokens:      [11, 12, 13, 2, 21, 22, 2, 0]
segment_ids: [ 0,  0,  0, 0,  1,  1, 1,-1]
validity:    [ 1,  1,  1, 1,  1,  1, 1, 0]
```

Packing improves utilization because several documents share one tensor row. However, an ordinary causal mask permits later documents to attend to earlier documents. Whether this is acceptable depends on the training design.

## 8. Two document-boundary strategies

### Strategy 1: Concatenate with EOS

Documents are separated by `<EOS>`, but later documents may attend to earlier ones:

```text
Document A → EOS → Document B
```

The model uses the standard lower-triangular causal mask.

Advantages:

- Simple.
- Maximizes utilization.
- Treats the dataset as a continuous token stream.

Potential issue:

- Tokens in Document B receive context from unrelated Document A.

Many large-scale pipelines accept this behavior.

### Strategy 2: Isolate packed documents

Use a block-diagonal causal mask so each document attends only within itself.

For:

```text
segment_ids = [0, 0, 0, 1, 1]
```

a query at position $i$ may attend to a key at position $j$ only if:

$$
M_{i,j}=(j\le i)\land(s_i=s_j)
$$

The first condition is causal. The second requires the same segment ID.

```text
        keys
        A A A B B
query A ✓ · · · ·
      A ✓ ✓ · · ·
      A ✓ ✓ ✓ · ·
      B · · · ✓ ·
      B · · · ✓ ✓
```

## 9. Constructing a packed causal mask

Assume:

```python
segment_ids = torch.tensor([
    [0, 0, 0, 1, 1]
])  # [B, T]
```

Build the causal relationship:

```python
causal = torch.tril(
    torch.ones(T, T, dtype=torch.bool)
)  # [T, T]
```

Build the same-segment relationship:

```python
same_segment = (
    segment_ids[:, :, None]
    == segment_ids[:, None, :]
)  # [B, T, T]
```

Combine them:

```python
packed_mask = (
    causal[None, :, :]
    & same_segment
)  # [B, T, T]
```

For multi-head attention:

```python
packed_mask = packed_mask[:, None, :, :]
```

Final shapes:

```text
scores:      [B, H, T, T]
packed_mask: [B, 1, T, T]
```

The mask broadcasts across the head dimension.

## 10. Masking EOS transitions in the loss

Consider concatenated documents:

```text
[A1, A2, EOS, B1, B2, EOS]
```

After shifting:

```text
input:  [A1, A2, EOS, B1, B2]
target: [A2, EOS, B1, B2, EOS]
```

The transition `EOS → B1` trains the model to predict the first token of a randomly selected next document.

Some pipelines accept this. Others mask it:

```text
input:  [A1, A2, EOS, B1, B2]
target: [A2, EOS, -100, B2, EOS]
```

This is a separate decision from attention isolation:

- The attention mask determines what context a token may read.
- The loss mask determines which predictions contribute gradients.

Blocking cross-document attention does not automatically mask cross-document targets.

## 11. Position IDs in packed sequences

For:

```text
tokens:      [A1, A2, A3, B1, B2]
segment_ids: [ 0,  0,  0,  1,  1]
```

two common choices are available.

### Continuous positions

```text
position_ids = [0, 1, 2, 3, 4]
```

This matches physical positions in the packed tensor.

### Reset positions per segment

```text
position_ids = [0, 1, 2, 0, 1]
```

This lets each document begin at position zero and removes dependence on where it was packed. If positions are reset, cross-document attention should generally be blocked. Otherwise, tokens with duplicated position IDs may attend across segments and create unintended positional relationships.

The exact choice depends on the model architecture and pretraining convention.

## 12. Why shifted target validity matters

Input validity and target validity do not always align after shifting.

Consider:

```text
tokens:         [10, 20, 30, PAD]
input_ids:      [10, 20, 30]
labels:         [20, 30, PAD]
input validity: [ 1,  1,  1]
target validity:[ 1,  1,  0]
```

Blindly copying the input mask would incorrectly treat the final padded label as valid.

Correct construction:

```python
input_valid = valid[:, :-1]
target_valid = valid[:, 1:]

labels[~target_valid] = -100
```

The reason is next-token alignment, not merely that language-model training is self-supervised.

## 13. Token-weighted loss normalization

Suppose two sequences contain 100 and 10 valid targets. A robust language-model objective averages over valid tokens:

$$
L=\frac{\sum_{b,t}m_{b,t}\ell_{b,t}}{\sum_{b,t}m_{b,t}}
$$

where $m_{b,t}$ is one for a valid target and zero otherwise.

```python
per_token_loss = F.cross_entropy(
    logits.reshape(-1, V),
    labels.reshape(-1),
    ignore_index=-100,
    reduction="none",
).reshape(B, T)

valid = labels != -100

loss = (
    per_token_loss * valid
).sum() / valid.sum()
```

This gives every valid token equal weight. Averaging each sequence first and then averaging sequences would give a short sequence the same total weight as a much longer sequence.

## 14. Gradient accumulation and global batch size

Large models may not fit the desired batch size in GPU memory.

Suppose:

```text
data-parallel workers = D
microbatch size       = B_micro
sequence length       = T
accumulation steps    = A
```

If every position is valid, the global token count per optimizer step is:

$$
B_{\text{tokens}}=D\times B_{\text{micro}}\times T\times A
$$

Example:

```text
D       = 8 GPUs
B_micro = 2 sequences per GPU
T       = 2048
A       = 4 accumulation steps
```

Then:

$$
8\times2\times2048\times4=131{,}072
$$

tokens per optimizer step. With padding, use the actual valid-token count instead of `B × T`.

## 15. Correct loss accumulation

A subtle issue occurs when microbatches have different valid-token counts.

Suppose:

```text
Microbatch 1: 100 valid tokens
Microbatch 2: 20 valid tokens
```

This is generally incorrect:

```python
loss = (mean_loss_1 + mean_loss_2) / 2
```

It weights the microbatches equally rather than weighting their tokens equally.

The correct global token average is:

$$
L=\frac{\text{loss-sum}_1+\text{loss-sum}_2}{100+20}
$$

Conceptually:

```python
total_loss_sum = 0
total_valid_tokens = 0

for microbatch in microbatches:
    loss_sum = compute_token_loss_sum(microbatch)
    valid_tokens = count_valid_targets(microbatch)

    total_loss_sum += loss_sum
    total_valid_tokens += valid_tokens

loss = total_loss_sum / total_valid_tokens
```

In distributed training, aggregate both the loss sum and valid-token count consistently across workers.

## 16. Compact batch-preparation example

```python
import torch


def prepare_batch(token_sequences, pad_id=0, ignore_index=-100):
    """
    token_sequences contains sequences of length <= T + 1.
    Each sequence includes the next-token target.
    """
    batch_size = len(token_sequences)
    max_length = max(len(x) for x in token_sequences)

    tokens = torch.full(
        (batch_size, max_length),
        pad_id,
        dtype=torch.long,
    )

    valid = torch.zeros(
        (batch_size, max_length),
        dtype=torch.bool,
    )

    for b, sequence in enumerate(token_sequences):
        length = len(sequence)
        tokens[b, :length] = torch.tensor(sequence)
        valid[b, :length] = True

    input_ids = tokens[:, :-1]
    labels = tokens[:, 1:].clone()

    input_valid = valid[:, :-1]
    target_valid = valid[:, 1:]

    labels[~target_valid] = ignore_index

    return {
        "input_ids": input_ids,
        "attention_mask": input_valid,
        "labels": labels,
    }
```

Example:

```python
batch = prepare_batch([
    [10, 20, 30, 40, 50],
    [60, 70, 80],
])
```

Shapes:

```text
tokens:         [2, 5]
input_ids:      [2, 4]
attention_mask: [2, 4]
labels:         [2, 4]
```

Values:

```text
input_ids:
[
  [10, 20, 30, 40],
  [60, 70, 80,  0],
]

labels:
[
  [20, 30, 40, 50],
  [70, 80, -100, -100],
]
```

## 17. Common interview mistakes

1. Confusing a `[B, T]` padding mask with a `[T, T]` causal mask.
2. Treating padding masks and loss masks as identical without accounting for shifted labels.
3. Assuming a causal mask prevents cross-document attention in packed sequences.
4. Resetting position IDs without considering segment boundaries.
5. Averaging sequence or microbatch losses instead of valid-token losses.
6. Calculating global batch size only in sequences rather than tokens.
7. Including padding tokens when reporting training throughput.
8. Forgetting that `EOS → next document` remains a training target unless explicitly masked.

## 18. Interview check and reviewed answers

Assume:

```text
B = 4
T = 1024
H = 16
```

### 1. Normal padding-mask shape

```text
[B, T] = [4, 1024]
```

`[1024,1024]` instead resembles a two-dimensional causal mask.

### 2. Broadcast key-mask shape

For scores `[4,16,1024,1024]`:

```python
key_mask = attention_mask[:, None, None, :]
```

```text
key_mask: [4, 1, 1, 1024]
scores:   [4, 16, 1024, 1024]
```

The batch and key dimensions vary; heads and query positions use broadcasting.

### 3. Token utilization

$$
\frac{3200}{4096}=0.78125=78.125\%
$$

Correct.

### 4. Why ordinary causal masking does not isolate documents

An ordinary causal mask checks only whether the key is in the past. It does not compare document or segment IDs. Therefore, a token in a later packed document may attend to earlier documents. Correct.

### 5. Query position 4 versus key position 1

For:

```text
segment_ids = [0, 0, 0, 1, 1]
positions:      0  1  2  3  4
```

the causal condition is true because `1 <= 4`, but the same-segment condition is false because segment `0 != 1`. Under a block-diagonal causal mask, position 4 cannot attend to position 1.

With only an ordinary causal mask, it could. The original answer recognized this conditional distinction but did not answer the specified block-diagonal case.

### 6. Same-segment comparison shape

For `segment_ids` shaped `[4,1024]`:

```python
segment_ids[:, :, None] == segment_ids[:, None, :]
```

has shape:

```text
[4, 1024, 1024]
```

Correct.

### 7. Tokens per optimizer step

Use the stated sequence length 2,048:

$$
8\times2\times2048\times4=131{,}072
$$

The original expression used `2024`, which was a numerical typo.

### 8. Token-weighted mean loss

Microbatch A has 100 tokens with mean loss 2.0. Microbatch B has 20 tokens with mean loss 4.0:

$$
\frac{100(2.0)+20(4.0)}{100+20}
=\frac{280}{120}
\approx2.333
$$

The original expression was correct.

### 9. Why target validity must be shifted

Input positions and next-token targets are offset by one. A final valid input token may have a padded target. Therefore:

```python
input_valid = valid[:, :-1]
target_valid = valid[:, 1:]
```

The shifted `target_valid` mask determines which labels become `-100`. The reason is target alignment, not simply that the objective is self-supervised.

### 10. Two independent document-boundary decisions

A pretraining pipeline must decide independently:

1. **Attention:** May the new document attend to the preceding packed document?
2. **Loss:** Should the `EOS → first token of the next document` prediction contribute to the loss?

Blocking cross-document attention does not automatically remove the cross-document target from the loss.

## 19. Interview-ready summary

Pretraining pipelines transform tokenized documents into fixed-shape batches by padding, bucketing, concatenating, or packing sequences. A padding mask begins as `[B,T]` and becomes `[B,1,1,T]` when hiding padded keys. Packed documents require an additional same-segment condition if cross-document attention should be blocked, producing a mask shaped `[B,1,T,T]`. Because language-model labels are shifted, loss validity must be derived from the shifted target mask. Losses should be normalized over valid tokens, including across gradient-accumulation steps and distributed workers.

## Key takeaway

Efficient pretraining is not only about filling tensors. The pipeline must preserve the semantics of attention, position IDs, document boundaries, and next-token targets. Correct masking and token-weighted normalization determine whether packing improves efficiency without silently changing the training objective.
