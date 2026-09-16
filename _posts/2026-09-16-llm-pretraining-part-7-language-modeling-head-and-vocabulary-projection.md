---
title: "LLM Pretraining Part 7: Language Modeling Head and Vocabulary Projection"
date: 2026-09-16 00:00:00 +0000
categories: [LLM, Pretraining]
tags: [language-modeling-head, vocabulary-projection, logits, weight-tying, cross-entropy]
description: A shape-first guide to the final normalization, vocabulary projection, logits, weight tying, loss computation, and training-versus-inference behavior of a decoder-only language model.
math: true
---

Part 6 converted token IDs into hidden vectors and added positional information. After those vectors pass through all Transformer blocks, the language-modeling head converts the final hidden states back into vocabulary scores.

## 1. End-to-end shape flow

Assume:

```text
B       = batch size
T       = sequence length
d_model = hidden dimension
V       = vocabulary size
N       = number of Transformer layers
```

The complete decoder-only path is:

```text
token IDs
[B, T]
    ↓ token embedding
[B, T, d_model]
    ↓ N Transformer blocks
[B, T, d_model]
    ↓ final normalization
[B, T, d_model]
    ↓ vocabulary projection
[B, T, V]
```

The Transformer blocks preserve the last dimension at `d_model`. The language-modeling head changes it to `V`.

## 2. Final normalization

Modern pre-norm decoder models usually apply one final normalization after the last Transformer block:

```python
hidden_states = final_norm(hidden_states)
```

Shape:

```text
before final norm: [B, T, d_model]
after final norm:  [B, T, d_model]
```

The normalization does not change the shape. It stabilizes the scale of the representation before vocabulary projection.

```python
x = token_embedding(input_ids)

for block in transformer_blocks:
    x = block(x)

x = final_norm(x)
logits = lm_head(x)
```

Without the final normalization, the output scale would depend more directly on the accumulated residual stream.

## 3. Vocabulary projection

Let the final normalized hidden states be:

$$
H\in\mathbb{R}^{B\times T\times d_{\text{model}}}
$$

A PyTorch linear layer stores its weight as:

$$
W_{\text{out}}\in\mathbb{R}^{V\times d_{\text{model}}}
$$

The vocabulary logits are:

$$
Z=HW_{\text{out}}^T
$$

Shape flow:

$$
[B,T,d_{\text{model}}][d_{\text{model}},V]=[B,T,V]
$$

Example:

```text
hidden_states:  [2, 512, 1024]
lm_head.weight: [50000, 1024]
weight.T:       [1024, 50000]
logits:         [2, 512, 50000]
```

In PyTorch:

```python
lm_head = nn.Linear(
    in_features=1024,
    out_features=50_000,
    bias=False,
)

logits = lm_head(hidden_states)
```

Although PyTorch stores the weight as `[V, d_model]`, the linear operation internally uses its transpose:

```python
logits = hidden_states @ lm_head.weight.T
```

## 4. What one logit represents

For one batch item $b$, position $t$, and vocabulary token $v$:

$$
z_{b,t,v}=h_{b,t}^Tw_v
$$

where:

```text
h[b, t]: [d_model]
w[v]:    [d_model]
```

Their dot product produces one scalar. Repeating the calculation for all vocabulary entries gives:

```text
logits[b, t]: [V]
```

Every sequence position therefore produces a complete score distribution over the vocabulary.

## 5. Logits and probabilities

Logits are unnormalized real-valued scores. They may be negative or positive, are unbounded, and do not sum to one.

Probabilities are obtained by applying softmax over the vocabulary dimension:

$$
p_{b,t,v}=\frac{\exp(z_{b,t,v})}{\sum_{j=1}^{V}\exp(z_{b,t,j})}
$$

```python
probabilities = torch.softmax(logits, dim=-1)
```

Shapes:

```text
logits:        [B, T, V]
probabilities: [B, T, V]
```

Softmax uses `dim=-1` because the last dimension contains the vocabulary candidates.

## 6. Numerically stable softmax

Directly calculating `exp(logits)` can overflow when a logit is large. Softmax is unchanged if the largest logit is subtracted:

$$
\operatorname{softmax}(z_i)=\frac{\exp(z_i-m)}{\sum_j\exp(z_j-m)},\qquad m=\max_jz_j
$$

```python
shifted = logits - logits.max(dim=-1, keepdim=True).values
probabilities = torch.exp(shifted)
probabilities = probabilities / probabilities.sum(
    dim=-1,
    keepdim=True,
)
```

Production implementations normally use fused, numerically stable softmax or cross-entropy kernels.

## 7. Why training produces logits at every position

During causal-language-model training, every valid position provides a prediction task.

Given:

```text
tokens = [A, B, C, D, E]
```

the model uses:

```text
input:  [A, B, C, D]
target: [B, C, D, E]
```

With a batch represented by one tensor:

```python
input_ids = tokens[:, :-1]
targets = tokens[:, 1:]
```

Shape example:

```text
tokens:    [B, T + 1]
input_ids: [B, T]
targets:   [B, T]
logits:    [B, T, V]
```

Prediction alignment:

| Hidden state | Context represented | Target |
|---|---|---|
| $h_0$ | `A` | `B` |
| $h_1$ | `A B` | `C` |
| $h_2$ | `A B C` | `D` |
| $h_3$ | `A B C D` | `E` |

Causal masking ensures that $h_t$ cannot use its target or any later token.

## 8. Training versus autoregressive inference

### Training

During training, the model predicts the next token at every position in parallel:

```text
hidden_states: [B, T, d_model]
logits:        [B, T, V]
```

All valid positions contribute to the loss.

### Cached inference

During generation, only the new token position needs vocabulary logits:

```text
new hidden state: [B, 1, d_model]
new logits:       [B, 1, V]
```

The length-one dimension is often removed:

```python
last_hidden = hidden_states[:, -1, :]  # [B, d_model]
next_logits = lm_head(last_hidden)     # [B, V]
```

Computing `[B, T, V]` logits for all previous positions during each decoding step would waste computation and memory. With a KV cache, the model calculates the new token's hidden state and new vocabulary logits without recomputing all earlier hidden states.

## 9. Weight tying

The token embedding table is:

$$
E\in\mathbb{R}^{V\times d_{\text{model}}}
$$

The language-modeling head has the same stored weight shape:

$$
W_{\text{out}}\in\mathbb{R}^{V\times d_{\text{model}}}
$$

The model can share them:

```python
self.lm_head.weight = self.token_embedding.weight
```

The same matrix is used in two different operations:

```text
Input:
token IDs → gather rows from E

Output:
hidden states @ E.T → vocabulary logits
```

For vocabulary token $v$:

$$
z_v=h^TE_v
$$

The logit is high when the final hidden state aligns strongly with that token's embedding.

### Parameter savings

Without weight tying:

```text
input embedding: V × d_model
output head:     V × d_model
total:           2 × V × d_model
```

With weight tying:

```text
shared parameters: V × d_model
```

For `V = 50,000` and `d_model = 1,024`, one matrix contains:

$$
50{,}000\times1{,}024=51{,}200{,}000
$$

parameters. Weight tying saves 51.2 million parameters in this example.

## 10. Why the language-modeling head can be expensive

Consider:

```text
B = 2
T = 512
V = 50,000
```

The logits tensor contains:

$$
2\times512\times50{,}000=51{,}200{,}000
$$

elements. In FP16 or BF16, each element uses two bytes:

$$
51{,}200{,}000\times2=102{,}400{,}000\text{ bytes}
$$

This is approximately 102.4 MB or 97.7 MiB for the logits tensor alone.

The vocabulary projection requires approximately:

$$
O(BT\,d_{\text{model}}V)
$$

multiplications. Its memory and compute grow with batch size, sequence length, vocabulary size, and model width.

## 11. Cross-entropy receives logits

PyTorch's `cross_entropy` expects raw logits:

```python
loss = torch.nn.functional.cross_entropy(
    logits.reshape(-1, vocab_size),
    targets.reshape(-1),
)
```

Do not apply softmax first:

```python
# Incorrect
probabilities = torch.softmax(logits, dim=-1)
loss = F.cross_entropy(probabilities, targets)
```

Cross-entropy internally combines a numerically stable `log_softmax` with negative log-likelihood.

Shape transformation:

```text
logits:  [B, T, V] → [B*T, V]
targets: [B, T]    → [B*T]
```

Every valid token position becomes one $V$-class classification example.

## 12. Padding and ignored targets

If sequences have different lengths, padded target positions should not contribute to the loss:

```text
targets:
[
  [21, 35,  9,  2],
  [17,  8, -100, -100]
]
```

```python
loss = F.cross_entropy(
    logits.reshape(-1, V),
    targets.reshape(-1),
    ignore_index=-100,
)
```

The model may still produce logits at padded positions, but those positions are excluded from the training objective.

## 13. Minimal implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class LanguageModelHead(nn.Module):
    def __init__(self, vocab_size, d_model, embedding):
        super().__init__()

        self.final_norm = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(
            d_model,
            vocab_size,
            bias=False,
        )

        # Weight tying
        self.lm_head.weight = embedding.weight

    def forward(self, hidden_states, targets=None):
        # hidden_states: [B, T, d_model]
        hidden_states = self.final_norm(hidden_states)

        # logits: [B, T, V]
        logits = self.lm_head(hidden_states)

        if targets is None:
            return logits

        B, T, V = logits.shape

        loss = F.cross_entropy(
            logits.reshape(B * T, V),
            targets.reshape(B * T),
            ignore_index=-100,
        )

        return logits, loss
```

For generation, project only the final position:

```python
last_hidden = hidden_states[:, -1:, :]  # [B, 1, d_model]
next_logits = lm_head(last_hidden)      # [B, 1, V]
```

## 14. Common interview mistakes

1. Using the wrong language-modeling-head weight shape. PyTorch stores `nn.Linear(d_model, V).weight` as `[V, d_model]`.
2. Applying softmax before cross-entropy.
3. Applying softmax along the sequence dimension instead of the vocabulary dimension.
4. Forgetting that the hidden state at position $t$ predicts token $t+1$.
5. Computing all-position logits during cached decoding.
6. Counting the tied embedding and output head as separate parameter tensors.
7. Multiplying logits memory by `d_model`, even though that dimension has been contracted by the projection.

## 15. Interview check and reviewed answers

Assume:

```text
B       = 4
T       = 256
V       = 32,000
d_model = 768
```

### 1. Final hidden-state shape

```text
[4, 256, 768]
```

Correct.

### 2. Stored shape of `lm_head.weight`

For:

```python
nn.Linear(768, 32000)
```

PyTorch stores:

```text
[out_features, in_features] = [32000, 768]
```

The original answer reversed these dimensions.

### 3. Shape of `lm_head.weight.T`

```text
[768, 32000]
```

The projection is:

```text
[4, 256, 768] @ [768, 32000]
    → [4, 256, 32000]
```

### 4. Training-logits shape

```text
[4, 256, 32000]
```

Correct.

### 5. Softmax dimension

```text
dim=-1
```

For this three-dimensional tensor, `dim=2` is equivalent. Correct.

### 6. Unique parameters with weight tying

$$
32{,}000\times768=24{,}576{,}000
$$

Correct. The shared embedding/head matrix is counted once.

### 7. FP16 memory for the complete logits tensor

The logits contain:

$$
4\times256\times32{,}000=32{,}768{,}000
$$

elements. FP16 uses two bytes per element:

$$
32{,}768{,}000\times2=65{,}536{,}000\text{ bytes}
$$

This is approximately 65.5 MB or 62.5 MiB. The original expression incorrectly included `d_model = 768`; that dimension has already been contracted during projection.

### 8. Cached-generation shapes

```text
input:  [4, 1, 768]
output: [4, 1, 32000]
```

Correct.

### 9. Why cross-entropy receives logits

Cross-entropy internally performs a stable `log_softmax` followed by negative log-likelihood. Passing raw logits is more stable and efficient than applying softmax first. Correct.

### 10. Shifted input and target

Given:

```text
tokens = [10, 20, 30, 40, 50]
```

the training pair is:

```text
input:  [10, 20, 30, 40]
target: [20, 30, 40, 50]
```

Correct.

## 16. Interview-ready summary

After the final Transformer block, a decoder-only language model applies a final normalization and projects `[B, T, d_model]` hidden states into `[B, T, V]` vocabulary logits. PyTorch stores the projection weight as `[V, d_model]` and multiplies hidden states by its transpose. During training, all valid positions produce next-token logits; during cached inference, only the newest position needs projection. Weight tying reuses the `[V, d_model]` input embedding table as the output head, reducing parameters. Cross-entropy receives raw logits and internally performs stable log-softmax and negative log-likelihood.

## Key takeaway

The language-modeling head converts contextual representations back into token scores. The crucial implementation details are the `[V, d_model]` PyTorch weight layout, the `[B, T, V]` logits shape, softmax over the vocabulary axis, correct target shifting, and projecting only the newest position during cached decoding.
