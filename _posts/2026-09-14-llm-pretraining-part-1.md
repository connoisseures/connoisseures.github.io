---
title: "LLM Pretraining Part 1: Next-Token Prediction and Cross-Entropy Loss"
date: 2026-09-14 00:00:00 +0000
categories: [LLM, Pretraining]
tags: [next-token-prediction, cross-entropy, teacher-forcing, causal-mask, pytorch]
description: A practical guide to the causal language-modeling objective, shifted targets, cross-entropy tensor shapes, padding masks, teacher forcing, and the core pretraining loop.
math: true
---
## 1. What does LLM pretraining do?

The central idea of causal language-model pretraining is:

> Given all preceding tokens, predict the next token.

For a token sequence \(w_1, w_2, \ldots, w_T\), the model factorizes the sequence probability as:

$$
P(w_{1:T}) = \prod_{t=1}^{T} P(w_t \mid w_{<t})
$$

At position \(t\), the model learns a probability distribution for the next token:

$$
P(w_{t+1} \mid w_1, \ldots, w_t)
$$

For example, one sequence produces several training examples:

| Model input | Training target |
|---|---|
| `So` | `long` |
| `So long` | `and` |
| `So long and` | `thanks` |
| `So long and thanks` | `for` |
| `So long and thanks for` | `all` |

The model does not memorize one fixed answer. It learns a conditional probability distribution over every token in its vocabulary.

## 2. Self-supervised learning and shifted targets

Pretraining does not require a human to label every example. The next token already present in the document becomes the label, so the text provides its own supervision.

```text
Raw documents
    ↓
Tokenization
    ↓
Token sequences
    ↓
Inputs and one-position-shifted targets
    ↓
Next-token prediction
```

Given:

```python
tokens = [So, long, and, thanks, for, all]
```

we construct:

```python
inputs  = [So,   long, and,    thanks, for]
targets = [long, and,  thanks, for,    all]
```

Each input position is paired with the token immediately following it:

```text
logits produced at “So”     → target “long”
logits produced at “long”   → target “and”
logits produced at “and”    → target “thanks”
logits produced at “thanks” → target “for”
logits produced at “for”    → target “all”
```

## 3. Logits, softmax, and cross-entropy

At every sequence position, the model outputs one score—called a **logit**—for every vocabulary token. If the vocabulary size is \(V\), the output at position \(t\) is:

$$
\mathbf{z}_t \in \mathbb{R}^{V}
$$

Softmax converts these raw scores into a probability distribution:

$$
\hat{y}_t[v] = \frac{e^{z_t[v]}}{\sum_{u=1}^{V} e^{z_t[u]}}
$$

If the correct next token is \(w_{t+1}\), the loss at that position is:

$$
\mathcal{L}_t = -\log P(w_{t+1} \mid w_{\leq t})
$$

The general cross-entropy expression is:

$$
\mathcal{L}_t = -\sum_{v \in V} y_t[v]\log \hat{y}_t[v]
$$

Because the target \(y_t\) is one-hot, only the correct token contributes. Therefore:

$$
\mathcal{L}_t = -\log \hat{y}_t[w_{t+1}]
$$

| Probability assigned to the correct token | Loss |
|---:|---:|
| 0.80 | \(-\log(0.80)=0.223\) |
| 0.20 | \(-\log(0.20)=1.609\) |
| 0.01 | \(-\log(0.01)=4.605\) |

A confident correct prediction produces a small loss. Assigning very little probability to the correct token produces a large loss.

## 4. Understanding the loss code

The core implementation is:

```python
loss = cross_entropy(
    logits.reshape(-1, vocab_size),
    targets.reshape(-1),
)
```

Before reshaping, the tensors normally have these shapes:

```python
logits.shape  == [batch_size, seq_len, vocab_size]  # [B, T, V]
targets.shape == [batch_size, seq_len]              # [B, T]
```

Every `(batch, position)` pair is one classification example:

- The input features are the logits for that position.
- The classes are the \(V\) vocabulary tokens.
- The label is the ID of the correct next token.

PyTorch's `cross_entropy` expects predictions shaped `[number_of_examples, number_of_classes]` and integer targets shaped `[number_of_examples]`. We therefore combine the batch and sequence dimensions:

```python
flat_logits = logits.reshape(-1, vocab_size)  # [B, T, V] → [B*T, V]
flat_targets = targets.reshape(-1)            # [B, T]    → [B*T]
```

Here, `-1` asks PyTorch to infer that dimension from the number of tensor elements.

### Concrete shape example

Suppose:

```python
batch_size = 2
seq_len = 3
vocab_size = 5
```

Then:

```python
logits.shape  == [2, 3, 5]
targets.shape == [2, 3]

flat_logits.shape  == [6, 5]
flat_targets.shape == [6]
```

The six flattened examples are:

| Flattened row | Batch | Position | Prediction | Correct label |
|---:|---:|---:|---|---|
| 0 | 0 | 0 | 5 vocabulary logits | `targets[0, 0]` |
| 1 | 0 | 1 | 5 vocabulary logits | `targets[0, 1]` |
| 2 | 0 | 2 | 5 vocabulary logits | `targets[0, 2]` |
| 3 | 1 | 0 | 5 vocabulary logits | `targets[1, 0]` |
| 4 | 1 | 1 | 5 vocabulary logits | `targets[1, 1]` |
| 5 | 1 | 2 | 5 vocabulary logits | `targets[1, 2]` |

The same reshaping order is applied to both tensors, so their rows remain aligned:

```python
flat_logits[i]  # prediction for example i
flat_targets[i] # correct token ID for example i
```

Reshaping does not change the values or mix the examples. It only presents the tensors in the two-dimensional and one-dimensional forms expected by cross-entropy.

### What `cross_entropy` does internally

Conceptually, the function performs these steps:

```python
log_probs = torch.log_softmax(flat_logits, dim=-1)

token_losses = -log_probs[
    torch.arange(batch_size * seq_len),
    flat_targets,
]

loss = token_losses.mean()
```

For each row, it:

1. Converts the vocabulary logits into log-probabilities.
2. Selects the log-probability at the correct target token ID.
3. Negates it to obtain the token loss.
4. Averages all valid token losses into one scalar.

The sequence-level objective is therefore:

$$
\mathcal{L}
=
-\frac{1}{BT}
\sum_{b=1}^{B}\sum_{t=1}^{T}
\log P(y_{b,t}\mid x_{b,\leq t})
$$

Pass **raw logits** to `cross_entropy`; do not apply softmax first. The function combines `log_softmax` and negative log-likelihood in a numerically stable implementation.

## 5. Ignoring padding and other masked labels

Sequences in a batch may have different lengths, so shorter sequences are padded. Padding tokens are not real prediction targets and should not affect the loss.

PyTorch commonly uses `-100` as the ignored target value:

```python
loss = torch.nn.functional.cross_entropy(
    logits.reshape(-1, vocab_size),
    targets.reshape(-1),
    ignore_index=-100,
)
```

For example:

```python
targets = [long_id, and_id, thanks_id, -100, -100]
```

The last two positions are excluded from both the summed loss and the averaging denominator. In instruction tuning, `-100` is also commonly assigned to prompt tokens when training should optimize only the response tokens.

## 6. Teacher forcing

During pretraining, the model receives the actual preceding tokens rather than its own sampled predictions.

To predict `thanks`, it receives:

```text
So long and
```

Even if the model predicted `but` after `long`, training still uses the correct token `and` at the next input position. This is called **teacher forcing**.

| Pretraining | Generation |
|---|---|
| Uses the correct previous tokens | Uses previously generated tokens |
| Processes all sequence positions in parallel | Usually generates tokens sequentially |
| Compares predictions with known targets | Selects tokens using greedy decoding or sampling |

## 7. Why causal masking is necessary

Training processes all positions simultaneously. Without a causal mask, the representation at one position could inspect future tokens and see the answer it is supposed to predict.

The mask enforces:

$$
\text{position } i \text{ may attend only to positions } j \leq i
$$

For five input tokens, the allowed attention pattern is:

$$
\begin{bmatrix}
1&0&0&0&0\\
1&1&0&0&0\\
1&1&1&0&0\\
1&1&1&1&0\\
1&1&1&1&1
\end{bmatrix}
$$

The current token is visible because it is valid context for predicting the next token. Only later positions must be hidden.

This masking allows a sequence of \(N+1\) tokens to supply \(N\) training targets in one parallel forward pass without target leakage.

## 8. What parameters are learned?

The scalar loss is backpropagated through the complete transformer. Pretraining updates:

- Token embeddings
- Query, key, value, and output projections
- Feedforward-network weights
- Normalization parameters
- Learnable positional parameters, when used
- The final language-modeling head

The model is not directly given grammar rules, semantic categories, or factual labels. It learns representations and patterns that help reduce next-token prediction loss.

## 9. Complete core training loop

```python
for token_batch in training_data:
    # token_batch: [B, T + 1]
    inputs = token_batch[:, :-1]   # [B, T]
    targets = token_batch[:, 1:]   # [B, T]

    logits = model(inputs)         # [B, T, V]

    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, vocab_size),  # [B*T, V]
        targets.reshape(-1),             # [B*T]
        ignore_index=-100,
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

Production implementations add mixed precision, distributed training, gradient accumulation, learning-rate scheduling, gradient clipping, checkpointing, and other optimizations. The fundamental objective remains next-token prediction.

## Key takeaway

LLM pretraining turns an unlabeled token sequence into many supervised next-token examples by shifting the sequence by one position. The transformer predicts every next token in parallel; causal masking hides future answers; cross-entropy scores the probability assigned to each correct token; and backpropagation updates the entire network.

In the loss implementation:

```text
[B, T, V] logits  → [B*T, V] predictions
[B, T] targets    → [B*T] labels
```

Flattening simply allows all token positions in the batch to be evaluated as one large collection of vocabulary-classification examples.
