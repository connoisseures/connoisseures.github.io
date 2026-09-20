---
layout: post
title: "LLM Coding & Debugging Interview — Question 2: Debugging Causal-LM Training Loss"
date: 2026-09-19
categories: [LLM, Interview Preparation, debugging]
tags: [causal-lm, training, debugging, mixed-precision, gradient-accumulation]
---

# LLM Coding & Debugging Interview — Question 2
# Debugging Causal-LM Training Loss

**Date:** 2026-09-19  
**Topic:** LLM training, causal language modeling, numerical stability, gradient debugging, mixed precision  
**Difficulty:** Intermediate → Advanced

---

## 1. Interview Prompt

We are given a causal language-model loss implementation:

```python
def causal_lm_loss(logits, input_ids, padding_mask):
    """
    logits:       [B, L, V]
    input_ids:    [B, L]
    padding_mask: [B, L]   # 1 = real token, 0 = padding
    """

    B, L, V = logits.shape

    # Convert logits to probabilities
    probs = np.exp(logits)
    probs = probs / probs.sum(axis=-1, keepdims=True)

    # Target token at each position
    targets = input_ids

    # Pick probability assigned to target token
    target_probs = probs[
        np.arange(B)[:, None],
        np.arange(L)[None, :],
        targets
    ]

    # Cross-entropy
    token_loss = -np.log(target_probs)

    # Ignore padding
    token_loss = token_loss * padding_mask

    return token_loss.sum() / padding_mask.sum()
```

For a decoder-only causal LM:

```text
input_ids = [A, B, C, D]
```

the model should learn:

```text
A -> B
B -> C
C -> D
```

The goal of the interview is to identify correctness bugs, numerical-stability bugs, masking bugs, and training-loop issues.

---

# Part I — Correctness of Causal-LM Loss

## 2. Bug #1: Targets Are Not Shifted

The original implementation does:

```python
targets = input_ids
```

That means the logits at position `t` are compared against the token at the same position `t`.

For:

```text
[A, B, C, D]
```

this incorrectly trains:

```text
A -> A
B -> B
C -> C
D -> D
```

A causal LM instead predicts the **next token**.

Correct alignment:

```python
shift_logits = logits[:, :-1, :]   # [B, L-1, V]
shift_targets = input_ids[:, 1:]   # [B, L-1]
```

For:

```text
input_ids = [A, B, C, D]
```

we now get:

```text
logit at A predicts B
logit at B predicts C
logit at C predicts D
```

### Interview rule

> **Shift logits left and labels right.**

Equivalent PyTorch pattern:

```python
shift_logits = logits[..., :-1, :].contiguous()
shift_labels = labels[..., 1:].contiguous()
```

---

## 3. Shapes After Shifting

Original tensors:

```text
logits:        [B, L, V]
input_ids:     [B, L]
padding_mask:  [B, L]
```

After shifting:

```text
shift_logits:  [B, L-1, V]
shift_targets: [B, L-1]
shift_mask:    [B, L-1]
```

If:

```text
B = 2
L = 512
V = 50000
```

then:

```text
shift_logits  = [2, 511, 50000]
shift_targets = [2, 511]
shift_mask    = [2, 511]
```

---

# Part II — Numerical Stability

## 4. Bug #2: Naive Softmax Can Overflow

The original code:

```python
probs = np.exp(logits)
probs = probs / probs.sum(axis=-1, keepdims=True)
```

is numerically unstable.

Suppose one logit is:

```text
1000
```

Then:

```text
exp(1000) -> inf
```

which can produce:

```text
inf / inf -> NaN
```

### Stable softmax

Subtract the maximum logit:

```python
stable_logits = shift_logits - np.max(
    shift_logits,
    axis=-1,
    keepdims=True
)

exp_logits = np.exp(stable_logits)

probs = exp_logits / exp_logits.sum(
    axis=-1,
    keepdims=True
)
```

Why this works:

```text
softmax(z) = softmax(z - c)
```

for any constant `c`.

Choose:

```text
c = max(z)
```

so the largest shifted logit becomes `0`.

Therefore:

```text
exp(0) = 1
```

and all other exponentials are at most `1`.

---

## 5. Bug #3: `log(0)`

The original loss:

```python
token_loss = -np.log(target_probs)
```

fails if:

```text
target_probs == 0
```

Then:

```text
-log(0) = +inf
```

A simple defensive fix is:

```python
eps = 1e-12

token_loss = -np.log(
    np.maximum(target_probs, eps)
)
```

or:

```python
token_loss = -np.log(
    np.clip(target_probs, 1e-12, 1.0)
)
```

However, this is still not the best implementation.

The preferred approach is to compute cross-entropy directly from logits using **log-sum-exp**.

---

# Part III — Stable Cross-Entropy

## 6. Cross-Entropy From Logits

For logits:

```text
z = [z_1, z_2, ..., z_V]
```

and target class `y`:

```text
p_y = exp(z_y) / sum_j exp(z_j)
```

Cross-entropy:

```text
CE = -log(p_y)
```

Substitute the softmax:

```text
CE
= -log(exp(z_y) / sum_j exp(z_j))
```

Rearrange:

```text
CE
= log(sum_j exp(z_j)) - z_y
```

Therefore:

```text
CE(z, y) = logsumexp(z) - z_y
```

This is one of the most useful formulas to remember in LLM interviews.

---

## 7. Stable Log-Sum-Exp

Naively:

```text
log(sum(exp(z)))
```

may overflow.

Let:

```text
m = max(z)
```

Then:

```text
logsumexp(z)
= m + log(sum(exp(z - m)))
```

Because every value in:

```text
z - m
```

is `<= 0`, the exponential is numerically safe.

---

## 8. Example: Stable Cross-Entropy

Suppose:

```text
logits = [1, 2, 3]
target = class 1
```

### Step 1: maximum

```text
m = 3
```

### Step 2: shift logits

```text
[-2, -1, 0]
```

### Step 3: exponentiate

```text
exp(-2) ≈ 0.1353
exp(-1) ≈ 0.3679
exp(0)  = 1
```

Sum:

```text
1.5032
```

### Step 4: log-sum-exp

```text
logsumexp
= 3 + log(1.5032)
≈ 3 + 0.4076
≈ 3.4076
```

Target logit:

```text
z_y = 2
```

Loss:

```text
3.4076 - 2
≈ 1.4076
```

So:

```text
CE ≈ 1.408
```

### Formula to memorize

```text
CE(z, y) = logsumexp(z) - z_y
```

---

# Part IV — Padding and Mask Alignment

## 9. Bug #4: Padding Mask Must Follow the Target

Suppose:

```text
input_ids    = [A, B, C, PAD, PAD]
padding_mask = [1, 1, 1, 0,   0]
```

After label shifting:

```text
shift_targets = [B, C, PAD, PAD]
```

The correct shifted mask is:

```python
shift_mask = padding_mask[:, 1:]
```

giving:

```text
[1, 1, 0, 0]
```

Predictions:

```text
A   -> B      valid
B   -> C      valid
C   -> PAD    ignore
PAD -> PAD    ignore
```

---

## 10. Why `padding_mask[:, :-1]` Is Wrong

If we use:

```python
padding_mask[:, :-1]
```

we get:

```text
[1, 1, 1, 0]
```

This incorrectly says:

```text
C -> PAD
```

should contribute to training.

But the target token is padding, so this loss must be ignored.

### Interview rule

> **Mask according to whether the target token is valid, not whether the input token is valid.**

So:

```python
shift_mask = padding_mask[:, 1:]
```

---

## 11. Bug #5: Wrong Loss Denominator

The original implementation divides by:

```python
padding_mask.sum()
```

After shifting, the valid training positions correspond to:

```python
shift_mask
```

Therefore the correct denominator is:

```python
shift_mask.sum()
```

Correct:

```python
return token_loss.sum() / shift_mask.sum()
```

Why?

The first token is context.

It is not itself a prediction target.

For:

```text
[A, B, C, D]
```

we only have three next-token targets:

```text
B
C
D
```

not four.

---

# Part V — Full Stable NumPy Loss

## 12. Corrected Implementation

```python
import numpy as np


def causal_lm_loss(logits, input_ids, padding_mask):
    """
    logits:       [B, L, V]
    input_ids:    [B, L]
    padding_mask: [B, L]

    padding_mask:
        1 = valid token
        0 = padding
    """

    B, L, V = logits.shape

    # 1. Shift for next-token prediction
    shift_logits = logits[:, :-1, :]      # [B, L-1, V]
    shift_targets = input_ids[:, 1:]      # [B, L-1]
    shift_mask = padding_mask[:, 1:]      # [B, L-1]

    # 2. Stable log-sum-exp
    max_logits = np.max(
        shift_logits,
        axis=-1,
        keepdims=True
    )

    log_sum_exp = (
        max_logits
        + np.log(
            np.sum(
                np.exp(shift_logits - max_logits),
                axis=-1,
                keepdims=True
            )
        )
    )

    # 3. Gather target logits
    target_logits = shift_logits[
        np.arange(B)[:, None],
        np.arange(L - 1)[None, :],
        shift_targets
    ]

    # 4. Cross-entropy
    token_loss = (
        log_sum_exp.squeeze(-1)
        - target_logits
    )

    # 5. Mask padding targets
    token_loss = token_loss * shift_mask

    # 6. Normalize over valid targets
    num_valid = shift_mask.sum()

    if num_valid == 0:
        return 0.0

    return token_loss.sum() / num_valid
```

---

# Part VI — NaN Debugging

## 13. Edge Case: No Valid Target Tokens

Suppose a batch contains:

```text
padding_mask =
[
    [1, 0, 0, 0],
    [1, 0, 0, 0]
]
```

After shifting:

```text
shift_mask =
[
    [0, 0, 0],
    [0, 0, 0]
]
```

Therefore:

```text
shift_mask.sum() = 0
```

If the numerator is also zero:

```text
loss = 0 / 0
```

which becomes:

```text
NaN
```

Minimum guard:

```python
num_valid = shift_mask.sum()

if num_valid == 0:
    return 0.0
```

In production, returning zero is not enough.

You should also investigate why the batch contains no trainable targets.

Possible causes:

- tokenization produced only one valid token
- aggressive truncation
- filtering bug
- malformed padding mask
- empty examples
- batching bug

---

## 14. NaN Debugging Checklist

If training suddenly goes:

```text
step 100: loss = 6.8
step 200: loss = 6.2
step 300: loss = 5.9
step 301: loss = inf
step 302: loss = nan
```

check systematically:

1. Are logits finite?

```python
torch.isfinite(logits).all()
```

2. Is the loss finite?

```python
torch.isfinite(loss)
```

3. Do gradients contain NaN or Inf?

```python
for p in model.parameters():
    if p.grad is not None:
        if not torch.isfinite(p.grad).all():
            ...
```

4. Is there a division by zero?

5. Is there a `log(0)`?

6. Is naive `exp()` overflowing?

7. Are target IDs valid?

```text
0 <= target < V
```

8. Is the learning rate too high?

9. Is mixed precision overflowing?

10. Are activations becoming unstable?

---

# Part VII — Gradient Clipping

## 15. Correct Training-Step Order

A common training loop:

```python
optimizer.zero_grad()

loss.backward()

torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0
)

optimizer.step()

scheduler.step()
```

Correct order:

```text
zero_grad
    ↓
forward
    ↓
loss
    ↓
backward
    ↓
gradient clipping
    ↓
optimizer.step
    ↓
scheduler.step
```

### Why clipping comes after backward

Before:

```python
loss.backward()
```

the gradients do not exist yet.

So clipping cannot meaningfully happen before backward.

### Why clipping comes before optimizer step

The optimizer should update parameters using the clipped gradients.

---

## 16. Gradient Clipping Is Not a Root-Cause Fix

If gradients explode, clipping may prevent catastrophic updates.

But still investigate:

- learning rate too large
- optimizer instability
- bad initialization
- corrupted data
- wrong labels
- unstable normalization
- FP16 overflow
- exploding activations
- unusually long sequences

---

# Part VIII — Gradient Accumulation

## 17. Goal of Gradient Accumulation

Suppose:

```text
microbatch size = 8
desired effective batch size = 32
```

Then:

```text
accumulation_steps = 4
```

We process four microbatches before calling:

```python
optimizer.step()
```

---

## 18. Bug: Calling `zero_grad()` Every Microbatch

Incorrect:

```python
for i, batch in enumerate(loader):
    optimizer.zero_grad()

    loss = model(batch)
    loss.backward()

    if (i + 1) % 4 == 0:
        optimizer.step()
```

This destroys accumulated gradients.

Why?

PyTorch accumulates gradients into:

```text
parameter.grad
```

but:

```python
optimizer.zero_grad()
```

clears them before the next microbatch.

So instead of:

```text
g1 + g2 + g3 + g4
```

you effectively keep only:

```text
g4
```

---

## 19. Correct Gradient Accumulation

```python
accumulation_steps = 4

optimizer.zero_grad()

for i, batch in enumerate(loader):

    loss = model(batch)

    loss = loss / accumulation_steps

    loss.backward()

    if (i + 1) % accumulation_steps == 0:

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
```

---

## 20. Why Divide Loss by `accumulation_steps`

Suppose gradients for four microbatches are:

```text
g1
g2
g3
g4
```

Without dividing:

```text
g_accum = g1 + g2 + g3 + g4
```

which is roughly 4× the mean gradient.

With:

```python
loss = loss / 4
```

the gradient becomes approximately:

```text
(g1 + g2 + g3 + g4) / 4
```

which matches the average gradient of the effective batch when microbatches are equally weighted.

---

## 21. Subtlety: Variable-Length LLM Batches

In language modeling, microbatches can have different numbers of valid target tokens.

Example:

```text
microbatch 1: 100 valid tokens
microbatch 2: 100 valid tokens
microbatch 3: 10 valid tokens
microbatch 4: 10 valid tokens
```

If each microbatch loss is already averaged independently and then divided by 4, each microbatch gets equal weight.

That is not identical to averaging over all valid tokens.

More exact token-level normalization is:

```text
total_loss_sum / total_valid_tokens
```

across the accumulation window.

This matters when sequence lengths vary significantly.

---

# Part IX — Leftover Microbatches

## 22. Bug: Final Partial Accumulation Window

Suppose:

```text
len(loader) = 10
accumulation_steps = 4```

Optimizer steps happen after:

```text
batches 1-4
batches 5-8
```

What about:

```text
batches 9-10
```

Their gradients are computed but never applied.

Correct condition:

```python
should_step = (
    (i + 1) % accumulation_steps == 0
    or
    (i + 1) == len(loader)
)
```

---

## 23. Correct Loop With Final Partial Window

```python
optimizer.zero_grad()

for i, batch in enumerate(loader):

    loss = model(batch)
    loss = loss / accumulation_steps
    loss.backward()

    should_step = (
        (i + 1) % accumulation_steps == 0
        or
        (i + 1) == len(loader)
    )

    if should_step:

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
```

---

## 24. Subtle Bug in the Final Partial Window

Suppose:

```text
accumulation_steps = 4
```

but the final window has only:

```text
2 microbatches
```

If each loss is divided by 4, then the final update is under-scaled by about 2×.

For an exact implementation, normalize by the actual number of microbatches in the final window, or better, normalize by valid target-token count.

---

# Part X — Mixed Precision

## 25. FP16 Problem: Gradient Underflow

FP16 has limited numerical range.

During backward, some gradients may become very small.

Example:

```text
true gradient = 1e-8
```

In FP16 this may round to:

```text
0
```

This is called:

```text
underflow
```

Once the gradient becomes zero, that information is lost.

---

## 26. Loss Scaling

Loss scaling intentionally multiplies the loss by a large scale before backward:

```python
scaled_loss = loss * scale
scaled_loss.backward()
```

Example:

```text
loss = 0.001
scale = 65536

scaled_loss = 65.536
```

Differentiation is linear:

```text
grad(scale * loss)
=
scale * grad(loss)
```

So:

```text
1e-8 * 65536
≈ 6.55e-4
```

which is much easier to represent in FP16.

---

## 27. Unscaling

Before using the gradients, convert them back to the true scale:

```text
true_grad
=
scaled_grad / scale
```

Conceptually:

```python
scaled_loss.backward()

unscale_gradients()

clip_grad_norm_(...)

optimizer.step()
```

---

## 28. Why Unscale Before Gradient Clipping

Suppose:

```text
true gradient norm = 0.5
scale = 65536
```

The scaled gradient norm is enormous:

```text
32768
```

If you clip before unscaling, the clipping decision is based on an artificial magnitude.

Therefore:

```text
scaled backward
    ↓
unscale
    ↓
clip
    ↓
optimizer step
```

---

# Part XI — PyTorch GradScaler

## 29. Typical FP16 AMP Training Step

```python
scaler = torch.amp.GradScaler("cuda")

optimizer.zero_grad(set_to_none=True)

with torch.autocast(
    device_type="cuda",
    dtype=torch.float16
):
    logits = model(input_ids)
    loss = causal_lm_loss(
        logits,
        labels,
        padding_mask
    )

scaler.scale(loss).backward()

scaler.unscale_(optimizer)

torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0
)

scaler.step(optimizer)

scaler.update()
```

Correct mnemonic:

```text
Clear
  ↓
Forward
  ↓
Scale
  ↓
Backward
  ↓
Unscale
  ↓
Clip
  ↓
Step
  ↓
Update scaler
```

Short interview mnemonic:

> **Clear → Backward → Unscale → Clip → Step**

---

## 30. Dynamic Loss Scaling

A fixed scale can itself become too large.

If scaled gradients overflow to:

```text
Inf
```

or:

```text
NaN
```

`GradScaler` can detect that condition.

It may:

```text
skip optimizer.step
reduce the scale
try again on later iterations
```

This provides a balance:

- scale large enough to avoid underflow
- not so large that gradients overflow

---

# Part XII — FP16 vs BF16

## 31. FP16

FP16 has:

```text
5 exponent bits
10 fraction bits
```

Main issue:

```text
limited dynamic range
```

Common numerical problems:

- underflow
- overflow
- Inf
- NaN

Loss scaling is commonly used.

---

## 32. BF16

BF16 has:

```text
8 exponent bits
7 fraction bits
```

Its exponent range is roughly similar to FP32.

That means BF16 is much less vulnerable to range-related underflow and overflow.

Therefore:

> **BF16 usually does not need GradScaler.**

Tradeoff:

BF16 has fewer mantissa bits, so it has lower precision than FP16 for values that are representable in both formats.

---

# Part XIII — Scheduler Ordering Nuance

## 33. Normal Case

A common order is:

```python
optimizer.step()
scheduler.step()
```

rather than:

```python
scheduler.step()
optimizer.step()
```

because the scheduler generally advances after an optimizer update.

---

## 34. GradScaler Nuance

With FP16, `GradScaler` may skip an optimizer step when it detects Inf/NaN gradients.

In that case, ideally the learning-rate scheduler should not advance as though an optimizer update occurred.

This matters in precise training-loop implementations.

---

# Part XIV — Full Mixed-Precision + Accumulation Pattern

## 35. Example

```python
scaler = torch.amp.GradScaler("cuda")

accumulation_steps = 4

optimizer.zero_grad(set_to_none=True)

for i, batch in enumerate(loader):

    with torch.autocast(
        device_type="cuda",
        dtype=torch.float16
    ):
        loss = model(batch)

        loss = loss / accumulation_steps

    scaler.scale(loss).backward()

    should_step = (
        (i + 1) % accumulation_steps == 0
        or
        (i + 1) == len(loader)
    )

    if should_step:

        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        scaler.step(optimizer)

        scaler.update()

        optimizer.zero_grad(set_to_none=True)

        scheduler.step()
```

Important ideas:

- do not unscale every microbatch
- do not clip every microbatch
- do not step every microbatch
- accumulate scaled gradients
- unscale once at the optimizer boundary
- clip after unscale
- update GradScaler once per optimizer step

---

# Part XV — PyTorch Causal-LM Loss Version

## 36. Using CrossEntropyLoss

PyTorch already implements numerically stable cross-entropy.

```python
import torch
import torch.nn.functional as F


def causal_lm_loss_torch(
    logits,
    input_ids,
    padding_mask
):
    """
    logits:       [B, L, V]
    input_ids:    [B, L]
    padding_mask: [B, L]
    """

    B, L, V = logits.shape

    shift_logits = logits[:, :-1, :]
    shift_targets = input_ids[:, 1:]
    shift_mask = padding_mask[:, 1:]

    # Flatten
    flat_logits = shift_logits.reshape(-1, V)
    flat_targets = shift_targets.reshape(-1)
    flat_mask = shift_mask.reshape(-1).bool()

    # Keep only valid targets
    valid_logits = flat_logits[flat_mask]
    valid_targets = flat_targets[flat_mask]

    if valid_targets.numel() == 0:
        return logits.sum() * 0.0

    return F.cross_entropy(
        valid_logits,
        valid_targets
    )
```

---

## 37. Alternative With Ignore Index

A common production-style approach is:

```python
labels = input_ids[:, 1:].clone()

labels[padding_mask[:, 1:] == 0] = -100
```

Then:

```python
loss = F.cross_entropy(
    shift_logits.reshape(-1, V),
    labels.reshape(-1),
    ignore_index=-100
)
```

This is convenient because padding positions are ignored automatically.

---

# Part XVI — Interview Q&A Review

## 38. Question: What Is the Most Important Bug?

**Answer:**

The labels are not shifted.

A decoder-only causal LM predicts token `t+1` using hidden state at token `t`.

Correct:

```python
shift_logits = logits[:, :-1, :]
shift_targets = input_ids[:, 1:]
```

---

## 39. Question: Why Is `np.exp(logits)` Dangerous?

**Answer:**

Large logits may overflow:

```text
exp(1000) -> Inf
```

Use:

```python
logits - max(logits)
```

before exponentiation, or use log-sum-exp / framework cross-entropy directly.

---

## 40. Question: How Do We Avoid `log(0)`?

Basic answer:

```python
-np.log(np.maximum(p, eps))
```

Better answer:

> Avoid explicit probabilities and compute stable cross-entropy directly from logits.

---

## 41. Question: Which Padding Mask Should Be Used?

Correct:

```python
shift_mask = padding_mask[:, 1:]
```

because masking should follow the **target token**.

---

## 42. Question: What Happens If There Are No Valid Targets?

Then:

```text
shift_mask.sum() == 0
```

and:

```text
0 / 0 -> NaN
```

Guard the denominator and investigate the data pipeline.

---

## 43. Question: Gradient-Clipping Order

Correct:

```text
optimizer.zero_grad
    ↓
loss.backward
    ↓
clip gradients
    ↓
optimizer.step
```

---

## 44. Question: Gradient Accumulation Bug

Wrong:

```python
optimizer.zero_grad()
```

inside every microbatch.

Why:

It erases previously accumulated gradients.

Correct:

```text
zero once
backward several times
step
zero again
```

---

## 45. Question: Why Divide Loss During Accumulation?

To approximate the mean gradient across the effective batch:

```text
(g1 + g2 + ... + gN) / N
```

instead of the sum:

```text
g1 + g2 + ... + gN
```

---

## 46. Question: What Happens to Leftover Microbatches?

If:

```text
10 batches
accumulation_steps = 4
```

batches:

```text
9 and 10
```

must still trigger a final optimizer step.

---

## 47. Question: Why Scale an FP16 Loss Up?

To prevent tiny gradients from underflowing to zero.

Loss scaling increases gradient magnitude during backward.

---

## 48. Question: Correct FP16 Operation Order

Given:

```text
A. optimizer.step
B. gradient clipping
C. scaled loss backward
D. unscale gradients
E. optimizer.zero_grad
```

Correct order:

```text
E -> C -> D -> B -> A
```

Mnemonic:

```text
Clear -> Backward -> Unscale -> Clip -> Step
```

---

# Part XVII — High-Value Interview Takeaways

## 49. The Core Mental Model

For causal-LM training, always think in this order:

```text
Input tokens
    ↓
Next-token alignment
    ↓
Valid-target masking
    ↓
Stable cross-entropy
    ↓
Valid-token normalization
    ↓
Backward
    ↓
Gradient stability
    ↓
Optimizer update
```

---

## 50. Five Rules Worth Memorizing

### Rule 1

```text
shift logits left
shift labels right
```

### Rule 2

```text
mask according to target validity
```

### Rule 3

```text
CE(z, y) = logsumexp(z) - z_y
```

### Rule 4

```text
backward -> clip -> optimizer step
```

For FP16:

```text
backward -> unscale -> clip -> step
```

### Rule 5

Gradient accumulation:

```text
zero once
accumulate
step
zero again
```

---

# Part XVIII — Final Debugging Checklist

When a causal-LM loss implementation looks wrong, check:

- [ ] Are labels shifted by one position?
- [ ] Are logits and labels the same sequence length after shifting?
- [ ] Is padding masking based on shifted targets?
- [ ] Is loss normalized only by valid target tokens?
- [ ] Is softmax numerically stable?
- [ ] Is `log(0)` possible?
- [ ] Is there a zero-valid-token batch?
- [ ] Are target IDs within `[0, V-1]`?
- [ ] Are logits finite?
- [ ] Are gradients finite?
- [ ] Is gradient clipping done after backward?
- [ ] Is `zero_grad()` placed correctly?
- [ ] Is gradient accumulation normalized correctly?
- [ ] Are leftover microbatches stepped?
- [ ] Is FP16 loss scaling used when needed?
- [ ] Are gradients unscaled before clipping?
- [ ] Is the scheduler advanced only with real optimizer updates?

---

# Final Summary

Question 2 covered much more than just the loss formula.

It tested four layers of LLM training knowledge:

```text
1. Objective correctness
   - causal next-token shifting
   - padding / label alignment

2. Numerical stability
   - stable softmax
   - log-sum-exp
   - avoiding log(0)
   - zero-denominator NaNs

3. Optimization correctness
   - gradient clipping
   - optimizer / scheduler ordering
   - gradient accumulation
   - leftover microbatches

4. Mixed precision
   - FP16 underflow
   - loss scaling
   - dynamic GradScaler
   - unscale before clipping
   - BF16 vs FP16
```

The most compact interview answer is:

> A correct causal-LM loss shifts logits and labels for next-token prediction, masks according to shifted target validity, computes cross-entropy stably from logits, and normalizes over valid target tokens. In the training loop, gradients are accumulated carefully, clipped after backward, and with FP16 are unscaled before clipping and stepping.
