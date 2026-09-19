---
layout: post
title: "LLM Coding & Debugging Interview — Question 2: Debugging Causal-LM Training Loss"
date: 2026-09-19
categories: [interview-preparation, llm]
tags: [causal-lm, training, debugging, mixed-precision, gradient-accumulation]
---

# LLM Coding & Debugging Interview — Question 2: Debugging Causal-LM Training Loss

## Prompt
Debug a causal language-model training loss implementation and identify correctness and numerical-stability issues.

## 1. Shift labels for next-token prediction
```python
shift_logits = logits[:, :-1, :]
shift_targets = input_ids[:, 1:]
shift_mask = padding_mask[:, 1:]
```

For `[A, B, C, D]`, the model should learn `A -> B`, `B -> C`, `C -> D`.

**Interview rule:** Shift logits left, labels right, and use the shifted target mask.

## 2. Stable softmax
Naive `np.exp(logits)` can overflow. Subtract the maximum first:
```python
stable_logits = shift_logits - np.max(shift_logits, axis=-1, keepdims=True)
exp_logits = np.exp(stable_logits)
probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
```

## 3. Avoid log(0)
```python
eps = 1e-12
token_loss = -np.log(np.maximum(target_probs, eps))
```
Better: compute cross-entropy directly from logits with log-sum-exp.

## 4. Stable cross-entropy with log-sum-exp
`CE(z, y) = logsumexp(z) - z_y`.

```python
def causal_lm_loss(logits, input_ids, padding_mask):
    B, L, V = logits.shape
    shift_logits = logits[:, :-1, :]
    shift_targets = input_ids[:, 1:]
    shift_mask = padding_mask[:, 1:]

    m = np.max(shift_logits, axis=-1, keepdims=True)
    log_sum_exp = m + np.log(
        np.sum(np.exp(shift_logits - m), axis=-1, keepdims=True)
    )

    target_logits = shift_logits[
        np.arange(B)[:, None],
        np.arange(L - 1)[None, :],
        shift_targets
    ]

    token_loss = log_sum_exp.squeeze(-1) - target_logits
    token_loss = token_loss * shift_mask
    return token_loss.sum() / shift_mask.sum()
```

Example: logits `[1, 2, 3]`, target index `1`.
- max = 3
- shifted logits = `[-2, -1, 0]`
- logsumexp ≈ 3.408
- target logit = 2
- token loss ≈ 1.408

## 5. Shifted padding mask
For:
```text
input_ids    = [A, B, C, PAD, PAD]
padding_mask = [1, 1, 1, 0,   0]
```
we need:
```text
shift_targets = [B, C, PAD, PAD]
shift_mask    = [1, 1, 0,   0]
```
so `C -> PAD` is ignored.

**Rule:** Mask according to whether the target token is valid, not whether the input token is valid.

## 6. Correct denominator
Use:
```python
return token_loss.sum() / shift_mask.sum()
```
not `padding_mask.sum()`.

## 7. NaN from zero valid targets
If all shifted mask entries are zero, the denominator is zero and `0/0 -> NaN`.

```python
num_valid = shift_mask.sum()
if num_valid == 0:
    return 0.0
return token_loss.sum() / num_valid
```

Also investigate why the batch has no trainable targets.

## 8. NaN debugging checklist
- logits contain NaN/Inf?
- softmax / exp overflow?
- log(0)?
- division by zero?
- invalid targets?
- exploding gradients?
- mixed-precision overflow?

## 9. Gradient clipping
Correct order:
```python
optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
scheduler.step()
```
Clip after backward and before optimizer.step.

## 10. Gradient accumulation
Do not call `zero_grad()` every microbatch.

```python
accumulation_steps = 4
optimizer.zero_grad()

for i, batch in enumerate(loader):
    loss = model(batch)
    loss = loss / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

With clipping and scheduler:
```python
if (i + 1) % accumulation_steps == 0:
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
```

## 11. Leftover microbatches
If the loader length is not divisible by the accumulation count, step on the final batch too:
```python
should_step = (
    (i + 1) % accumulation_steps == 0
    or (i + 1) == len(loader)
)
```
For a partial final window, normalize carefully. With variable-length LLM batches, token-weighted normalization may be preferable.

## 12. FP16 loss scaling
FP16 can underflow very small gradients to zero. Scale the loss before backward:
```python
scaled_loss = loss * scale
scaled_loss.backward()
```
Then unscale gradients before clipping and stepping.

## 13. PyTorch GradScaler
```python
scaler = torch.amp.GradScaler("cuda")

optimizer.zero_grad()

with torch.autocast("cuda", dtype=torch.float16):
    logits = model(input_ids)
    loss = causal_lm_loss(logits, labels)

scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
scaler.step(optimizer)
scaler.update()
```

Mnemonic: **Clear → Backward → Unscale → Clip → Step**

Dynamic loss scaling can reduce the scale and skip an unsafe update when overflow is detected.

## 14. FP16 vs BF16
FP16 has a narrower exponent range, so loss scaling is commonly needed. BF16 has roughly FP32-like exponent range and usually does not require loss scaling, though it has less mantissa precision.

## Final interview summary
1. Shift labels for next-token prediction.
2. Shift the padding mask with the targets.
3. Normalize by valid shifted targets.
4. Prefer stable log-sum-exp cross-entropy.
5. Avoid `log(0)`.
6. Guard against zero valid targets.
7. Debug NaNs systematically.
8. Clip gradients after backward and before the optimizer step.
9. Preserve gradients across microbatches.
10. Normalize gradient accumulation correctly.
11. Handle leftover microbatches.
12. Use FP16 loss scaling to prevent tiny-gradient underflow.
13. Unscale before clipping.
14. Remember: **Clear → Backward → Unscale → Clip → Step**.
