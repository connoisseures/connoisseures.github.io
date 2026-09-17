---
title: "LLM Pretraining Part 10: Evaluation and the Complete Pretraining Pipeline"
date: 2026-09-17 00:00:00 +0000
categories: [LLM, Pretraining]
tags: [perplexity, evaluation, adamw, mixed-precision, checkpointing]
description: "Token-weighted evaluation, perplexity, leakage prevention, AdamW, mixed precision, checkpointing, and the complete LLM pretraining loop."
math: true
---

# LLM Pretraining — Part 10: Evaluation and the Complete Pretraining Pipeline

Part 10 connects the entire pretraining system: data preparation, the Transformer forward pass, next-token loss, AdamW optimization, reliable evaluation, and resumable checkpoints.

## 1. Evaluation loss and perplexity

For valid target tokens indexed by $i$, the token-level negative log-likelihood is

$$
\ell_i = -\log p_\theta(y_i \mid x_{\le i}).
$$

The correct corpus-level mean loss is

$$
\bar{\ell} = \frac{\sum_i \ell_i}{N},
$$

where $N$ is the total number of valid, unmasked target tokens. Perplexity is

$$
\mathrm{PPL} = \exp(\bar{\ell}).
$$

If the mean loss is $2$, then

$$
\mathrm{PPL} = e^2 \approx 7.39.
$$

Intuitively, lower perplexity means the model assigns more probability to the observed continuation. It is not literally the number of choices available at every token, but it can be understood as an effective branching factor.

### Aggregate by tokens, not batches

Suppose:

- Batch A has loss sum $180$ across $90$ valid tokens.
- Batch B has loss sum $60$ across $30$ valid tokens.

Then

$$
\bar{\ell} = \frac{180 + 60}{90 + 30} = 2.
$$

Do not average batch perplexities. Exponentiation is nonlinear, and batches may contain different numbers of valid tokens. Accumulate the total loss sum and valid-token count, divide once, and exponentiate once.

```python
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_nll = torch.zeros((), device=device, dtype=torch.float64)
    total_tokens = torch.zeros((), device=device, dtype=torch.long)

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids)  # [B, T, V]
        loss_sum = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )
        valid_tokens = (labels != -100).sum()

        total_nll += loss_sum.double()
        total_tokens += valid_tokens

    mean_loss = total_nll / total_tokens
    perplexity = mean_loss.exp()
    return mean_loss.item(), perplexity.item()
```

In distributed evaluation, all-reduce `total_nll` and `total_tokens` across workers before computing the mean.

## 2. When perplexity comparisons are valid

Perplexity is directly comparable only when the evaluation protocol is aligned. Important variables include:

- tokenizer and vocabulary;
- exact evaluation corpus and preprocessing;
- context length and sliding-window policy;
- treatment of document boundaries;
- ignored tokens and loss mask;
- whether beginning-of-sequence tokens receive reduced context.

A tokenizer that splits text into more tokens changes the denominator and therefore changes token-level perplexity. For cross-tokenizer comparisons, also report a normalized metric such as bits per byte when appropriate.

### Long-document evaluation

For documents longer than the context window, use a sliding window. Score each token once while providing as much left context as possible. Do not repeatedly count the overlapping targets.

## 3. Prevent evaluation leakage

Randomly splitting already-created chunks is unsafe when related chunks come from the same source. Leakage can occur through:

- exact duplicates;
- near-duplicates;
- adjacent chunks from the same document;
- mirrors or quotations of the same source;
- later versions of the same content;
- content from the same user, thread, or session;
- benchmark questions included in training data.

Split at the highest meaningful group level—such as source document, repository, domain, user, or time period—then tokenize and pack the splits separately. Deduplicate both within and across splits.

## 4. Evaluation mode versus disabled gradients

These operations solve different problems:

```python
model.eval()
with torch.no_grad():
    logits = model(input_ids)
```

- `model.eval()` changes module behavior, such as disabling dropout and using evaluation behavior for modules that distinguish training from evaluation.
- `torch.no_grad()` prevents construction of the autograd graph, reducing memory and computation.

Neither performs an optimizer update. For ordinary evaluation, use both.

## 5. Complete tensor-shape walkthrough

Assume:

- batch size $B=2$;
- sequence length $T=512$;
- vocabulary size $V=50{,}000$;
- model width $d_{model}=1024$;
- heads $H=16$;
- head width $d_{head}=64$;
- feed-forward width $d_{ff}=2816$;
- layers $L=24$.

| Tensor | Shape |
|---|---:|
| Input token IDs | `[2, 512]` |
| Token embeddings | `[2, 512, 1024]` |
| Q, K, V before head split | `[2, 512, 1024]` each |
| Q, K, V after head split | `[2, 16, 512, 64]` each |
| Attention scores | `[2, 16, 512, 512]` |
| Attention output after head merge | `[2, 512, 1024]` |
| MLP intermediate | `[2, 512, 2816]` |
| Final hidden states | `[2, 512, 1024]` |
| Vocabulary logits | `[2, 512, 50000]` |
| Flattened logits for cross-entropy | `[1024, 50000]` |
| Flattened targets | `[1024]` |

If a raw token stream contains 513 tokens, it can be shifted into 512 input tokens and 512 targets. The model's `input_ids` tensor in this example is still `[2, 512]`.

## 6. The complete training pipeline

1. Ingest raw documents and retain source metadata.
2. Normalize and filter content.
3. Remove exact and near duplicates.
4. Create leakage-safe train, validation, and test splits.
5. Tokenize each split.
6. Pack token sequences and build labels, masks, and boundary handling.
7. Run token embeddings and positional representations.
8. Pass hidden states through all Transformer blocks.
9. Apply the final normalization and vocabulary projection.
10. Compute masked next-token cross-entropy.
11. Backpropagate, clip gradients if needed, and perform an AdamW update.
12. Advance the learning-rate schedule.
13. Log training signals, evaluate periodically, and save checkpoints.

## 7. Gradient accumulation

Gradient accumulation simulates a larger effective batch with several microbatches. With four equal-size microbatches, divide each mean microbatch loss by four before calling `backward()`:

```python
optimizer.zero_grad(set_to_none=True)

for _ in range(4):
    with autocast_context:
        logits = model(input_ids)
        loss = compute_loss(logits, labels)
        scaled_loss = loss / 4
    scaler.scale(scaled_loss).backward()

scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
scaler.step(optimizer)
scaler.update()
scheduler.step()
```

This simple division is exact when each microbatch contributes the same number of valid target tokens. With variable valid-token counts, accumulate the summed token loss and normalize gradients by the total valid-token count for the entire optimizer step.

## 8. Mixed precision

Mixed precision reduces memory use and can increase throughput.

- **BF16** has the same exponent range as FP32 and usually does not require dynamic loss scaling.
- **FP16** has a smaller exponent range, so gradient scaling is commonly used to avoid underflow.

For FP16 with `GradScaler`, unscale gradients before clipping them:

```python
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
scaler.step(optimizer)
scaler.update()
```

Clipping scaled gradients would apply the threshold to artificial scaled values rather than the real gradients.

## 9. AdamW in depth

### Adam's adaptive update

For gradient $g_t$, Adam tracks exponential moving averages of the first and second moments:

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t,
$$

$$
v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2.
$$

Because both start at zero, Adam applies bias correction:

$$
\hat m_t = \frac{m_t}{1-\beta_1^t}, \qquad
\hat v_t = \frac{v_t}{1-\beta_2^t}.
$$

The adaptive parameter update is

$$
\theta_t = \theta_{t-1} - \eta
\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.
$$

Coordinates with historically large squared gradients receive smaller effective steps, while coordinates with smaller squared gradients receive larger effective steps.

### Why ordinary L2 regularization is different in Adam

L2 regularization adds $\lambda\theta$ to the gradient. Under SGD, this is algebraically equivalent to shrinking every parameter by a common factor. Under Adam, however, the regularization term passes through the moment estimates and coordinate-wise normalization. The resulting shrinkage is no longer uniform.

AdamW decouples weight decay from the adaptive gradient update:

$$
\theta_t = (1-\eta\lambda)\theta_{t-1}
- \eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.
$$

This has two independent parts:

1. shrink the parameter by the decay factor $1-\eta\lambda$;
2. apply Adam's adaptive update using the data gradient.

If $\theta=2$, learning rate $\eta=10^{-4}$, and weight decay $\lambda=0.1$, the decay-only factor is

$$
1-\eta\lambda=0.99999,
$$

so the decay-only value becomes $1.99998$. A weight decay of `0.1` does **not** mean that 10% of the parameter is removed on every step; its per-step effect is multiplied by the learning rate.

### Parameter groups

A common Transformer recipe applies weight decay to weight matrices but not to bias vectors or normalization scale parameters:

```python
decay, no_decay = [], []

for name, parameter in model.named_parameters():
    if not parameter.requires_grad:
        continue
    if parameter.ndim >= 2:
        decay.append(parameter)
    else:
        no_decay.append(parameter)

optimizer = torch.optim.AdamW(
    [
        {"params": decay, "weight_decay": 0.1},
        {"params": no_decay, "weight_decay": 0.0},
    ],
    lr=3e-4,
    betas=(0.9, 0.95),
    eps=1e-8,
)
```

Whether to decay embedding matrices is recipe-dependent. If token embeddings and the language-model head are tied, the shared parameter must occur only once across all optimizer groups.

### Memory cost

AdamW normally stores two optimizer-state tensors per parameter: the first moment `m` and second moment `v`. If both use FP32, they consume about eight bytes per parameter in total, excluding parameters, gradients, and any master-weight copy. For seven billion parameters, the two moment tensors alone require roughly 56 GB before sharding.

## 10. Learning-rate schedule and gradient clipping

A common schedule combines linear warmup with cosine decay:

- warmup prevents unstable large updates while optimizer statistics and activations are still settling;
- decay reduces the step size later so training can refine the solution.

Gradient clipping limits unusually large gradient norms:

```python
total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

Clipping is a safety mechanism, not a substitute for fixing persistent instability caused by an excessive learning rate, bad data, numerical overflow, or an architectural bug.

## 11. Correct optimizer-step order

At the beginning of an accumulation cycle, clear old gradients. Then:

1. run all microbatch forward and backward passes;
2. unscale gradients when using FP16 gradient scaling;
3. clip gradients;
4. call the optimizer step;
5. update the gradient scaler, if used;
6. call the scheduler step;
7. begin the next cycle with zeroed gradients.

`optimizer.zero_grad()` may be placed immediately after an optimizer step instead, as long as stale gradients are cleared before the next backward pass. The scheduler normally advances after the optimizer so the intended learning rate is associated with the correct update.

## 12. Activation checkpointing versus training checkpoints

These are different concepts:

- **Activation checkpointing** saves memory during a forward pass by storing fewer activations and recomputing them during backward.
- **Training checkpoints** persist state to storage so training can resume after interruption.

A robust training checkpoint commonly stores:

1. model parameters;
2. optimizer state, including AdamW moments;
3. learning-rate scheduler state;
4. training progress such as optimizer step and tokens seen;
5. random-number-generator states;
6. mixed-precision gradient-scaler state when using FP16;
7. data-loader or sampler state for exact data-order resumption;
8. relevant configuration and tokenizer identity.

Saving only model weights permits inference but does not reproduce an exact training resume.

## 13. Monitoring the run

Track at least:

- training loss and validation loss;
- learning rate;
- gradient norm;
- tokens per second and step time;
- valid tokens per batch;
- memory use;
- overflow or skipped-step counts;
- data-loader time;
- checkpoint and evaluation duration.

Interpret combinations of signals. For example:

- falling training loss with rising validation loss suggests overfitting or distribution mismatch;
- sudden loss spikes with large gradient norms suggest unstable updates or anomalous data;
- flat loss with a near-zero learning rate suggests a scheduler problem;
- falling throughput with unchanged shapes may indicate input-pipeline or system issues.

## 14. Simplified end-to-end training loop

```python
model.train()

for optimizer_step in range(max_steps):
    optimizer.zero_grad(set_to_none=True)

    for micro_step in range(grad_accum_steps):
        batch = next(train_iterator)
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        with autocast_context:
            logits = model(input_ids)  # [B, T, V]
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
            loss = loss / grad_accum_steps

        scaler.scale(loss).backward()

    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()

    if optimizer_step % eval_interval == 0:
        validation_loss, validation_ppl = evaluate(model, validation_loader, device)
        model.train()

    if optimizer_step % checkpoint_interval == 0:
        save_checkpoint(...)
```

In distributed training, synchronize only where needed, reduce metrics correctly, and ensure every rank agrees on whether an optimizer step was skipped before advancing the scheduler.

## 15. Concept-check review

| # | Submitted answer | Review | Correct answer |
|---:|---|---|---|
| 1 | `(180 + 60) / (90 + 30) = 2` | Correct | Global mean loss is `2`. |
| 2 | `exp(2)` | Correct | Perplexity is $e^2 \approx 7.39$. |
| 3 | Consider it by tokens | Correct | Sum token NLL and token counts globally; do not average per-batch perplexities. |
| 4 | `input_ids [2,513]`; flattened logits `[2,1]` | Needs correction | Input IDs `[2,512]`; hidden states `[2,512,1024]`; logits `[2,512,50000]`; flattened logits `[1024,50000]`; flattened targets `[1024]`. |
| 5 | `eval()` skips dropout; `no_grad()` prevents gradient descent | Mostly correct | `eval()` changes layer behavior; `no_grad()` disables graph construction and gradient tracking. Neither itself performs gradient descent. |
| 6 | Avoid data leakage | Correct | Related chunks must remain in one split so source overlap does not inflate validation/test performance. |
| 7 | Unknown | Filled in | With four equal-size microbatches, divide each microbatch mean loss by `4` before `backward()`. |
| 8 | Unknown | Filled in | With FP16, call `scaler.unscale_(optimizer)` **before** gradient clipping. |
| 9 | Optimizer state | Incomplete | Store model, optimizer, scheduler, progress counters, and RNG states at minimum; also scaler and sampler state when applicable. |
| 10 | Backward → clip → scheduler → optimizer → zero | Needs correction | Zero gradients → backward pass(es) → FP16 unscale → clip → optimizer step → scaler update → scheduler step. |

## Interview-ready summary

Reliable LLM evaluation aggregates negative log-likelihood by valid tokens and computes perplexity only after global reduction. Train/validation/test splits must be leakage-safe at the source level. During training, gradient accumulation, mixed precision, AdamW, clipping, and scheduling must occur in the correct order. A resumable checkpoint contains far more than model weights: it must restore optimizer dynamics, scheduling, progress, randomness, and often data order.

