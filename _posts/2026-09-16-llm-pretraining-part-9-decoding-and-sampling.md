---
title: "LLM Pretraining Part 9: Decoding and Sampling"
date: 2026-09-16 00:00:00 +0000
categories: [LLM, Pretraining]
tags: [decoding, sampling, temperature, top-k, top-p, kv-cache]
description: A shape-first guide to autoregressive generation, greedy decoding, temperature, top-k and top-p sampling, repetition controls, EOS handling, and KV-cached inference.
math: true
---

Parts 1–8 explained how an LLM produces vocabulary logits. This part explains how those logits become generated tokens.

## 1. Autoregressive generation

Given a prompt:

```text
"The capital of France is"
```

generation proceeds one token at a time:

```text
Step 1:
"The capital of France is"
→ logits
→ choose "Paris"

Step 2:
"The capital of France is Paris"
→ logits
→ choose "."

Step 3:
"The capital of France is Paris."
→ logits
→ choose EOS
```

Formally:

$$
x_{t+1}\sim P(x_{t+1}\mid x_1,\ldots,x_t)
$$

The decoding algorithm determines how the next token is selected from this probability distribution.

## 2. Shapes during generation

Suppose:

```text
B       = 4
T       = current context length
d_model = 1024
V       = 50,000
```

The Transformer produces:

```text
hidden_states: [4, T, 1024]
```

Only the final position is needed for the next token:

```python
last_hidden = hidden_states[:, -1, :]
```

```text
last_hidden: [4, 1024]
```

The language-modeling head produces:

```python
next_logits = lm_head(last_hidden)
```

```text
next_logits: [4, 50000]
```

After selection:

```text
next_tokens: [4]
```

or with a preserved token dimension:

```text
next_tokens: [4, 1]
```

## 3. Greedy decoding

Greedy decoding selects the token with the largest logit:

$$
x_{t+1}=\arg\max_v z_v
$$

```python
next_token = torch.argmax(
    next_logits,
    dim=-1,
)
```

Shapes:

```text
next_logits: [B, V]
next_token:  [B]
```

Softmax is unnecessary because it preserves ordering:

$$
\arg\max_v z_v=\arg\max_v\operatorname{softmax}(z)_v
$$

Advantages:

- Deterministic
- Fast
- Simple
- Reproducible for identical logits

Limitations:

- Can produce repetitive text
- May settle into locally likely but globally poor continuations
- Does not explore alternative plausible tokens

## 4. Multinomial sampling

Instead of always selecting the highest-probability token, sample from the distribution:

$$
x_{t+1}\sim\operatorname{Categorical}(p)
$$

```python
probs = torch.softmax(next_logits, dim=-1)

next_token = torch.multinomial(
    probs,
    num_samples=1,
)
```

Shapes:

```text
logits:     [B, V]
probs:      [B, V]
next_token: [B, 1]
```

Sampling introduces diversity, but unrestricted sampling may choose extremely unlikely tokens.

## 5. Temperature

Temperature rescales logits before softmax:

$$
p_i=\frac{\exp(z_i/\tau)}{\sum_j\exp(z_j/\tau)}
$$

where $\tau>0$.

```python
scaled_logits = logits / temperature
probs = torch.softmax(scaled_logits, dim=-1)
```

### Low temperature

When $\tau<1$, differences between logits become larger:

```text
original logits: [4.0, 2.0, 1.0]
temperature:      0.5
scaled logits:   [8.0, 4.0, 2.0]
```

The distribution becomes sharper and more deterministic.

### High temperature

When $\tau>1$:

```text
original logits: [4.0, 2.0, 1.0]
temperature:      2.0
scaled logits:   [2.0, 1.0, 0.5]
```

The distribution becomes flatter and more diverse.

As $\tau\rightarrow0^+$, sampling approaches greedy decoding. Implementations often interpret `temperature = 0` as greedy decoding rather than dividing by zero.

## 6. Numerically stable softmax

After applying temperature and other logit transformations:

```python
scaled_logits = logits / temperature

shifted_logits = (
    scaled_logits
    - scaled_logits.max(dim=-1, keepdim=True).values
)

probs = torch.exp(shifted_logits)
probs = probs / probs.sum(dim=-1, keepdim=True)
```

Top-k and top-p filters usually replace excluded logits with negative infinity:

```python
filtered_logits[remove_mask] = -torch.inf
```

Those tokens receive probability zero after softmax.

## 7. Top-k sampling

Top-k sampling keeps only the $k$ tokens with the largest logits.

For:

```text
V = 50,000
k = 50
```

only 50 tokens remain eligible.

```python
top_values, top_indices = torch.topk(
    logits,
    k=k,
    dim=-1,
)
```

Shapes:

```text
logits:      [B, 50000]
top_values:  [B, 50]
top_indices: [B, 50]
```

Sample within the reduced set:

```python
top_probs = torch.softmax(top_values, dim=-1)

sampled_position = torch.multinomial(
    top_probs,
    num_samples=1,
)  # [B, 1]

next_token = torch.gather(
    top_indices,
    dim=-1,
    index=sampled_position,
)  # [B, 1]
```

Top-k removes extremely unlikely tokens and gives a predictable candidate count, but it does not adapt to model uncertainty.

## 8. Top-p or nucleus sampling

Top-p retains the smallest set of high-probability tokens whose cumulative probability reaches $p$.

Example:

```text
token A: 0.40
token B: 0.30
token C: 0.15
token D: 0.08
token E: 0.07
```

With `p = 0.80`, retain A, B, and C because their cumulative probability is 0.85.

```python
def top_p_filter(logits, top_p):
    sorted_logits, sorted_indices = torch.sort(
        logits,
        descending=True,
        dim=-1,
    )

    sorted_probs = torch.softmax(
        sorted_logits,
        dim=-1,
    )

    cumulative_probs = torch.cumsum(
        sorted_probs,
        dim=-1,
    )

    remove = cumulative_probs > top_p

    # Keep the first token that crosses the threshold.
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False

    sorted_logits = sorted_logits.masked_fill(
        remove,
        -torch.inf,
    )

    filtered_logits = torch.full_like(
        logits,
        -torch.inf,
    )

    filtered_logits.scatter_(
        dim=-1,
        index=sorted_indices,
        src=sorted_logits,
    )

    return filtered_logits
```

The removal mask is shifted right so the first token that makes cumulative probability exceed $p$ is retained. Without the shift, the retained probability mass could remain below the requested threshold.

## 9. Top-k versus top-p

| Property | Top-k | Top-p |
|---|---|---|
| Candidate-set size | Fixed | Variable |
| Selection rule | Keep the $k$ largest logits | Keep enough tokens to reach probability $p$ |
| Adapts to uncertainty | No | Yes |
| Typical setting | `k=40` or `k=50` | `p=0.9` or `p=0.95` |

When the model is confident, top-p may retain very few tokens. When the distribution is flat, it may retain many.

## 10. Combining sampling controls

A reasonable processing order is:

```text
raw logits
→ repetition and token constraints
→ temperature scaling
→ top-k filtering
→ top-p filtering
→ softmax
→ multinomial sampling
```

```python
logits = apply_repetition_penalty(
    logits,
    generated_tokens,
)

logits = logits / temperature
logits = top_k_filter(logits, top_k)
logits = top_p_filter(logits, top_p)

probs = torch.softmax(logits, dim=-1)
next_token = torch.multinomial(probs, 1)
```

Processor order can vary between implementations, so an interview answer should state the chosen convention.

## 11. Repetition controls

Autoregressive models can enter loops because generated tokens become part of subsequent context.

### Repetition penalty

One common rule uses penalty $r>1$:

```python
if logit > 0:
    adjusted_logit = logit / r
else:
    adjusted_logit = logit * r
```

Example:

```text
positive logit:  4.0 / 1.2 = 3.33
negative logit: -2.0 × 1.2 = -2.40
```

Both changes make the token less likely. Dividing negative logits by the penalty would make them less negative and unintentionally more likely.

### Presence and frequency penalties

Let $c_v$ be the number of times token $v$ has appeared:

$$
z'_v=z_v-\lambda_{\text{presence}}\mathbf{1}[c_v>0]-\lambda_{\text{frequency}}c_v
$$

- Presence penalty applies once if a token has appeared.
- Frequency penalty increases with the number of occurrences.

These are inference-time heuristics and do not change model weights.

## 12. Token constraints

A generation system may adjust logits to enforce rules, such as:

- Masking EOS before a minimum generation length
- Banning selected tokens
- Forcing a prefix
- Enforcing structured output
- Blocking repeated n-grams

To ban a token:

```python
logits[:, banned_token_id] = -torch.inf
```

To prevent EOS before `min_new_tokens`:

```python
if step < min_new_tokens:
    logits[:, eos_token_id] = -torch.inf
```

These operations occur before softmax or `argmax`.

## 13. EOS handling in a batch

Different batch items may finish at different steps:

```text
finished = [False, True, False]
```

A common implementation forces finished rows to emit padding:

```python
next_token = torch.where(
    finished[:, None],
    torch.full_like(next_token, pad_token_id),
    next_token,
)
```

Update completion state:

```python
finished = finished | (
    next_token.squeeze(-1) == eos_token_id
)
```

Stop only when all batch items are finished or `max_new_tokens` is reached:

```python
if finished.all():
    break
```

A production server may remove finished sequences from the active batch to avoid unnecessary decode work.

## 14. Prefill and decode phases

KV-cached generation has two phases.

### Prefill

Process the complete prompt:

```text
input_ids: [B, T_prompt]
```

The model computes:

```text
hidden states: [B, T_prompt, d_model]
logits:        [B, T_prompt, V]
```

Only the final prompt logit is used for the first generated token:

```python
next_logits = logits[:, -1, :]  # [B, V]
```

The model also builds the K/V cache for all prompt tokens.

### Decode

Each subsequent step processes one new token:

```text
new input token:  [B, 1]
new hidden state: [B, 1, d_model]
new logits:       [B, 1, V]
```

The cache supplies keys and values for earlier positions.

## 15. KV-cache shapes

For one Transformer layer:

```text
cached K: [B, H_kv, T_past, d_head]
cached V: [B, H_kv, T_past, d_head]
```

A new token produces:

```text
new K: [B, H_kv, 1, d_head]
new V: [B, H_kv, 1, d_head]
```

After appending:

```text
updated K: [B, H_kv, T_past + 1, d_head]
updated V: [B, H_kv, T_past + 1, d_head]
```

For multi-head attention, `H_kv = H_query`. For grouped-query or multi-query attention, `H_kv < H_query`, reducing cache memory.

Across $L$ layers, cache element count is:

$$
2LBH_{\text{kv}}T_{\text{past}}d_{\text{head}}
$$

The factor two accounts for keys and values.

## 16. Why the KV cache accelerates decoding

Without caching, each step recomputes keys and values for the entire context. With caching, previous K/V tensors are reused and only the new token's K/V tensors are computed.

For a single decode step:

- Without cache, attention recomputation grows approximately quadratically with the current context.
- With cache, the new query attends over cached positions in approximately linear time with the current context.

Across a complete generated sequence, cached full-context attention remains quadratic in total sequence length because every new query attends to all previous keys. The cache removes repeated computation; it does not make exact full-context attention constant-time.

The tradeoff is memory:

$$
\text{KV memory}\propto L\times B\times H_{\text{kv}}\times T\times d_{\text{head}}
$$

## 17. Complete sampling function

```python
import torch


def sample_next_token(
    logits,
    temperature=1.0,
    top_k=None,
    top_p=None,
):
    """
    logits: [B, V]
    returns: [B, 1]
    """
    if temperature == 0:
        return torch.argmax(
            logits,
            dim=-1,
            keepdim=True,
        )

    if temperature < 0:
        raise ValueError("temperature must be non-negative")

    logits = logits / temperature

    if top_k is not None:
        k = min(top_k, logits.shape[-1])

        threshold = torch.topk(
            logits,
            k=k,
            dim=-1,
        ).values[..., -1, None]

        logits = logits.masked_fill(
            logits < threshold,
            -torch.inf,
        )

    if top_p is not None and top_p < 1.0:
        logits = top_p_filter(logits, top_p)

    probs = torch.softmax(logits, dim=-1)

    return torch.multinomial(
        probs,
        num_samples=1,
    )
```

A production implementation should validate that `top_k >= 1`, `0 < top_p <= 1`, at least one finite logit remains, and the probability tensor contains no NaNs.

## 18. Simplified cached generation loop

```python
@torch.no_grad()
def generate(
    model,
    input_ids,
    max_new_tokens,
    eos_token_id,
    pad_token_id,
    temperature=1.0,
    top_k=None,
    top_p=None,
):
    batch_size = input_ids.shape[0]
    generated = input_ids

    finished = torch.zeros(
        batch_size,
        dtype=torch.bool,
        device=input_ids.device,
    )

    # Prefill
    outputs = model(
        input_ids=input_ids,
        use_cache=True,
    )

    cache = outputs.past_key_values
    next_logits = outputs.logits[:, -1, :]

    for _ in range(max_new_tokens):
        next_token = sample_next_token(
            next_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        next_token = torch.where(
            finished[:, None],
            torch.full_like(next_token, pad_token_id),
            next_token,
        )

        generated = torch.cat(
            [generated, next_token],
            dim=1,
        )

        finished = finished | (
            next_token.squeeze(-1) == eos_token_id
        )

        if finished.all():
            break

        outputs = model(
            input_ids=next_token,
            past_key_values=cache,
            use_cache=True,
        )

        cache = outputs.past_key_values
        next_logits = outputs.logits[:, -1, :]

    return generated
```

## 19. Sampling reproducibility

Sampling can be made reproducible under controlled conditions:

```python
generator = torch.Generator(device=logits.device)
generator.manual_seed(42)

next_token = torch.multinomial(
    probs,
    num_samples=1,
    generator=generator,
)
```

Exact reproducibility can still depend on hardware, kernel selection, numerical precision, distributed execution, and framework configuration. Greedy decoding is naturally deterministic for identical logits.

## 20. Common interview mistakes

1. Applying temperature after softmax instead of to logits.
2. Computing softmax before greedy `argmax` even though it is unnecessary.
3. Assuming top-k and top-p both retain a fixed number of tokens.
4. Removing the top-p boundary token that crosses the threshold.
5. Applying a naïve repetition penalty that makes negative logits less negative.
6. Recomputing the full prompt at every cached decode step.
7. Ignoring per-sequence EOS state in a batch.
8. Claiming the KV cache makes total full-context generation linear.
9. Sampling from raw logits without forming probabilities.
10. Allowing every logit to become `-inf`, producing invalid softmax output.

## 21. Interview check and reviewed answers

Assume:

```text
B        = 2
V        = 50,000
T_prompt = 128
L        = 32
H_kv     = 8
d_head   = 64
```

### 1. Last-position logits after prefill

All prefill logits have shape:

```text
[B, T_prompt, V] = [2, 128, 50000]
```

Selecting the final position gives:

```text
[B, V] = [2, 50000]
```

The original answer correctly identified the vocabulary dimension but omitted the batch dimension.

### 2. Does greedy decoding require softmax?

No. Greedy decoding selects the largest logit, and softmax preserves the ordering of logits. Correct.

### 3. Temperature behavior

- `temperature = 0.5` sharpens the distribution and makes sampling more deterministic.
- `temperature = 2.0` flattens the distribution and increases diversity.

Correct.

### 4. Top-k candidate count

With `top_k = 50`, exactly 50 vocabulary tokens remain eligible before sampling. Correct.

### 5. Top-p candidate count

Top-p does not retain a fixed number of tokens. It keeps sorted tokens until their cumulative probability reaches the threshold. Correct.

### 6. Why shift the top-p mask?

The shift keeps the boundary token that causes cumulative probability to reach or exceed $p$. Otherwise, the retained set could have total probability below the requested threshold. Correct.

### 7. Per-layer cache after prompt prefill

Each cache has shape:

```text
K: [2, 8, 128, 64]
V: [2, 8, 128, 64]
```

Correct.

### 8. Cache after one generated token

The new token contributes:

```text
new K: [2, 8, 1, 64]
new V: [2, 8, 1, 64]
```

After appending:

```text
K: [2, 8, 129, 64]
V: [2, 8, 129, 64]
```

Only the sequence-length dimension grows. The original answer incorrectly kept it at 128.

### 9. One sequence reaches EOS

The entire loop should not stop. Mark the completed sequence as finished and continue for active sequences. Finished rows can emit padding or be removed from the active batch. Stop when all sequences finish or `max_new_tokens` is reached.

The original answer correctly said not to stop the full batch but needed to state the per-sequence behavior.

### 10. Sampling operation order

```text
raw logits
→ repetition/token constraints
→ temperature scaling
→ top-k/top-p filtering
→ softmax
→ multinomial sampling
```

Correct.

## 22. Interview-ready summary

Autoregressive decoding converts `[B,V]` next-token logits into `[B,1]` token IDs one step at a time. Greedy decoding uses `argmax` without requiring softmax. Temperature reshapes the distribution, top-k retains a fixed number of candidates, and top-p retains a variable nucleus that reaches a probability threshold. Repetition and token constraints modify logits before normalization. KV-cached decoding separates prompt prefill from single-token decode steps and grows each per-layer cache from `[B,H_kv,T,d_head]` to `[B,H_kv,T+1,d_head]` after every token.

## Key takeaway

The trained model supplies logits; the decoding policy determines the generated behavior. Sampling quality depends on the order and correctness of logit transformations, while generation efficiency depends on reusing past keys and values and managing EOS independently for every sequence in the batch.
