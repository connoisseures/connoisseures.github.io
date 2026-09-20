---
layout: post
title: "LLM & NLP Interview Preparation — Day 04: Questions 16–20"
date: 2026-09-20
math: true
categories: [LLM, Interview Preparation, Questions]
tags: [prompt-engineering, pipeline-parallelism, mixed-precision, sparse-attention, llm-serving]
description: "Day 04 study notes covering prompt engineering, pipeline bubbles, mixed-precision training, sparse attention, and scalable LLM inference-service architecture."
---

# LLM & NLP Interview Preparation — Day 4

**Questions 16–20**  
Source: [MLE Interview Question Bank — LLM & NLP](https://chengz.art/zh/mle-prep)

## Progress Summary

| Question | Topic | Status |
|---|---|---|
| 16 | Improving output quality with prompt engineering | Complete |
| 17 | Pipeline bubbles during inference | Complete |
| 18 | Mixed-precision training | Complete |
| 19 | Sparse attention in Transformers | Complete |
| 20 | Scalable LLM inference-service architecture | Complete |

---

## Question 16 — Improving Output Quality with Prompt Engineering

### Interview question

How can prompt engineering improve an LLM's output quality? Explain prompt structure, zero-shot and few-shot prompting, explicit constraints, output schemas, evaluation, and common failure modes.

### Review

Few-shot prompting is an important technique. Representative examples communicate the task, expected format, label meanings, style, and decision boundaries. A complete answer should also cover clear instructions, relevant context, constraints, schemas, uncertainty behavior, and systematic evaluation.

### Prompt structure

A well-structured prompt commonly contains:

1. **Role or context:** the perspective or operating context.
2. **Task instruction:** a precise statement of what to do.
3. **Relevant data:** the information required to answer.
4. **Constraints:** required and prohibited behavior.
5. **Examples:** representative demonstrations when useful.
6. **Output schema:** the expected response structure.
7. **Uncertainty policy:** what to do when information is missing.

Explicit schemas make outputs easier to validate and integrate into downstream systems. They reduce ambiguity but do not eliminate the need for programmatic validation.

### Zero-shot vs. few-shot prompting

**Zero-shot prompting** provides instructions without demonstrations. It is concise and inexpensive but depends more heavily on the model understanding the task from the instructions alone.

**Few-shot prompting** includes representative input-output demonstrations. It can clarify labels, decision boundaries, tone, style, and formatting.

Too many examples can:

- Consume the context window.
- Increase prefill cost and time to first token.
- Bias the model toward unrepresentative patterns.
- Introduce contradictions.
- Cause the model to copy accidental mistakes.

Examples should be correct, concise, diverse, and representative.

### Evaluation

A prompt that succeeds on a few hand-picked examples has not been shown to generalize. A stronger evaluation process is:

1. Create a held-out set representing normal traffic, rare cases, adversarial inputs, languages, and important user segments.
2. Define task-specific quality metrics before comparing prompts.
3. Run the old and new prompts on the same examples with controlled model and decoding settings.
4. Use deterministic checks where possible, human evaluation for subjective quality, and model-based graders only after validating them against human labels.
5. Compare task success, format-validity rate, safety violations, hallucination or groundedness, latency, and token cost.
6. Compute confidence intervals or paired statistical tests rather than relying only on averages.
7. Shadow-test on production traffic, followed by a staged rollout or A/B test with rollback criteria.

### Common failure modes

- Ambiguous or contradictory instructions.
- Incorrect, biased, or unrepresentative examples.
- Prompt injection from retrieved or user-provided content.
- Overfitting to the evaluation examples.
- Position and ordering effects.
- Requesting a schema without validating the response.
- Higher latency and cost from excessive prompt length.
- Improving average quality while harming safety or important subgroups.

### Interview-ready answer

Prompt engineering improves output quality by reducing ambiguity and showing the model the desired task, constraints, examples, and output format. I would structure the prompt with a clear role, precise instruction, relevant context, constraints, representative demonstrations, a machine-checkable schema, and an uncertainty policy.

Zero-shot prompting uses instructions alone, whereas few-shot prompting supplies demonstrations that clarify behavior and formatting. I would keep examples minimal and representative because excessive examples increase cost and can bias the model.

I would compare a revised prompt against the current prompt on a held-out, representative evaluation set using task quality, format validity, safety, latency, and cost metrics. After offline evaluation, I would use shadow testing and a staged A/B rollout.

### Key takeaway

- Prompt engineering reduces ambiguity but does not change model weights.
- Few-shot examples communicate behavior and format.
- Explicit schemas still require downstream validation.
- Prompt changes need representative, paired evaluation.
- Quality, safety, latency, and cost must be evaluated together.

---

## Question 17 — Pipeline Bubbles During Inference

### Interview question

What is a pipeline bubble in model inference, why do stages become idle, and how can its impact be reduced?

### Review

A pipeline bubble can be described as waiting for batches. More precisely, it is a period when one or more pipeline stages are idle because a required microbatch, activation, or token has not arrived.

### Sources of bubbles

Suppose a model is divided into four stages:

- GPU 1: layers 1–8
- GPU 2: layers 9–16
- GPU 3: layers 17–24
- GPU 4: layers 25–32

When the first microbatch starts, only GPU 1 can work. The remaining GPUs wait for input activations. This is the **pipeline-fill bubble**.

When the last microbatch moves through the pipeline, earlier stages finish first and become idle while later stages continue. This is the **pipeline-drain bubble**.

Idle time can also arise from:

- Uneven computation across stages.
- Communication delays between devices.
- Different microbatch or sequence lengths.
- A slow or overloaded device.
- Autoregressive dependencies between decoding steps.
- Dynamic batching and requests entering or leaving at different times.

### Microbatching

Microbatching lets different pipeline stages work concurrently. For example:

- Stage 1 processes microbatch 3.
- Stage 2 processes microbatch 2.
- Stage 3 processes microbatch 1.

Once the pipeline is filled, all stages can remain active. For a simplified pipeline with \(p\) stages and \(m\) microbatches:

$$
\text{bubble fraction}
\approx
\frac{p-1}{m+p-1}
$$

Increasing \(m\) reduces the fixed fill-and-drain overhead as a fraction of useful work.

However, too many microbatches can create smaller matrix operations, reduce GPU efficiency, increase kernel-launch and scheduling overhead, increase communication events, and complicate activation management.

### Stage imbalance

Adding microbatches does not fix a stage that consistently takes twice as long as the others. The slowest stage determines pipeline throughput and forces faster stages to wait.

Possible corrections include:

- Repartition layers using measured execution time instead of equal layer counts.
- Move expensive layers from the slow stage to faster stages.
- Use virtual or interleaved pipeline stages.
- Place the bottleneck stage on faster hardware.
- Replicate a stage when the serving architecture supports stage-level parallelism.
- Optimize the slow stage's kernels, communication, or memory access.
- Bucket requests by length to reduce stragglers.

### Interview-ready answer

A pipeline bubble is idle device time caused by pipeline fill, drain, stage imbalance, or delayed communication. Splitting work into more microbatches allows stages to process different microbatches concurrently and reduces the relative fill-and-drain overhead.

Microbatching cannot remove a persistent bottleneck stage. I would profile stage execution and communication times, repartition layers based on measured costs, use interleaved scheduling where appropriate, and place communication-heavy boundaries on faster links. The objective is to balance stage time while keeping microbatches large enough for efficient GPU kernels.

### Key takeaway

- A bubble is stage idle time, not merely an empty batch.
- More microbatches reduce fill-and-drain overhead.
- Excessively small microbatches reduce kernel efficiency.
- The slowest stage determines throughput.
- Profile and rebalance stages to address persistent bubbles.

---

## Question 18 — Mixed-Precision Training

### Interview question

What are the advantages and challenges of mixed-precision training? Why use FP16 or BF16 instead of FP32?

### Advantages

Using 16-bit formats for most eligible computation:

- Reduces activation, gradient, and eligible parameter memory.
- Reduces memory-bandwidth traffic.
- Allows larger batches or longer sequences.
- Accelerates matrix operations on GPU tensor cores.
- Can reduce distributed communication volume.

Mixed precision keeps FP32 where range, precision, or accumulation stability matters.

### Numerical challenges

FP16 has a much smaller representable range than FP32. Small gradients can **underflow to zero**, while large values can **overflow to infinity**.

Common safeguards include:

- **Loss scaling:** multiply the loss before backpropagation, then unscale gradients before clipping and stepping.
- **FP32 master weights:** update high-precision parameter copies.
- **FP32 accumulation:** use higher precision for reductions or optimizer states.
- **Selective precision:** keep sensitive normalization, reduction, or softmax operations in FP32 when needed.
- **Overflow detection:** skip unsafe steps and adjust the dynamic scale.

### FP16 vs. BF16

| Format | Exponent bits | Mantissa bits | Main property |
|---|---:|---:|---|
| FP16 | 5 | 10 | More significand precision but narrow range |
| BF16 | 8 | 7 | FP32-like range but less significand precision |
| FP32 | 8 | 23 | Wide range and high precision |

BF16 needs loss scaling less often because it has eight exponent bits, the same as FP32. It is therefore less likely than FP16 to overflow or underflow. Its reduced mantissa precision introduces more rounding, but training usually tolerates rounding better than FP16's limited numeric range.

### Correct FP16 training order

1. Zero gradients.
2. Run the forward pass and compute the loss.
3. Scale the loss.
4. Backpropagate the scaled loss.
5. Unscale gradients.
6. Check for overflow or non-finite gradients.
7. Clip the **unscaled** gradients.
8. Perform the optimizer step only if gradients are finite.
9. Update the dynamic loss scale.
10. Advance the learning-rate scheduler according to its defined step semantics.

Clipping scaled gradients is incorrect because the clipping threshold would be applied to artificially enlarged values.

### Interview-ready answer

Mixed-precision training improves throughput and memory efficiency by using FP16 or BF16 for tensor-core operations while retaining FP32 for numerically sensitive computations, accumulation, optimizer states, or master weights.

The primary challenge is numerical stability. FP16's narrow exponent range can cause gradient underflow and activation overflow, so it commonly requires dynamic loss scaling, FP32 accumulation, non-finite checks, and careful operation ordering. BF16 has an FP32-like exponent range and usually does not need loss scaling, although FP32 accumulation and stability-sensitive operations remain useful.

I would monitor non-finite losses and gradients, skipped optimizer steps, loss-scale behavior, convergence against an FP32 or validated baseline, and final evaluation quality.

### Key takeaway

- Lower precision improves speed, memory usage, and bandwidth efficiency.
- FP16 commonly requires dynamic loss scaling.
- BF16 has a much wider range but fewer mantissa bits.
- Unscale before checking and clipping gradients.
- Validate convergence and final quality, not only throughput.

---

## Question 19 — Sparse Attention in Transformers

### Interview question

How can sparse attention be used in Transformers, and why is it useful for long sequences?

### Standard-attention cost

Standard self-attention compares every query token with every key token:

$$
QK^T \in \mathbb{R}^{n \times n}
$$

For sequence length \(n\) and head dimension \(d_h\), the score computation is approximately:

$$
O(n^2 d_h)
$$

The dense score matrix also requires \(O(n^2)\) memory when materialized. Doubling sequence length creates roughly four times as many query-key pairs.

### Sparse patterns

Sparse attention allows each query to attend to only selected keys:

- **Sliding-window attention:** attend to a local window of \(w\) tokens.
- **Global tokens:** selected tokens communicate with all positions.
- **Block-sparse attention:** compute only selected block-to-block interactions.
- **Dilated or strided attention:** use spaced connections for broader coverage.
- **Random connections:** supplement local patterns with long-range links.
- **Content-based routing:** dynamically select relevant blocks or keys.

For a fixed sliding window:

$$
O(n w d_h)
$$

If \(w\) is constant and much smaller than \(n\), this is approximately linear in \(n\). For \(w=512\):

$$
O(512 n d_h) = O(n d_h)
$$

### Trade-offs

Sparse patterns reduce compute and memory but remove direct query-key connections. The model may miss dependencies outside the allowed pattern. Hybrid designs combine local windows with global tokens, dilated links, or routed long-range connections.

Information can propagate beyond one window across multiple Transformer layers, but this is less direct than full attention.

### Implementation requirement

Creating the full dense \(n \times n\) score matrix and then masking most entries does **not** provide the full sparse-attention speedup. The dense matrix multiplication and allocation have already paid the quadratic cost.

True savings require sparse or block-sparse kernels that compute only allowed query-key blocks. The sparsity pattern must also align with hardware-friendly tiles; arbitrary sparsity may have poor utilization, and its overhead can outweigh the saved FLOPs.

### Interview-ready answer

Full attention has quadratic query-key interactions and becomes expensive for long sequences. Sparse attention reduces this by restricting each query to a structured subset of keys, such as a local window, selected global tokens, or allowed blocks.

A fixed window of width \(w\) changes the attention computation from \(O(n^2 d_h)\) to \(O(n w d_h)\), which is approximately linear in sequence length when \(w\) is fixed. The trade-off is reduced access to distant information, so I would combine local attention with a small number of global or long-range connections.

The implementation must use an actual sparse or block-sparse kernel. Applying a mask after dense score computation preserves model semantics but not the main computational savings.

### Key takeaway

- Full attention is quadratic in sequence length.
- A fixed window makes attention approximately linear in sequence length.
- Sparse patterns trade global connectivity for efficiency.
- Hybrid local/global patterns preserve important long-range communication.
- Dense computation followed by masking is not true sparse acceleration.

---

## Question 20 — Scalable LLM Inference-Service Architecture

### Interview question

How would you design a scalable LLM inference-service architecture?

### Core architecture

1. **API gateway:** authentication, request validation, quotas, and rate limiting.
2. **Router and admission control:** choose the model version, region, priority class, and worker pool; reject or defer work when token or memory capacity is exhausted.
3. **Inference scheduler:** form continuous batches and schedule by token and KV-cache budgets rather than request count alone.
4. **GPU workers:** execute prefill and autoregressive decoding, manage the KV cache, and stream generated tokens.
5. **Streaming layer:** deliver tokens while handling cancellation, timeouts, reconnects, and backpressure.

### Scaling strategy

- Replicate workers horizontally when a model fits on one accelerator.
- Use tensor parallelism, pipeline parallelism, or both when one model spans multiple accelerators.
- Autoscale using queue delay, token throughput, KV-cache pressure, and latency—not only requests per second.
- Route requests by model, prompt length, modality, tenant priority, and cache locality.
- Separate prefill and decode pools at high scale because prefill is compute-heavy, while decode is often memory-bandwidth- and KV-cache-bound.
- Use PagedAttention or another block-based KV-cache allocator to reduce fragmentation and improve batching.
- Apply quantization where its quality/latency trade-off is acceptable.

### Why request count is a poor capacity unit

Two requests can differ by orders of magnitude:

- Long prompts require much more prefill compute.
- Long generations require many more serial decode steps.
- Longer sequences consume more KV-cache memory.
- Output length is initially uncertain.

The scheduler should therefore enforce token, memory, and concurrency budgets, such as queued input tokens, active decode tokens, and KV-cache blocks.

### Reliability and worker-crash recovery

- Mark a failed worker unhealthy, remove it from routing, and reclaim its scheduler and KV-cache allocations.
- Retry only eligible and idempotent requests on a healthy replica.
- Reconstruct the request from the original prompt plus the generated token prefix, then rebuild the KV cache by running prefill again.
- Give each stream a request ID and token offset so the client can reconnect without receiving duplicate tokens.
- Preserve the generated prefix and, when exact stochastic continuation matters, the sampling seed, RNG state, and generation parameters.
- The difficult state is the worker-local KV cache: it is large and disappears with the worker. Continuous replication or checkpointing is expensive, so many systems recompute it or fail and retry the request.
- Exact continuation is not always guaranteed after replay under nondeterministic kernels or when sampling state was not retained.

### Observability

Track:

- Time to first token (TTFT).
- Time per output token or inter-token latency.
- End-to-end latency.
- Queue delay.
- Input and output token throughput.
- Batch utilization.
- KV-cache occupancy and eviction.
- OOM and retry rates.
- Accelerator utilization and cost per token.

### Interview-ready answer

I would place an authenticated, rate-limited API gateway in front of a model-aware router and admission controller. A central or distributed scheduler would use continuous batching and token/KV-memory budgets to dispatch work to replicated GPU worker groups. Small models can be replicated per GPU; large models can use tensor or pipeline parallel groups, with horizontal scaling across groups.

At higher scale, I would consider disaggregating prefill and decode, use block-based KV-cache management, stream tokens with backpressure, and autoscale from queue, latency, throughput, and memory signals. For failures, I would remove unhealthy workers and replay eligible requests on healthy replicas from the prompt and already-generated prefix, while using request IDs and offsets to prevent duplicate streamed tokens.

I would optimize against service-level objectives such as TTFT, inter-token latency, throughput, availability, and cost per token.

### Key takeaway

A scalable service coordinates routing, admission control, token-aware scheduling, efficient KV-cache management, parallel GPU execution, streaming, autoscaling, and failure recovery.

---

## Day 4 Final Review

| Topic | Core idea | Main trade-off or risk |
|---|---|---|
| Prompt engineering | Reduce ambiguity with instructions, context, examples, constraints, and schemas | Longer prompts increase latency/cost and may introduce bias or conflicts |
| Pipeline bubbles | Idle time comes from fill, drain, communication, and stage imbalance | More microbatches reduce bubbles but can lower kernel efficiency |
| Mixed precision | Use lower precision for speed and memory efficiency while preserving FP32 where needed | FP16 can underflow or overflow; ordering and loss scaling matter |
| Sparse attention | Restrict query-key connections to reduce long-context cost | Long-range information may be lost, and real speedups require sparse kernels |
| Scalable inference | Combine routing, token-aware scheduling, KV-cache management, streaming, and fault recovery | Must balance latency, throughput, reliability, quality, and cost |

### Compact recall

1. **Prompt engineering:** clear instructions plus representative examples and measurable evaluation.
2. **Pipeline bubbles:** microbatching helps fill/drain overhead; balancing fixes persistent bottlenecks.
3. **Mixed precision:** FP16 usually needs loss scaling; BF16 offers FP32-like range.
4. **Sparse attention:** a fixed local window changes quadratic attention toward linear scaling.
5. **Inference architecture:** schedule by tokens and KV memory, not request count alone.
