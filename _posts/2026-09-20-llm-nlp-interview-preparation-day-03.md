---
layout: post
title: "LLM & NLP Interview Preparation — Day 03: Questions 11–15"
date: 2026-09-20
math: true
categories: [LLM, Interview Preparation, Questions]
tags: [memory-bandwidth, multi-task-learning, tensor-parallelism, pipeline-parallelism, llm-serving, llm-safety]
description: "Day 03 study notes covering inference memory bandwidth, multi-task training, tensor and pipeline parallelism, low-latency serving, and LLM safety filtering."
---

# LLM & NLP Interview Preparation — Day 3

**Questions 11–15**  
Source: [MLE Interview Question Bank — LLM & NLP](https://chengz.art/zh/mle-prep)

## Progress Summary

| Question | Topic | Status |
|---|---|---|
| 11 | Inference memory-bandwidth bottlenecks | Complete |
| 12 | Multi-task LLM training pipeline | Complete |
| 13 | Tensor vs. pipeline parallelism | Complete |
| 14 | Low-latency online inference | Complete |
| 15 | LLM safety-filtering system | Complete |

---

## Question 11 — Inference Memory-Bandwidth Bottlenecks

### Interview question

Why is autoregressive LLM decoding often memory-bandwidth-bound rather than compute-bound, especially at small batch sizes? How do batching, quantization, and KV-cache optimization help?

### Key correction

HBM is GPU memory. Data moves from HBM into on-chip cache, SRAM, and registers before computation.

- **Prefill** processes many prompt tokens in parallel using large matrix multiplications. It has relatively high arithmetic intensity and is often compute-bound.
- **Decode** processes one new token per sequence per step. At small batch sizes, large weight matrices are loaded from HBM for relatively little computation, so decode is often memory-bandwidth-bound.

### Interview-ready answer

During autoregressive decoding, the model generates one token at a time. At every step, it reads most model weights from HBM and reads the existing KV cache for attention. At a small batch size, each loaded weight supports little computation, so the GPU finishes arithmetic faster than HBM can deliver data. Tensor cores remain underutilized and memory bandwidth limits performance.

Increasing the batch size allows the same loaded weights to serve multiple sequences. Weight traffic does not grow proportionally with the batch size, but FLOPs do, so arithmetic intensity and GPU utilization improve.

$
\text{Arithmetic intensity}
=
\frac{\text{FLOPs}}{\text{bytes transferred}}
$

For memory-bound decoding:

$
\text{latency per token}
\approx
\frac{\text{bytes read per decoding step}}
{\text{HBM bandwidth}}
$

For a batched linear layer:

$
Y_{B\times d_{out}}=X_{B\times d_{in}}W_{d_{in}\times d_{out}}
$

The same weight matrix is reused across the $B$ sequences. Quantization reduces parameter bytes transferred, while MQA, GQA, KV-cache quantization, and paged memory management reduce KV-cache capacity or bandwidth costs.

### Takeaways

- Prefill is usually more compute-intensive; decode is often bandwidth-bound.
- Larger batches improve weight reuse.
- Quantization reduces memory traffic.
- KV-cache traffic still grows with batch size and context length.

---

## Question 12 — Multi-Task LLM Training Pipeline

### Interview question

How would you design a training pipeline for a large language model that supports multiple tasks?

### Review

Data parallelism and model parallelism are useful scaling mechanisms, but they do not define the multi-task learning strategy. A complete answer must also cover task formatting, dataset mixing, sampling, loss balancing, conflicting gradients, and per-task evaluation.

### Interview-ready answer

I would convert all tasks into a unified instruction-response format so the model can use a shared autoregressive language-modeling objective. I would maintain separate training and evaluation datasets for each task, then combine them using a configurable data mixer.

Sampling directly in proportion to dataset size can cause large datasets to dominate. Temperature-based task sampling provides control:

$
p_i=\frac{n_i^\alpha}{\sum_j n_j^\alpha}
$

where $n_i$ is the size of task $i$:

- $\alpha=1$: sampling is proportional to dataset size.
- $\alpha=0$: tasks are sampled uniformly.
- $0<\alpha<1$: compromise between proportional and uniform sampling.

The total objective may be:

$
\mathcal{L}=\sum_i\lambda_i\mathcal{L}_i
$

The weights $\lambda_i$ can reflect task importance, loss scale, learning progress, or gradient magnitude. For conflicting tasks, possible approaches include PCGrad, dynamic loss weighting, balanced sampling, or task-specific adapters or heads.

For scale, I would combine data parallelism with tensor or pipeline parallelism based on model size and cluster topology. I would monitor validation metrics separately for every task and rebalance the mixture if one task improves while another regresses.

### Imbalance example

If Task A contains 100 million examples and Task B contains 100,000, combined-example sampling gives:

$
P(A)\approx99.9\%,\qquad P$B$\approx0.1\%
$

Task B receives almost no training signal. Corrections include temperature sampling, task-uniform sampling, weighted losses, minimum task quotas, or controlled oversampling.

### Takeaways

- Standardize tasks into a shared format.
- Control the sampling mixture.
- Balance losses and gradient conflicts.
- Scale execution with distributed parallelism.
- Evaluate every task independently.

---

## Question 13 — Tensor Parallelism vs. Pipeline Parallelism

### Interview question

Explain the difference between tensor parallelism and pipeline parallelism, including communication patterns, bottlenecks, and when they should be combined.

### Tensor parallelism

Tensor parallelism partitions tensors and computation **within a layer**. For:

$
Y=XW
$

the weights can be partitioned:

$
W=[W_1\;W_2],\qquad Y_1=XW_1,\quad Y_2=XW_2
$

Multiple GPUs jointly compute the same layer and combine partial results using all-reduce, all-gather, or reduce-scatter. Communication occurs multiple times per Transformer block, so tensor parallelism works best over high-bandwidth links such as NVLink or NVSwitch.

### Pipeline parallelism

Pipeline parallelism partitions the model **by depth**:

- GPU 1: layers 1–8
- GPU 2: layers 9–16
- GPU 3: layers 17–24
- GPU 4: layers 25–32

Microbatches flow through these stages. Adjacent stages exchange activations during the forward pass and activation gradients during the backward pass.

### Pipeline bubbles

A pipeline bubble is GPU idle time during pipeline fill and drain, or when a faster stage waits for a slower stage. For a simplified forward pipeline with $p$ stages and $m$ microbatches:

$
\text{bubble fraction}\approx\frac{p-1}{m+p-1}
$

Increasing the number of microbatches reduces the relative fill-and-drain overhead, although too many microbatches can add scheduling overhead or reduce kernel efficiency.

### Interview-ready answer

Tensor parallelism divides matrix operations within each layer, so multiple GPUs jointly compute the same Transformer layer. It requires frequent collective communication but does not inherently create pipeline bubbles.

Pipeline parallelism assigns groups of layers to different stages. It communicates activations and gradients between adjacent stages, but suffers from pipeline bubbles and stage imbalance.

I would usually use tensor parallelism within a node with fast interconnects. I would add pipeline parallelism across tensor-parallel groups when the model cannot fit within one group. Large systems commonly combine tensor, pipeline, and data parallelism as 3D parallelism.

| Method | Partition | Communication | Main bottleneck |
|---|---|---|---|
| Tensor parallelism | Operations within a layer | Frequent collectives | Interconnect bandwidth and latency |
| Pipeline parallelism | Groups of layers | Activations and gradients between stages | Bubbles and stage imbalance |

---

## Question 14 — Low-Latency Online LLM Inference

### Interview question

How would you design a low-latency online LLM inference service?

### Core metrics

- **TTFT:** time from request arrival to the first generated token.
- **TPOT:** average decoding time per output token.
- **ITL:** observed delay between streamed tokens.
- **End-to-end latency:** request arrival until generation finishes.
- **Throughput:** aggregate tokens generated per second.
- **Tail latency:** p95 or p99 latency.

For a long prompt:

$
\text{TTFT}
\approx
\text{queueing}
+
\text{prefill}
+
\text{first decode step}
$

Long prompts most directly increase TTFT because the service must process the prompt and construct its KV cache before generating the first token. They can also increase TPOT because each generated token attends over a larger cache.

### Request flow

1. API gateway authenticates, validates, and rate-limits requests.
2. Admission control checks queue, token, and capacity budgets.
3. The scheduler routes based on model, lengths, priority, and KV-cache capacity.
4. Prefill workers process prompts and build KV caches.
5. Decode workers generate tokens autoregressively.
6. The service streams tokens immediately.
7. Metrics, traces, and outcomes are recorded.

### Continuous batching

Static batching holds a fixed batch until all requests finish. Short requests leave wasted slots while long requests continue.

Continuous batching updates the batch at token boundaries:

1. Finished requests leave immediately.
2. New requests fill released slots.
3. Active requests keep decoding.

This improves utilization and throughput for heterogeneous request lengths. The scheduler must still limit queue delay, batch size, token budget, and KV-cache consumption.

### Optimizations

- Continuous batching.
- PagedAttention for efficient KV allocation.
- Prefix caching for repeated prompts.
- Quantization, MQA, and GQA.
- Optimized kernels, FlashAttention, CUDA graphs, and kernel fusion.
- Immediate token streaming.
- Optional prefill/decode disaggregation.
- Length-, priority-, and memory-aware routing.

### Overload protection

Do not admit requests indefinitely when traffic exceeds capacity. Use:

- Bounded queues.
- Concurrency and token-budget limits.
- Backpressure or retryable HTTP 429 responses.
- Priority scheduling.
- Deadlines and cancellation.
- Autoscaling.
- Controlled load shedding.

These mechanisms protect p95 and p99 latency instead of allowing queue time to grow until requests time out.

---

## Question 15 — LLM Safety-Filtering System

### Interview question

How would you design a system that prevents an LLM service from accepting harmful inputs or returning harmful outputs?

### Layered architecture

Use defense in depth:

1. Input guardrails.
2. A centralized policy engine.
3. Constrained model and tool execution.
4. Output guardrails.
5. Monitoring, red teaming, and continuous improvement.

### Input guardrails

- Authenticate and rate-limit users.
- Detect harmful content, jailbreaks, and prompt injection.
- Detect or redact PII, secrets, and credentials.
- Classify risk category, severity, and confidence.
- Allow, constrain, transform, refuse, or escalate.

### Policy engine

The policy engine maps signals to actions:

- **Allow:** proceed normally.
- **Allow with constraints:** restrict tools or use a safer configuration.
- **Transform:** redact or safely rewrite.
- **Refuse:** return a compliant refusal.
- **Escalate:** route ambiguous high-risk cases to review.

Policies should be versioned, auditable, and configurable by product, jurisdiction, age, and risk tolerance.

### Model and tool safeguards

- Apply strong system instructions and output schemas.
- Give tools least-privilege permissions.
- Validate tool arguments.
- Require confirmation for consequential actions.
- Treat retrieved content and tool results as untrusted.
- Limit iterations, tokens, and spending.

### Output guardrails

Input filtering cannot guarantee safe output. The model is probabilistic and may see retrieved documents, tool results, or later conversational context after the input check.

Before release:

- Classify generated content.
- Detect PII or secret leakage.
- Check grounding when sources are required.
- Validate structured outputs.
- Allow, redact, regenerate, refuse, or escalate.

For streaming, use incremental moderation or a short buffer so unsafe tokens are not released before checking.

### Evaluation

Track both safety and utility:

- Unsafe escape or false-negative rate.
- False-positive or over-refusal rate.
- Precision and recall by risk category.
- Jailbreak success rate.
- Performance by language and user segment.
- Appeals and human-review outcomes.

To reduce over-refusal without materially increasing unsafe output:

1. Build labeled unsafe and benign edge-case evaluation sets.
2. Tune precision-recall thresholds per category.
3. Calibrate classifier confidence.
4. Route ambiguous cases to a stronger classifier or human reviewer.
5. Add hard negative examples.
6. Shadow-test policy changes.
7. Use staged rollout and rollback criteria.

### Interview-ready answer

I would use layered input moderation, a versioned policy engine, constrained model and tool execution, output moderation, and continuous monitoring. Each layer would emit structured risk categories and confidence scores, and the policy engine would decide whether to allow, constrain, transform, refuse, or escalate.

Both input and output filtering are necessary because responses depend on probabilistic generation, multi-turn context, retrieval, and tools. I would measure unsafe escapes and over-refusals together, calibrate category-specific thresholds, and validate changes through red teaming, shadow testing, and staged rollout.

---

## Day 3 Final Review

1. **Decode bottleneck:** small-batch autoregressive decoding often waits on HBM bandwidth.
2. **Multi-task training:** unify formats, control sampling, balance losses, and evaluate per task.
3. **Model parallelism:** tensor parallelism splits layers internally; pipeline parallelism splits by depth.
4. **Online inference:** optimize TTFT and TPOT separately, use continuous batching, and protect tail latency.
5. **Safety:** combine input, execution, and output safeguards while measuring both unsafe escapes and over-refusals.
