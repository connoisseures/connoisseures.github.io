---
layout: post
title: "LLM & NLP Interview Preparation — Day 05: Questions 21–25"
date: 2026-09-20
math: true
toc: true
categories: [LLM, Interview Preparation, Questions]
tags: [attention, inference-memory, gradient-clipping, long-context, pipeline-parallelism]
description: "Day 05 study notes covering attention heads, inference memory, gradient clipping, long-context efficiency, and inference pipeline parallelism."
---

# LLM & NLP Interview Preparation — Day 5

Questions 21–25  
Study date: 2026-09-20  
Question-bank reference: [MLE Interview Question Bank — LLM & NLP](https://chengz.art/zh/mle-prep)

## Progress summary

All five questions have been explained. Completion means covered, not independently mastered; revisit them through active recall.

- Q21: Attention heads and multi-head attention — covered.
- Q22: Reducing LLM inference memory — covered.
- Q23: Gradient clipping — covered.
- Q24: Long-sequence Transformer efficiency — covered.
- Q25: Pipeline parallelism during inference — covered.

## Question 21 — Attention heads and multi-head attention

### Interview question

What is an attention head, and what advantages does multi-head attention provide? Must different heads learn different patterns?

### Explanation

An attention head is one scaled dot-product attention operation with its own learned query, key, and value projections:

$$
Q_i=XW_i^Q,\qquad K_i=XW_i^K,\qquad V_i=XW_i^V
$$

$$
\operatorname{head}_i=
\operatorname{softmax}\left(\frac{Q_iK_i^\top}{\sqrt{d_h}}+M\right)V_i
$$

The mask M contains zero for allowed connections and negative infinity for disallowed connections. Softmax operates over key positions.

Queries describe what a token is seeking, keys describe matching information, and values carry the information aggregated by attention weights.

For batch size B, sequence length T, model width D, and H heads with head width d_h = D/H:

- X: [B, T, D].
- Each Q, K, V head: [B, T, d_h].
- Each attention-score matrix: [B, T, T].
- Each head output: [B, T, d_h].
- Concatenated output: [B, T, D].

$$
\operatorname{MHA}(X)=
\operatorname{Concat}(\operatorname{head}_1,\ldots,\operatorname{head}_H)W^O
$$

Separate projections let heads attend using different representation subspaces and different attention distributions. Heads can capture local context, long-range dependencies, syntax, coreference, or other feature combinations simultaneously. These are possible learned behaviors, not guaranteed assignments.

### Gaps and pitfalls

- A head is not assigned a particular linguistic role by default.
- Heads can learn redundant patterns.
- With fixed total model width, increasing head count typically reduces width per head; it does not multiply all projection computation by the head count.
- Attention weights alone are not a complete causal explanation of a model's output.

### Interview-ready answer

An attention head computes scaled dot-product attention using independent learned query, key, and value projections. It scores query–key compatibility, normalizes scores with softmax, and aggregates values. Multi-head attention runs several such operations in different learned subspaces, concatenates their outputs, and applies an output projection. This lets the model represent multiple relationships simultaneously. Heads can specialize, but specialization and nonredundancy are not guaranteed.

## Question 22 — Reducing memory during LLM inference

### Interview question

What techniques can reduce memory consumption during LLM inference?

### Explanation: start with the memory sources

Inference memory includes model weights, the KV cache, temporary activations, kernel workspaces, and allocator overhead. Ordinary inference does not need optimizer state or backward activations.

#### Model weights

For P parameters stored with b bytes each:

$$
M_{\text{weights}}\approx Pb
$$

A 7B-parameter model uses roughly 14 GB in FP16 versus an idealized 3.5 GB in INT4. Quantization metadata, unquantized components, and runtime overhead increase actual usage.

Options:

- Quantize weights, subject to hardware support and quality validation.
- Shard weights with tensor or pipeline parallelism; this lowers per-device memory, not necessarily aggregate memory.
- Offload weights to CPU or NVMe, accepting transfer latency.
- Use a smaller or distilled model.

#### KV cache

For B equal-length sequences, L layers, cached length T, H_KV KV heads, head width d_h, and b bytes per cache element:

$$
M_{\text{KV}}\approx 2BLTH_{\text{KV}}d_hb
$$

The factor 2 accounts for keys and values. For variable lengths, replace BT by the sum of cached sequence lengths. The expression excludes padding, quantization metadata, allocation overhead, and replicated cache copies.

Options:

- MQA/GQA: fewer KV heads than query heads. This is normally an architectural choice, not an arbitrary runtime switch for an existing MHA checkpoint.
- KV-cache quantization: lower bytes per cache element, with quality and kernel trade-offs.
- PagedAttention: allocate cache in blocks to reduce fragmentation and wasted reservation; it does not shrink the necessary KV data for a fixed sequence.
- Prefix sharing: reuse identical prefix KV blocks across compatible requests when supported.
- Sliding-window attention: bound retained context for layers whose attention semantics permit it.
- Cache eviction/compression: can lose useful information; evaluate quality.
- Limit active batch size, prompt length, or generation length.
- Offload cache when transfer costs are acceptable.

Prefix reuse requires compatible model state and identical relevant token/position context. Reuse is not valid merely because two prompts have similar meanings.

#### Temporary tensors

- FlashAttention avoids materializing the full attention matrix in GPU memory.
- Fused kernels can eliminate intermediate tensors.
- Disable gradient tracking.
- Use token-budgeted scheduling and, where supported, chunked prefill to control peak work.
- Continuous batching improves utilization but can increase memory pressure by admitting more concurrent sequences.

### Interview-ready answer

I separate inference memory into weights, KV cache, and temporary tensors. For weights, I use quantization, sharding, offloading, or a smaller model. For KV memory, I consider GQA/MQA, cache quantization, prefix sharing, supported windowed attention, and concurrency/context limits. Block-based cache allocation reduces fragmentation. For intermediates, FlashAttention and fused kernels avoid large allocations, and gradient tracking is disabled. I distinguish actual byte reduction from better allocation and per-device sharding, and measure quality, latency, throughput, and memory together.

## Question 23 — Gradient clipping

### Interview question

What is gradient clipping, why is it used, and how do value clipping and global-norm clipping differ?

### Explanation

For ordinary SGD:

$$
\theta\leftarrow\theta-\eta g
$$

An unusually large gradient can cause an excessive update, a loss spike, or divergence. Clipping limits the gradient supplied to the optimizer.

Value clipping clamps each coordinate:

$$
g'_j=\max(-c,\min(g_j,c))
$$

It may change the gradient's direction.

Global-norm clipping treats all parameter gradients as a single vector and rescales them together:

$$
g'=g\min\left(1,\frac{c}{\lVert g\rVert_2}\right)
$$

For a zero norm, leave the gradient unchanged. Implementations use numerical safeguards. The common scale factor preserves direction.

Example: g = [3, 4] has norm 5. With c = 1, global-norm clipping produces [0.6, 0.8], whose norm is 1. With value threshold 1, the result is [1, 1], which changes direction.

### Correct ordering and limitations

1. Compute the loss and run backward.
2. Finish gradient accumulation for the optimizer step.
3. If using loss scaling, unscale gradients.
4. Handle non-finite gradients; clipping is not a repair for NaNs or infinities.
5. Clip the gradients.
6. Perform the optimizer step.

In distributed training, a true global norm must account for parameter shards; a local shard norm is not automatically the global model norm.

Clipping helps exploding gradients, not vanishing gradients. It does not fix bad data, numerical bugs, or a poorly chosen learning rate. Persistent clipping warrants investigation. With AdamW, a bound on input gradient norm is not a direct bound on the final parameter update because moment estimates, normalization, and weight decay also matter.

### Interview-ready answer

Gradient clipping limits unusually large gradients before the optimizer step to stabilize training. Value clipping clamps individual elements and may change direction. Global-norm clipping rescales all gradients together when their combined norm exceeds a threshold, preserving direction. I clip after backward and accumulation, and after unscaling in loss-scaled training. It mitigates exploding gradients but does not resolve the underlying cause or guarantee a bounded AdamW update.

## Question 24 — Efficient long-sequence Transformers

### Interview question

How can we reduce computational or memory cost when Transformers process long sequences?

### Explanation

For one head:

$$
QK^\top\in\mathbb{R}^{T\times T}
$$

Dense attention takes approximately O(T squared times d_h) arithmetic and, when naively materialized, O(T squared) score storage. The central distinction is between reducing token-pair interactions and executing the same exact attention more efficiently.

#### Fewer or different interactions

- Sliding-window attention: each query attends to w nearby keys.

$$
O(Twd_h)
$$

For fixed w, this is linear in T. Direct access to distant tokens is limited, although information can propagate over multiple layers.

- Sparse attention: selected local, global, strided, routed, or block connections. Complexity depends on the number of allowed edges.
- Linear attention: reformulate or approximate attention without constructing all query–key pairs. Some forms are linear in T for fixed feature dimensions, but dimension-dependent costs remain, and semantics differ from standard softmax attention.
- Token compression: pool or summarize a long sequence into fewer tokens, risking loss of details.

For T = 32,768 and w = 1,024, a window uses about 32 times fewer pairs than a full square attention matrix, ignoring boundaries. This is not a guaranteed 32-fold wall-clock speedup; causal triangular attention also changes the pair-count comparison.

#### Same exact attention, better memory behavior

FlashAttention computes exact attention in tiles without materializing the entire score matrix in GPU memory. It reduces memory traffic and intermediate storage, but dense-attention arithmetic remains quadratic in T. Floating-point evaluation order can cause small numerical differences.

Fused kernels and memory-aware scheduling can also lower peak intermediate memory.

### Gaps and pitfalls

- A dense score matrix followed by a sparse mask still incurs dense computation. Real savings need kernels that skip disallowed blocks.
- KV caching avoids repeated projection computation for earlier tokens. A new decode token still attends over its cached context; total full-attention work across a growing generation remains quadratic.
- Attention complexity is not the entire Transformer cost: projections and feed-forward layers contribute too.
- Windowing, linear attention, and compression may require architectural changes, training, or quality validation.

### Interview-ready answer

I distinguish reducing attention interactions from implementing exact attention efficiently. Windowed or sparse attention reduces token pairs, while linear attention and token compression change how context is represented or aggregated. These approaches trade efficiency against quality and long-range information access. FlashAttention preserves full attention and improves memory traffic and storage, but does not remove quadratic arithmetic. I would evaluate end-to-end latency, peak memory, and task quality rather than complexity alone.

## Question 25 — Pipeline parallelism during inference

### Interview question

How does pipeline parallelism work during LLM inference, and what are its trade-offs for single-request latency versus serving throughput?

### Explanation

Pipeline parallelism places consecutive groups of layers on different GPUs. For a 32-layer model:

- GPU 1: layers 1–8.
- GPU 2: layers 9–16.
- GPU 3: layers 17–24.
- GPU 4: layers 25–32.

Each stage keeps its layer weights and normally the KV cache for those layers. Activations cross stage boundaries. The final stage produces logits used to select the next token.

#### Single-request latency

One token must traverse all stages sequentially. Under ordinary autoregressive decoding, the next token for that same sequence is unknown until the current pass finishes and sampling occurs.

Splitting four serial layer groups across four GPUs therefore does not make one token four times faster. Stage-to-stage communication and scheduling may add latency. Pipeline parallelism can nevertheless enable a model that otherwise would not fit or would require slow offloading.

#### Serving throughput

Different independent requests or microbatches can occupy different stages simultaneously. While GPU 3 handles request A, GPU 2 can handle B and GPU 1 can handle C. This overlaps different work, rather than removing layer dependencies within one token.

Idealized example with four balanced stages, each taking 5 ms, and negligible communication:

- First microbatch finishes in 20 ms.
- Once filled, the pipeline can finish a microbatch every 5 ms.
- A single autoregressive request cannot automatically achieve one token every 5 ms; it still needs the complete pass.
- With enough independent requests, aggregate throughput can approach that completion cadence.

For p balanced stages, m independent microbatches, and stage time t:

$$
T_{\text{pipeline}}\approx(m+p-1)t
$$

$$
U\approx\frac{m}{m+p-1}
$$

This is a simplified forward-only fill/drain model, not a complete performance prediction for dynamic autoregressive serving.

### Bottlenecks and mitigations

- Pipeline bubbles: idle stages during fill, drain, or insufficient independent work.
- Stage imbalance: the slowest stage limits steady-state throughput; partition by measured time, not merely equal layer count.
- Communication: activation transfers can dominate when work units are small.
- Microbatch size: very small batches reduce GPU efficiency; large batches may hurt latency or exceed memory budgets.
- Variable sequence lengths: can create stragglers and scheduling imbalance.
- Dynamic serving: use token-aware batching and enough independent work while respecting latency SLOs.
- Prefill and decode differ: long-prompt prefill has substantial token-parallel work, whereas decode exposes autoregressive dependencies.

Pipeline parallelism splits layers. Tensor parallelism splits operations within a layer. Data parallel serving replicates whole model workers. They solve different problems and can be combined.

### Interview-ready answer

Pipeline parallelism partitions consecutive model layers across GPUs and passes activations between stages. This reduces per-GPU weight memory and usually distributes layer-local KV-cache memory. With enough independent requests or microbatches, stages process different work concurrently, improving aggregate utilization and throughput. It does not inherently reduce single-request token latency because every token still passes through all stages, and standard autoregressive decoding waits for that token before starting the next. Communication, stage imbalance, and pipeline bubbles are the main costs. I would balance stages using measured times and schedule microbatches against throughput and latency targets.

## Day 5 takeaways

1. Heads can learn different attention patterns, but specialization is not guaranteed.
2. Separate weight memory, KV-cache memory, temporary tensors, and allocation overhead.
3. Global-norm clipping preserves gradient direction; apply it before the optimizer step and after unscaling.
4. FlashAttention reduces memory traffic, not quadratic full-attention arithmetic.
5. Pipeline parallelism provides model capacity and cross-request concurrency, not automatic single-request latency reduction.

Next study session: Day 6, beginning at Question 26. Revisit these five topics through active recall before treating them as mastered.
