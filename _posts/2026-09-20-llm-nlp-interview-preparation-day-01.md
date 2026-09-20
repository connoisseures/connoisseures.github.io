---
layout: post
title: "LLM & NLP Interview Preparation — Day 01: Questions 1–5"
date: 2026-09-20
math: true
categories: [LLM, Interview Preparation, Questions]
tags: [llm-inference, attention, kv-cache, quantization, long-context]
description: Day 01 interview notes covering GPT inference optimization, Transformer attention, KV caching, model quantization, and 128K long-context design.
---

# LLM & NLP Interview Preparation — Day 01

Questions 1–5 from the LLM & NLP interview-preparation sequence based on the [MLE interview guide](https://chengz.art/zh/mle-prep).

**Topics:** inference optimization, Transformer attention, KV cache, model quantization, and long-context systems  
**Status:** 5/5 complete  
**Study dates:** September 10–11, 2026

## Progress summary

| Question | Topic | Initial score | Main improvement |
|---|---|---:|---|
| 1 | GPT inference latency | 7.5/10 | Organize techniques by prefill, decoding, and serving bottlenecks |
| 2 | Attention and complexity | 6.5/10 | Correct formula placement, tensor shapes, and MQA/GQA interpretation |
| 3 | KV cache | 7/10 | Distinguish per-step and total-generation complexity |
| 4 | Quantization | 3.5/10 | Cover PTQ/QAT, quantization targets, precision, granularity, and hardware support |
| 5 | Long-context design | 4.5/10 | Address positional extension, training, cache memory, serving, and evaluation |

---

# Question 1 — Reducing GPT Inference Latency at Scale

## Interview question

When deploying a GPT-like model at scale, how would you reduce inference latency without significantly reducing model quality?

## Review

The answer identified most important optimizations. The primary improvement is to connect each technique to a specific bottleneck and clearly distinguish prefill from decoding.

Important refinements:

- Continuous batching removes completed sequences and admits new requests between decoding iterations.
- KV caching primarily accelerates autoregressive decoding. The initial prompt must still be processed during prefill.
- Prefix caching accelerates prefill only when requests share an identical reusable prefix.
- FlashAttention is especially useful during prefill and with long prompts.
- Decoding is often memory-bandwidth-bound, making weight and KV-cache compression valuable.
- CUDA Graphs are easiest to use when execution shapes are predictable.

## Interview-ready answer

I would first divide latency into queueing time, prefill latency, and autoregressive decoding latency. The important product metrics are time to first token, inter-token latency, end-to-end latency, and throughput under an explicit quality target.

At the serving layer, I would use continuous batching so completed requests leave the active batch and new requests enter between decoding iterations. I would limit both waiting time and total batched tokens—not only request count—because prompt lengths vary substantially.

For decoding, I would use a paged KV cache to avoid recomputing keys and values for previous tokens and to reduce memory fragmentation. When requests share an identical system prompt, prefix caching can reuse its KV blocks and reduce prefill work.

At the kernel and model levels, I would use FlashAttention, fused kernels, and CUDA Graphs when shapes are predictable. FP8, INT8, or carefully evaluated INT4 quantization can reduce weight memory and bandwidth. KV-cache quantization is particularly valuable for long contexts and high concurrency.

I would also evaluate speculative decoding. A smaller draft model proposes several tokens, and the target model verifies them in parallel. Its benefit depends on the acceptance rate and verification overhead.

Finally, I would benchmark using realistic traffic distributions and measure p50/p95/p99 TTFT, inter-token latency, throughput, GPU utilization, KV-cache utilization, cost per generated token, and quality regression.

## Follow-up — Why can larger batches worsen p99 TTFT?

Larger batches can increase the time spent waiting to form a batch. They also consume more GPU compute and KV-cache capacity, and a long prefill can block shorter or newly arrived requests.

Mitigations include:

- A strict maximum batch wait time
- Token-based rather than request-count-based limits
- Continuous batching
- Prompt-length bucketing
- Chunked prefill
- Priority queues or reserved latency-sensitive capacity
- Admission control and autoscaling based on queue depth and KV-cache utilization

---

# Question 2 — Transformer Attention and Computational Complexity

## Interview question

Explain the Transformer attention mechanism and its computational complexity.

## Correct formulation

Given input

$$
X\in\mathbb{R}^{B\times n\times d_{\text{model}}},
$$

the learned projections are

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V.
$$

Scaled dot-product attention is

$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V,
$$

where $M$ is an optional causal or padding mask. Scaling must occur **before** softmax.

If the components of Q and K have roughly unit variance, their dot product has variance proportional to $d_k$. Dividing by $\sqrt{d_k}$ keeps logits at a stable scale and prevents softmax saturation.

## Complexity

- Linear projections: $O(nd_{\text{model}}^2)$
- Attention scores and weighted sum: $O(n^2d_{\text{model}})$
- Naive attention-matrix memory: $O(n^2)$

The quadratic term becomes the dominant challenge at long sequence lengths.

## MHA, MQA, and GQA

- **MHA:** every query head has a corresponding K and V head.
- **MQA:** all query heads share one K head and one V head.
- **GQA:** groups of query heads share a smaller set of K/V heads.

MQA and GQA reduce KV-cache memory and decoding bandwidth, but they do not eliminate the quadratic token-to-token attention computation.

## Interview-ready answer

Self-attention allows every token to aggregate information from other tokens according to learned relevance scores. The model projects the input into Q, K, and V. The product $QK^\top$ produces pairwise token scores. Scaling stabilizes their magnitude, a mask enforces causal or padding constraints, and softmax converts each row into weights used to combine value vectors.

Multi-head attention performs this operation in several learned representation subspaces, allowing heads to capture different relationships. Its attention computation costs $O(n^2d)$, while the projections cost $O(nd^2)$. Naive attention also materializes a quadratic score matrix.

MQA and GQA reduce the number of K/V heads, decreasing cache size and memory bandwidth during decoding, although each query head must still attend over all token positions.

## Follow-up — Tensor shapes

Given

$$
B=2,\quad n=1024,\quad d_{\text{model}}=4096,\quad h=32,
$$

the per-head dimension is

$$
d_h=4096/32=128.
$$

### Standard multi-head attention

| Tensor | Shape |
|---|---|
| Q | $[2,32,1024,128]$ |
| K | $[2,32,1024,128]$ |
| V | $[2,32,1024,128]$ |
| Attention scores | $[2,32,1024,1024]$ |
| Per-head result | $[2,32,1024,128]$ |
| Concatenated output | $[2,1024,4096]$ |

### Multi-query attention

| Tensor | Shape |
|---|---|
| Q | $[2,32,1024,128]$ |
| K | $[2,1,1024,128]$ |
| V | $[2,1,1024,128]$ |
| Attention scores | $[2,32,1024,1024]$ |
| Per-head result | $[2,32,1024,128]$ |
| Concatenated output | $[2,1024,4096]$ |

The initial $[2,1024,128]$ answer described a single head with the head axis omitted. The final output concatenates all 32 heads back to $d_{\text{model}}=4096$.

---

# Question 3 — KV Cache in Autoregressive Inference

## Interview question

What is a KV cache, and how does it improve autoregressive LLM inference?

## Why K and V are cached but Q is not

At decoding step $t$, the new query attends to all earlier keys and values:

$$
q_tK_{1:t}^{\top}.
$$

Earlier queries are not reused because their corresponding attention outputs have already been computed. Each layer therefore computes the new token's Q, K, and V, appends K and V to the cache, and discards Q after producing the output.

## Complexity

At decoding step $t$:

- Without caching, reprocessing the full prefix requires approximately $O(t^2d)$ attention work.
- With caching, the new query attends to $t$ cached positions in $O(td)$.

Across $T$ generated tokens, the approximate attention cost changes from $O(T^3d)$ without caching to $O(T^2d)$ with caching. KV caching does not make attention constant-time because each new query still scans the growing cache.

## Memory formula

$$
M_{\text{KV}}=2LBTH_{kv}d_hs,
$$

where:

- 2 represents K and V
- $L$ is the layer count
- $B$ is batch size
- $T$ is cached sequence length
- $H_{kv}$ is the number of KV heads
- $d_h$ is the head dimension
- $s$ is bytes per element

For standard MHA, $H_{kv}d_h=d_{\text{model}}$. GQA and MQA reduce $H_{kv}$, decreasing cache size and bandwidth.

## Interview-ready answer

A KV cache stores keys and values produced for previous tokens at every Transformer layer. During decoding, those tensors do not change, so caching them avoids recomputing the full prefix at every step. Previous queries are not cached because their attention outputs have already been calculated.

At step $t$, using the cache reduces attention work from approximately $O(t^2d)$ to $O(td)$. The trade-off is memory that grows linearly with batch size, layers, context length, KV heads, head dimension, and precision.

Long contexts and high concurrency can exhaust GPU memory and limit batching. Mitigations include GQA/MQA, paged allocation, KV-cache quantization, prefix sharing, sliding-window attention, eviction policies, and admission control.

## Follow-up — KV-cache calculation

Given

$$
L=32,\quad B=1,\quad T=4096,\quad H_{kv}=8,\quad d_h=128,
$$

with BF16 storage:

$$
2\times32\times1\times4096\times8\times128=2^{28}\text{ elements}.
$$

BF16 uses 2 bytes per element:

$$
2^{28}\times2=2^{29}\text{ bytes}
=536{,}870{,}912\text{ bytes}
=512\text{ MiB}.
$$

The initial $2^{28}$ result was the element count; converting to bytes required the additional BF16 factor.

---

# Question 4 — LLM Model Quantization

## Interview question

What types of model quantization are commonly used, and when would you use each one?

## Main categories

### Post-training quantization

PTQ converts an already-trained model, often using a small calibration dataset. It is inexpensive and is usually the first method to try. Aggressive PTQ can degrade quality, particularly when weights or activations contain outliers. Examples include GPTQ, AWQ, and SmoothQuant.

PTQ is faster to **apply** than QAT, but it does not automatically make inference faster. Runtime improvement requires supported hardware and optimized kernels, and must outweigh dequantization overhead.

### Quantization-aware training

QAT simulates rounding and clipping during training so the model learns to tolerate quantization error. It can preserve quality better at low precision but requires training data, additional computation, and more engineering.

### Weight-only quantization

Formats such as W4A16 or W8A16 compress weights while retaining FP16/BF16 activations. They reduce model memory and weight-loading bandwidth and are useful for memory-bandwidth-bound autoregressive decoding.

### Weight-and-activation quantization

Formats such as W8A8 or FP8 quantize both weights and activations. They can reduce memory traffic and accelerate matrix multiplication, making them useful for compute-heavy prefill and high-throughput inference. Activations are harder to quantize because their ranges change dynamically and may contain outliers.

### KV-cache quantization

KV-cache quantization reduces cache memory and bandwidth. It is particularly useful for long contexts and high concurrency, although aggressive cache quantization may degrade attention quality.

## Precision and granularity

- **FP8:** relatively safe speed-quality trade-off on supported accelerators
- **INT8:** stronger compression, often good quality after calibration
- **INT4:** large memory savings, usually weight-only, but more sensitive to error
- **Per-tensor:** one scale per tensor; simplest but least flexible
- **Per-channel:** separate scales by channel; better range handling
- **Group-wise:** separate scales for small groups; common for INT4
- **Per-token activation:** dynamic scale per token to handle activation variation

## Interview-ready answer

I classify quantization by when it is applied, which tensors are quantized, precision, and granularity. PTQ converts a trained model using a representative calibration set and is inexpensive. QAT simulates quantization during training and may recover quality at the cost of additional training.

Weight-only formats such as W4A16 reduce model memory and bandwidth and are attractive for decoding. Weight-and-activation formats such as W8A8 or FP8 can also accelerate matrix multiplication and are useful for prefill. KV-cache quantization reduces memory that grows with sequence length and concurrency.

I would use per-channel or group-wise scaling at low precision, calibrate on representative data, and inspect outliers. I would make the final choice based on hardware and kernel support and measure task quality, TTFT, inter-token latency, throughput, memory, and cost per token.

## Follow-up — 70B model storage

Ignoring scales and other metadata, a 70-billion-parameter model requires:

| Precision | Bytes per parameter | Storage |
|---|---:|---:|
| FP16 | 2 | 140 GB |
| INT8 | 1 | 70 GB |
| INT4 | 0.5 | 35 GB |

These values exclude scales, zero points, packing overhead, temporary buffers, and KV-cache memory.

The initial calculation used $70\times10^{12}$ instead of $70\times10^9$, and used FP32/FP16/INT8 byte widths rather than FP16/INT8/INT4 widths.

---

# Question 5 — Designing a 128K Long-Context LLM

## Interview question

How would you design an LLM to support long contexts such as 128K tokens?

## Design considerations

### Positional encoding

RoPE alone does not guarantee 128K generalization. A shorter-context model typically needs an extension method such as position interpolation, NTK-aware scaling, or YaRN, followed by continued training on long sequences.

### Attention computation

Standard attention costs $O(n^2d)$. FlashAttention reduces HBM traffic and avoids materializing the full attention matrix, but it does not reduce the quadratic FLOP count. Sliding-window, block-sparse, dilated, or global-token attention can reduce asymptotic work, with a possible loss of long-range interactions.

### KV-cache memory

The cache grows linearly with context length. GQA/MQA, lower-precision cache storage, paged allocation, prefix sharing, sliding windows, and eviction policies can reduce its cost.

### Training

Use a sequence-length curriculum that gradually increases context length. Training data must contain tasks requiring genuine distant dependencies instead of only concatenated unrelated documents.

### Serving

Use chunked prefill, token-based admission control, continuous batching, and possibly disaggregated prefill and decoding. A single 128K prefill can otherwise monopolize GPU compute and increase other requests' latency.

### Evaluation

Evaluate retrieval, multi-hop reasoning, long-document comprehension, and accuracy by evidence position. Needle-in-a-haystack testing alone is insufficient. Also measure prefill latency, TTFT, throughput, peak memory, KV-cache utilization, and quality degradation as the context grows.

### RAG

RAG retrieves a smaller relevant evidence set and can avoid sending a full corpus. It does not extend the model's native context window, but it complements long-context modeling by reducing irrelevant tokens.

## Interview-ready answer

I would address 128K context at the architecture, training, kernel, memory, serving, and evaluation levels. I would extend RoPE using position interpolation, NTK-aware scaling, or YaRN and continue training on representative long sequences. Increasing the configured position limit alone is insufficient.

For exact attention, FlashAttention reduces HBM traffic and intermediate memory while retaining quadratic computation. If that cost is unacceptable, I would evaluate sparse patterns such as sliding-window or block-sparse attention, with selected global interactions where necessary.

During decoding, KV-cache memory grows linearly with context length. I would use GQA/MQA, paged allocation, quantized cache storage, and carefully designed sliding-window or eviction policies. Chunked prefill and token-based scheduling prevent very long prompts from blocking other requests.

I would train with a sequence-length curriculum and examples requiring real long-range dependencies. Evaluation should cover retrieval, multi-hop reasoning, long-document comprehension, evidence-position robustness, prefill latency, TTFT, throughput, and memory. RAG is complementary when only a small part of a large corpus is relevant.

## Follow-up — Scaling the KV cache to 128K

The sequence-length multiplier is

$$
\frac{131{,}072}{4{,}096}=32.
$$

Because cache memory is linear in sequence length:

$$
512\text{ MiB}\times32
=16{,}384\text{ MiB}
=16\text{ GiB}.
$$

KV caching prevents recomputation, but it does not remove the growing attention scan or solve cache-capacity constraints.

---

# Day 01 rapid-review checklist

- [ ] Explain TTFT versus inter-token latency and prefill versus decoding.
- [ ] Explain why continuous batching uses token limits and wait-time limits.
- [ ] Write the scaled dot-product attention equation correctly.
- [ ] Derive MHA and MQA tensor shapes, including the head axis.
- [ ] Explain why K and V are cached but Q is not.
- [ ] Calculate KV-cache memory from layers, tokens, KV heads, head dimension, and precision.
- [ ] Compare PTQ and QAT and select weight-only versus W+A quantization.
- [ ] Calculate FP16, INT8, and INT4 parameter storage.
- [ ] Explain why RoPE configuration alone does not create long-context capability.
- [ ] Distinguish FlashAttention, sparse attention, native long context, and RAG.

## Core formulas

$$
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V
$$

$$
M_{\text{KV}}=2LBTH_{kv}d_hs
$$

$$
d_h=\frac{d_{\text{model}}}{h}
$$

For raw model-weight storage:

$$
\text{bytes}=\text{parameter count}\times\text{bytes per parameter}.
$$
