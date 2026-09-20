---
layout: post
title: "LLM & NLP Interview Preparation — Day 02: Questions 6–10"
date: 2026-09-20
math: true
categories: [LLM, Interview Preparation, Questions]
tags: [flashattention, pipeline-parallelism, tensor-parallelism, inference-optimization, gradient-accumulation]
description: "Day 02 study notes covering FlashAttention, pipeline and mixed parallelism, inference optimization, and gradient accumulation."
---

# LLM & NLP Interview Preparation — Day 02

Questions 6–10 from the LLM & NLP interview-preparation sequence based on the [MLE interview guide](https://chengz.art/zh/mle-prep).

**Topics:** FlashAttention, pipeline parallelism, 3D parallelism, inference optimization, and gradient accumulation  
**Status:** 5/5 complete  
**Study date:** September 20, 2026

## Progress summary

| Question | Topic | Study result | Main interview focus |
|---|---|---:|---|
| 6 | FlashAttention | 7.5/10 | IO-aware exact attention, tiling, online softmax, backward recomputation |
| 7 | Pipeline parallelism | 2.5/10 | Microbatching, bubbles, GPipe, 1F1B, stage balance |
| 8 | Mixed/3D parallelism | 2.5/10 | DP × TP × PP, communication patterns, topology-aware placement |
| 9 | Inference optimization | 3/10 | Model, kernel, cache, decoding, and serving optimizations |
| 10 | Gradient accumulation | 2/10 | Effective batch size, update timing, loss scaling, DDP synchronization |

---

## Question 6 — FlashAttention

### Interview question

Explain the core idea and advantages of FlashAttention.

### Review

The central ideas are tiling, online softmax, exact computation, unchanged quadratic arithmetic, and reduced IO. The key refinement is that standard attention cannot generally place all of Q, K, and V in SRAM. Its major overhead comes from materializing and repeatedly reading and writing the $n\times n$ score and probability matrices in HBM.

### Standard attention's IO problem

Standard attention performs:

$$
S=QK^\top
$$

$$
P=\operatorname{softmax}(S)
$$

$$
O=PV
$$

A conventional implementation may write $S$ to HBM, read it for softmax, write $P$, and read $P$ again for multiplication with $V$. These quadratic intermediates create high memory traffic.

### Tiling

FlashAttention partitions Q, K, and V into blocks sized to fit in fast on-chip SRAM. It computes one score block at a time and immediately incorporates that block into the output. It therefore avoids materializing the complete score and probability matrices in HBM.

### Online softmax

For each query row, FlashAttention maintains:

- Running maximum $m$
- Running normalization sum $\ell$
- Running weighted-output accumulator $o$

When a later tile changes the maximum from $m_{\text{old}}$ to $m_{\text{new}}$, earlier accumulated terms are rescaled by:

$$
e^{m_{\text{old}}-m_{\text{new}}}
$$

This expresses all earlier exponentials using the new maximum and preserves numerical stability.

### Exactness and complexity

FlashAttention computes mathematically exact dense attention, aside from small floating-point differences caused by operation order. It is not sparse or approximate.

- Arithmetic complexity: $O(n^2d)$, unchanged from standard dense attention
- Standard score/probability memory: $O(n^2)$
- FlashAttention additional attention memory: approximately $O(nd)$

Its runtime improvement comes primarily from reducing HBM transfers rather than reducing FLOPs.

### Backward pass

Instead of storing the full attention probability matrix, FlashAttention saves compact row-wise statistics such as log-sum-exp values. During backward propagation, it recomputes score and probability tiles from Q, K, and V. This trades additional computation for lower activation memory and less memory traffic.

### Interview-ready answer

Standard attention materializes an $n\times n$ score matrix and often an $n\times n$ softmax probability matrix in HBM. Writing and rereading these intermediates is expensive and can make attention IO-bound.

FlashAttention is an IO-aware exact-attention algorithm. It tiles Q, K, and V into blocks that fit in on-chip SRAM, computes score blocks locally, and incorporates them into the output without storing the full attention matrix.

It uses online softmax because a query row is processed across multiple key blocks. The algorithm tracks a running maximum, normalization sum, and weighted output. When the maximum increases, it rescales previous contributions by $e^{m_{\text{old}}-m_{\text{new}}}$, preserving the correct softmax normalization.

FlashAttention retains $O(n^2d)$ arithmetic complexity but reduces intermediate memory from quadratic to approximately linear in sequence length and substantially reduces HBM traffic. During backward propagation, it recomputes attention tiles instead of storing the full probability matrix.

### Follow-up

Given $m_{\text{old}}=2$ and $m_{\text{new}}=5$:

$$
e^{2-5}=e^{-3}
$$

The rescaling is required because earlier exponential terms were represented relative to the old maximum. Multiplying the old normalization sum and output accumulator by $e^{-3}$ rewrites them relative to the new maximum without revisiting earlier logits.

---

## Question 7 — Pipeline Parallelism for Large-Model Training

### Interview question

How does pipeline parallelism improve the training efficiency of very large models?

### Pipeline partitioning

Pipeline parallelism partitions a model by depth. For a 48-layer model divided across four GPUs:

- GPU 0 holds layers 1–12.
- GPU 1 holds layers 13–24.
- GPU 2 holds layers 25–36.
- GPU 3 holds layers 37–48.

During the forward pass, each stage sends output activations to the next stage. During backpropagation, activation gradients travel in the reverse direction.

Data parallelism is different: it replicates a model and divides training examples. The two methods can be combined.

### Why microbatches are required

Processing the whole global batch as one unit would leave most stages idle. Dividing it into microbatches lets an upstream stage start the next microbatch after sending the current one downstream. Once the pipeline is filled, different stages process different microbatches concurrently.

Microbatch gradients are accumulated before the optimizer update so the intended effective global batch size is preserved.

### Pipeline bubbles

The pipeline has idle periods while filling during warm-up and draining at the end. These idle slots form the pipeline bubble.

For a simplified one-direction pipeline with $p$ stages and $m$ microbatches:

$$
\text{efficiency}\approx\frac{m}{m+p-1}
$$

$$
\text{bubble fraction}\approx\frac{p-1}{m+p-1}
$$

More microbatches reduce the relative bubble, but very small microbatches can reduce GPU kernel efficiency and increase scheduling and communication overhead.

### GPipe versus 1F1B

**GPipe** runs the forward pass for every microbatch and then runs all backward passes. It is simple but retains activations for many microbatches, increasing peak memory.

**1F1B** alternates one forward and one backward microbatch after warm-up. Backward processing starts earlier, so fewer microbatch activations remain live and peak activation memory is lower.

### Communication and load balancing

Activations cross stage boundaries during the forward pass, and activation gradients cross in the reverse direction. Each stage owns its local parameters and optimizer state.

Stages should be balanced by measured execution time and memory rather than merely assigning the same number of layers. A slow stage limits the throughput of the complete pipeline.

### Interview-ready answer

Pipeline parallelism partitions consecutive model layers across devices, enabling training when the model does not fit on one GPU. Forward activations move from stage to stage, and activation gradients return in the reverse direction.

The global batch is divided into microbatches so different stages can operate concurrently. The main inefficiency is the pipeline bubble during fill and drain. Increasing the number of microbatches reduces the relative bubble but introduces trade-offs in kernel efficiency, communication, scheduling, and gradient accumulation.

GPipe performs all forwards before all backwards and has higher activation memory. A 1F1B schedule interleaves forward and backward work and lowers peak activation memory. I would balance stages using measured compute time and memory and combine pipeline parallelism with tensor and data parallelism when necessary.

### Follow-up calculation

Given $p=4$ stages and $m=8$ microbatches:

$$
\text{efficiency}=\frac{8}{8+4-1}=\frac{8}{11}\approx72.7\%
$$

$$
\text{bubble fraction}=1-\frac{8}{11}=\frac{3}{11}\approx27.3\%
$$

Increasing to 16 microbatches gives:

$$
\frac{16}{16+4-1}=\frac{16}{19}\approx84.2\%
$$

The improvement must be balanced against the overhead and reduced matrix efficiency of smaller microbatches.

---

## Question 8 — Mixed (3D) Parallel Training

### Interview question

What is mixed—or 3D—parallel training, and what problems does it solve?

### Data parallelism

Data-parallel workers process different training examples. In basic data parallelism, every worker stores a complete model replica and synchronizes gradients using all-reduce. ZeRO or FSDP can additionally shard optimizer states, gradients, and parameters.

Its primary purpose is to increase throughput. Basic replicated data parallelism does not make an oversized layer fit on one GPU.

### Tensor parallelism

Tensor parallelism partitions computation within each Transformer layer. Large matrix multiplications may be divided by rows or columns, and attention heads may be split across devices.

Partial results are combined using collectives such as all-reduce, all-gather, and reduce-scatter. Because communication occurs in almost every layer, a tensor-parallel group should normally remain within a node connected by NVLink or another high-bandwidth interconnect.

### Pipeline parallelism

Pipeline parallelism partitions the model by depth. Each stage owns consecutive layers. Forward activations move to the next stage, and activation gradients move backward. Microbatching improves concurrency but introduces bubbles and load-balancing challenges.

### GPU-count formula

$$
\text{GPUs per model replica}=TP\times PP
$$

$$
\text{Total GPUs}=DP\times TP\times PP
$$

Data parallelism controls how many replicas of the TP-and-PP-partitioned model exist.

### Communication patterns

- **Data parallelism:** gradient synchronization, or sharded parameter and gradient collectives with FSDP/ZeRO
- **Tensor parallelism:** frequent collectives inside Transformer layers
- **Pipeline parallelism:** point-to-point activation and activation-gradient transfers

### Choosing the configuration

1. Choose enough TP to make individual layers and their activations fit.
2. Keep TP groups within high-bandwidth nodes when possible.
3. Add PP if the complete model still does not fit or increasing TP creates excessive communication.
4. Use remaining GPUs for DP to increase throughput.
5. Validate global batch size, microbatch size, stage balance, communication overhead, and memory.

### Interview-ready answer

Mixed or 3D parallelism combines data, tensor, and pipeline parallelism to train models that do not fit on one GPU while maintaining throughput.

Tensor parallelism divides computation inside a layer and uses frequent collectives, so I would generally keep each TP group within a high-bandwidth node. Pipeline parallelism divides consecutive layers into stages; activations and their gradients cross stage boundaries, and microbatches keep stages active. Data parallelism creates multiple replicas of the tensor-and-pipeline-partitioned model and processes different examples.

The total GPU count is $DP\times TP\times PP$. I would first select TP and PP so the model and activations fit, then use remaining devices for DP. The configuration should minimize cross-node tensor collectives, pipeline bubbles, stage imbalance, and data-parallel synchronization while maintaining an appropriate global batch size.

### Follow-up calculation and placement

Given $DP=4$, $TP=8$, and $PP=2$:

$$
\text{GPUs per replica}=8\times2=16
$$

$$
\text{Total GPUs}=4\times8\times2=64
$$

With eight GPUs per node, each eight-GPU TP group should remain entirely within one node. The two pipeline stages of one model replica occupy two nodes, and four data-parallel replicas require eight nodes total.

---

## Question 9 — LLM Inference Optimization

### Interview question

What are the main techniques for optimizing LLM inference?

### KV cache and PagedAttention

A KV cache stores previous tokens' keys and values, avoiding prefix recomputation during autoregressive decoding. Cache memory grows with context length and concurrency and can become fragmented as variable-length requests enter and leave.

PagedAttention manages the KV cache in fixed-size physical blocks. A request sees a logical sequence of blocks, but those blocks need not be contiguous in GPU memory. Blocks are allocated as sequences grow and returned to a shared pool when requests finish. This reduces fragmentation and supports larger continuous batches.

PagedAttention does not eliminate cache memory or change mathematical attention. It improves cache allocation and serving efficiency.

### Main optimization categories

**Model level**

- Quantization
- Distillation
- Pruning when appropriate
- GQA or MQA to reduce KV-cache memory and decoding bandwidth

**Kernel level**

- FlashAttention
- Fused operators
- Optimized GEMM kernels
- CUDA Graphs

**Memory and cache level**

- KV caching
- PagedAttention
- Prefix caching and block sharing
- KV-cache quantization
- Eviction or sliding-window policies

**Decoding level**

- Speculative decoding
- Efficient sampling kernels
- Early stopping and appropriate output-length controls

**Serving level**

- Continuous batching
- Token-based capacity limits
- Chunked prefill
- Prompt-length-aware scheduling
- Admission control and autoscaling
- Disaggregated prefill and decoding when appropriate

### Interview-ready answer

I would optimize LLM inference across model, kernel, memory, decoding, and serving layers.

At the model level, quantization, distillation, and GQA/MQA reduce computation, model memory, or KV-cache bandwidth. At the kernel level, FlashAttention, fused operations, optimized matrix kernels, and CUDA Graphs reduce HBM traffic and launch overhead.

For memory, a KV cache avoids recomputing previous keys and values. PagedAttention stores the cache in noncontiguous fixed-size blocks, reducing fragmentation and enabling larger continuous batches. Prefix caching, cache quantization, and eviction policies provide additional savings.

For decoding, speculative decoding can reduce the number of sequential target-model steps. At the serving layer, continuous batching, token-based scheduling, chunked prefill, admission control, and autoscaling balance throughput and tail latency.

I would evaluate p50/p95/p99 TTFT, inter-token latency, throughput, GPU utilization, KV-cache utilization, cost per generated token, and quality regression.

### FlashAttention versus PagedAttention

| Technique | Optimizes | Mechanism | Most relevant phase |
|---|---|---|---|
| FlashAttention | Attention-kernel HBM traffic | Tiles Q/K/V into SRAM and avoids full attention intermediates | Training and prefill; also inference kernels |
| PagedAttention | Persistent KV-cache allocation in HBM | Allocates fixed-size, noncontiguous cache blocks | Serving and autoregressive decoding |

Both involve GPU HBM, but they solve different problems:

- FlashAttention improves how exact attention is computed.
- PagedAttention improves how the KV cache is stored and managed.

---

## Question 10 — Gradient Accumulation

### Interview question

What is gradient accumulation, and why is it useful when training large language models?

### Core idea

Gradient accumulation obtains an effective batch size that cannot fit into GPU memory as one physical batch:

1. Clear gradients before the accumulation window.
2. Run forward and backward passes on $K$ microbatches.
3. Allow parameter gradients to accumulate in `.grad`.
4. Perform one optimizer update.
5. Clear gradients and begin the next window.

For per-GPU microbatch size $b$, accumulation steps $K$, and data-parallel world size $D$:

$$
B_{\text{effective}}=bKD
$$

If each microbatch loss is already a mean, divide it by $K$ before `backward()` so the accumulated gradient represents the average instead of the sum.

### Correct operation order

1. `optimizer.zero_grad()` before the accumulation window
2. Forward pass
3. Compute `loss / K`
4. Call `backward()`
5. Repeat forward/backward $K$ times
6. If using mixed precision, unscale gradients
7. Clip gradients if required
8. Call `optimizer.step()`
9. Update the mixed-precision scaler
10. Call `scheduler.step()`, generally once per optimizer update
11. Call `optimizer.zero_grad()`

The scheduler convention must match whether its configured step count means optimizer updates, epochs, or another unit.

### Distributed data parallelism

DDP normally synchronizes gradients during backward. Synchronizing after every accumulation microbatch wastes communication. Use `no_sync()` for the first $K-1$ microbatches, then allow synchronization on the final backward pass before the optimizer update.

### Trade-offs

Advantages:

- Lower peak activation memory through smaller microbatches
- Larger effective global batch
- Potentially reduced gradient variance
- Less frequent DDP synchronization when `no_sync()` is used

Disadvantages:

- More serial forward/backward passes per optimizer update
- Very small microbatches may reduce GPU efficiency
- Fewer optimizer and scheduler updates for a fixed number of examples
- Larger batches may require learning-rate adjustment
- Stateful layers such as BatchNorm may prevent exact equivalence to one physical large batch

### Interview-ready answer

Gradient accumulation allows training with a large effective batch when that batch cannot fit in GPU memory. I process $K$ smaller microbatches, call backward on each one, and delay the optimizer update until the accumulation window finishes.

For microbatch size $b$, $K$ accumulation steps, and $D$ data-parallel workers, the effective global batch is $bKD$. If the microbatch loss is a mean, I divide it by $K$ before backward. I clear gradients before the window and call `optimizer.step()` and `scheduler.step()` only after $K$ backward passes.

With DDP, I suppress synchronization for the first $K-1$ microbatches using `no_sync()` and synchronize on the final backward pass. This reduces unnecessary all-reduce operations.

The method reduces peak activation memory and may reduce gradient noise, but it adds sequential work and can hurt utilization when microbatches are too small.

### Follow-up calculation

Given $b=2$, $K=4$, and $D=8$:

$$
B_{\text{effective}}=2\times4\times8=64
$$

For 100 microbatches:

$$
\text{optimizer updates}=\frac{100}{4}=25
$$

Assuming each microbatch loss is already a mean:

$$
\text{scaled loss}=\frac{\text{loss}}{4}
$$

This scaling makes the accumulated gradient equivalent to the average over four microbatches.

---

## Day 02 rapid-review checklist

- [ ] Explain why FlashAttention is IO-aware, exact, and still $O(n^2d)$.
- [ ] Derive the online-softmax rescaling factor.
- [ ] Compare GPipe and 1F1B activation memory.
- [ ] Calculate pipeline efficiency and bubble fraction.
- [ ] Explain what DP, TP, and PP partition and how they communicate.
- [ ] Calculate total GPUs as $DP\times TP\times PP$.
- [ ] Explain why TP groups should generally remain within a high-bandwidth node.
- [ ] Distinguish FlashAttention from PagedAttention.
- [ ] Organize inference optimization by model, kernel, cache, decoding, and serving layer.
- [ ] Calculate effective global batch size under gradient accumulation.
- [ ] Explain correct optimizer, scheduler, scaler, clipping, and zero-gradient order.
- [ ] Explain how DDP `no_sync()` reduces accumulation communication.

## Core formulas

$$
\text{Pipeline efficiency}\approx\frac{m}{m+p-1}
$$

$$
\text{Pipeline bubble}\approx\frac{p-1}{m+p-1}
$$

$$
\text{Total GPUs}=DP\times TP\times PP
$$

$$
B_{\text{effective}}=bKD
$$

$$
\text{Online-softmax rescaling}=e^{m_{\text{old}}-m_{\text{new}}}
$$
