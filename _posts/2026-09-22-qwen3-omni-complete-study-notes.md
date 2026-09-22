---
title: "Qwen3-Omni: Complete Study Notes"
date: 2026-09-22 00:00:00 +0000
categories: [Speech LLM, Multimodal]
tags: [qwen3-omni, speech-llm, multimodal, aut, thinker-talker, rvq, mtp, code2wav, streaming, full-duplex]
description: A complete technical study guide to Qwen3-Omni covering AuT, TM-RoPE, the MoE Thinker-Talker architecture, RVQ and MTP speech generation, causal Code2Wav, training, evaluation, modality non-degradation, and the path toward full-duplex interaction.
math: true
---

**Paper:** Qwen3-Omni Technical Report  
**arXiv:** 2509.17765  
**Study focus:** architecture, AuT, multimodal temporal alignment, Thinker–Talker, streaming speech generation, training, evaluation, non-degradation, and the path toward full-duplex interaction.

---

## 1. Big Picture

Qwen3-Omni is a unified multimodal model supporting text, image, audio, and video inputs, with text and speech outputs. It evolves the Thinker–Talker architecture from Qwen2.5-Omni while redesigning the audio front end and speech-generation stack for stronger general audio understanding and lower-latency streaming.

### High-level architecture

```text
Audio ──► AuT ───────────────┐
                             │
Image ──► Vision Encoder ────┤
                             ├──► TM-RoPE ──► MoE Thinker (30B-A3B)
Video ──► Vision Encoder ────┤                    │
                             │                    │ text / semantics
Text ────────────────────────┘                    ▼
                                             MoE Talker (3B-A0.3B)
                                                   │
                                                   ▼
                                            primary RVQ token
                                                   │
                                                   ▼
                                                  MTP
                                                   │
                                                   ▼
                                           residual codebooks
                                                   │
                                                   ▼
                                            causal Code2Wav
                                                   │
                                                   ▼
                                                 Speech
```

### Five ideas to remember

1. **AuT replaces the Whisper-based audio encoder.** It is intended as a general audio representation model rather than an ASR-only front end.
2. **Thinker and Talker use Mixture-of-Experts.** The Thinker is approximately 30B-A3B; the Talker approximately 3B-A0.3B.
3. **Thinker and Talker are more modular.** Text can serve as an intervention point for tools, RAG, safety filters, and business logic.
4. **Speech generation uses RVQ + MTP.** The large Talker predicts the primary codec stream while a small MTP module predicts residual codebooks.
5. **Code2Wav is causal.** Codec frames can be converted to waveform without waiting for future frames, supporting low-latency streaming.

---

## 2. Qwen2.5-Omni → Qwen3-Omni

Qwen2.5-Omni established the basic Thinker–Talker design:

```text
multimodal input
      │
      ▼
   Thinker
      │
      ▼
   Talker
      │
      ▼
    Speech
```

Qwen3-Omni keeps this decomposition but changes several major components:

| Component | Qwen2.5-Omni | Qwen3-Omni |
|---|---|---|
| Audio front end | Whisper-derived encoder | AuT |
| Thinker | Transformer | MoE Transformer, ~30B-A3B |
| Talker | Transformer | MoE Transformer, ~3B-A0.3B |
| Thinker/Talker interface | More tightly coupled | More modular / decoupled |
| Speech representation | Streaming codec | Multi-codebook RVQ |
| Residual codec prediction | — | MTP |
| Waveform generation | Heavier/block-oriented path | Causal Code2Wav |
| Reported theoretical first packet | — | ~234 ms in stated setup |

The conceptual shift is from simply supporting native speech to making the entire multimodal-to-speech path more modular and streaming-oriented.

---

# Part 2 — AuT: Audio Transformer

## 3. Why replace Whisper?

A Whisper-centered representation is naturally optimized around speech recognition:

```text
Audio → speech recognition → text
```

An omni model needs a representation useful for much more:

- ASR
- language identification
- prosody and emotion
- speaker information
- environmental sounds
- music
- general audio understanding

Qwen3-Omni therefore introduces **AuT**, trained from scratch on roughly **20 million hours of supervised audio**.

### Reported AuT training mixture

| Data | Approximate proportion |
|---|---:|
| Chinese + English pseudo-labeled ASR | 80% |
| Other-language ASR | 10% |
| Audio-understanding data | 10% |

The important conceptual change is that the training objective tells the encoder not to compress all useful acoustic information into “which words were spoken?”

---

## 4. Audio preprocessing

The input pipeline is approximately:

```text
Raw waveform
    │
    ▼
16 kHz audio
    │
    ▼
128-channel Mel spectrogram
25 ms window
10 ms hop
    │
    │ ~100 frames/s
    ▼
Conv2D downsampling
8× temporal reduction
    │
    ▼
AuT
    │
    │ 12.5 representations/s
    ▼
Thinker
```

A 10 ms hop gives approximately 100 Mel frames per second. After 8× downsampling:

\[
100 / 8 = 12.5 \text{ representations/s}
\]

Therefore each output representation corresponds to approximately:

\[
1/12.5 = 0.08 \text{ s} = 80 \text{ ms}
\]

### Why this matters

For a 10-second utterance:

- Mel level: about 1000 frames.
- AuT output: about 125 representations.

For 40 minutes of audio:

- 100 Hz representation: about 240,000 frames.
- 12.5 Hz representation: about 30,000 representations.

The reduction is crucial for attention cost and KV-cache pressure in long-context audio.

---

## 5. Continuous embeddings instead of transcript bottlenecks

AuT produces continuous representations:

\[
H = [h_1, h_2, \ldots, h_n], \quad h_i \in \mathbb{R}^d
\]

These representations enter the Thinker directly.

```text
Audio
  ↓
AuT
  ↓
[h1][h2][h3]...[hn]
  ↓
Thinker
```

Contrast this with a classical pipeline:

```text
Audio
  ↓
ASR
  ↓
"hello how are you"
  ↓
text tokenizer
  ↓
LLM
```

The ASR path is highly effective when lexical content is sufficient, but the transcript can discard prosody, emotion, speaker characteristics, non-speech sounds, and music.

---

## 6. Streaming AuT and dynamic attention windows

A fully bidirectional audio encoder can depend on future frames, which creates streaming latency. AuT uses dynamic attention-window sizes, with training windows spanning roughly 1–8 seconds, to support real-time prefill caching while retaining useful acoustic context.

The key systems goal is:

```text
chunk 1 → cached state
chunk 2 → reuse cached state
chunk 3 → reuse cached state
```

rather than repeatedly recomputing all previous audio.

---

# Part 3 — Multimodal Temporal Alignment and Speech Generation

## 7. TM-RoPE: time-aligned multimodal positions

Audio and video need a shared notion of time. Qwen3-Omni uses **TM-RoPE**, a time-aligned multimodal rotary positional representation.

For vision, positional information can involve temporal and spatial dimensions:

\[
(T,H,W)
\]

For audio, the temporal granularity is naturally approximately 80 ms because AuT emits 12.5 representations per second.

```text
Time:      0      80      160      240      320 ms

Audio:     A0      A1       A2       A3       A4
            |       |        |        |        |
Video:    frame   frame     ...      frame     ...
```

The goal is to let the Thinker reason about synchronized audiovisual events without relying only on fixed-duration chunks.

---

## 8. Residual Vector Quantization (RVQ)

For output speech, one acoustic frame is represented using multiple codec codebooks.

Given a continuous acoustic vector \(z\), RVQ approximates it progressively:

\[
z \approx e_{c_0} + e_{c_1} + e_{c_2} + \cdots + e_{c_K}
\]

Conceptually:

```text
z
│
├─ codebook 0 → coarse approximation
│
├─ residual → codebook 1 → finer approximation
│
├─ residual → codebook 2 → finer approximation
│
└─ ...
```

A frame is therefore represented as:

\[
C_t = (c_t^0,c_t^1,\ldots,c_t^K)
\]

The extra codebooks increase acoustic representation capacity for voice identity, timbre, prosody, and other detailed characteristics.

---

## 9. Talker + MTP

A naive design could make the large Talker autoregressively predict every codebook token. Qwen3-Omni instead separates temporal sequence modeling from within-frame refinement.

### Talker

The Talker predicts the primary codebook token:

\[
c_t^0
\]

for each temporal frame.

```text
time →

t0       t1       t2       t3
│        │        │        │
c0       c0       c0       c0
```

### MTP

The approximately 80M dense **Multi-Token Prediction** module predicts the residual codebooks:

```text
Talker
  │
  ▼
 c^0
  │
  ▼
 MTP
 /|\
c1 c2 c3 ...
```

This creates a useful hierarchy:

- **Talker:** long-range temporal speech modeling, linguistic/contextual planning, primary codec stream.
- **MTP:** local within-frame acoustic refinement.

The larger model is used where long-range modeling matters most; a much smaller model handles local acoustic detail.

---

## 10. 12.5 Hz output generation

The output codec rate is also approximately **12.5 frames per second**.

Therefore:

\[
1 / 12.5 = 80 \text{ ms}
\]

One Talker step corresponds to roughly 80 ms of output speech.

This gives a useful symmetry:

```text
INPUT AUDIO
waveform → AuT → 12.5 representations/s → Thinker

OUTPUT AUDIO
Talker → 12.5 codec frames/s → MTP → Code2Wav → waveform
```

The 80 ms / 12.5 Hz rate connects perception efficiency, multimodal temporal alignment, and low-latency speech generation.

---

## 11. Causal Code2Wav

After RVQ/MTP produces a complete codec frame, Qwen3-Omni uses a lightweight **causal ConvNet** called Code2Wav to reconstruct waveform samples.

```text
codec frame t
    │
    ▼
Code2Wav
    │
    ▼
~80 ms waveform
    │
    ▼
PLAY
```

Causality matters because the decoder does not need future codec frames. A non-causal decoder might require \(C_{t+1}\) or \(C_{t+2}\), forcing the system to wait.

The hierarchy is:

```text
Thinker
  ↓ high-level semantics/reasoning
Talker
  ↓ temporal speech sequence
MTP
  ↓ within-frame acoustic detail
Code2Wav
  ↓ local waveform reconstruction
Audio samples
```

---

## 12. Streaming and first-packet latency

The paper reports a theoretical cold-start first-audio-packet latency of approximately **234 ms** under its stated setup.

| Component | Reported latency |
|---|---:|
| Tail-packet preprocessing | 72 ms |
| Thinker TTFT | 88 ms |
| Talker TTFT | 57 ms |
| MTP | 14 ms |
| Codec decoder | 3 ms |
| **Total** | **234 ms** |

\[
72 + 88 + 57 + 14 + 3 = 234 \text{ ms}
\]

This is a theoretical first-packet figure for the specified setup, not a universal real-world latency guarantee.

### Real-time factor

For streaming speech generation:

\[
RTF = \frac{\text{generation time}}{\text{audio duration}}
\]

Real-time playback requires:

\[
RTF < 1
\]

The paper reports RTF below 1 in the evaluated concurrency settings discussed in our study, meaning generation can remain ahead of playback under those conditions.

---

## 13. Chunked prefilling

Thinker and Talker processing can overlap:

```text
time -------------------------------------------------->

Thinker: [chunk 1][chunk 2][chunk 3][chunk 4]
Talker:           [chunk 1][chunk 2][chunk 3]
```

Once the Thinker has processed an input chunk, corresponding multimodal representations can begin prefilling the Talker asynchronously while the Thinker continues processing later chunks.

---

# Part 4 — Training

## 14. Three-stage pretraining

A compact mnemonic is:

\[
\boxed{\text{Align}} \rightarrow \boxed{\text{Integrate}} \rightarrow \boxed{\text{Extend}}
\]

### Stage 1 — Encoder alignment

The LLM is initially frozen while the audio and vision interfaces are aligned to the language model representation space. Training is staged carefully so the pretrained encoders do not unnecessarily distort themselves merely to compensate for a frozen downstream model.

Conceptually:

```text
Audio → AuT → Adapter ─┐
                      ├→ frozen LLM
Image → Vision → Adapter┘
```

### Stage 2 — General multimodal pretraining

All parameters are unfrozen and the model trains on a very large multimodal mixture described as approximately **2T tokens**.

Reported modality quantities discussed in the paper include approximately:

| Modality | Tokens |
|---|---:|
| Text | 0.57T |
| Audio | 0.77T |
| Image | 0.82T |
| Video | 0.05T |
| Video + Audio | 0.05T |

The sum of the listed quantities is greater than 2T; our study preserves the paper's reported values rather than silently reconciling them.

Training includes unimodal and cross-modal relationships such as:

```text
text
image ↔ text
audio ↔ text
video ↔ text
video ↔ audio
video ↔ audio ↔ text
```

The model also uses varied natural-language prompts instead of relying on one rigid prompt formulation per task.

### Stage 3 — Long context

The context length is extended approximately:

\[
8K \rightarrow 32K
\]

while increasing long-audio and long-video examples. This is important because simply changing a context-window configuration does not teach the model to use long-range information effectively.

---

## 15. Thinker post-training

The Thinker pipeline is approximately:

```text
SFT
 ↓
Strong-to-Weak Distillation
 ├─ off-policy
 └─ on-policy
 ↓
GSPO
```

### SFT

Supervised fine-tuning teaches instruction following and conversational behavior while preserving pretrained multimodal features.

### Off-policy distillation

Strong teacher models generate high-quality responses, and the student learns from those externally generated trajectories.

### On-policy distillation

The student generates its own sequences. Teacher and student distributions are then compared on trajectories the student actually visits.

Conceptually:

```text
Student rollout
      │
      ├───────────────┐
      ▼               ▼
student logits    teacher logits
      │               │
      └───────┬───────┘
              ▼
            KL loss
```

### GSPO

The final stage applies preference/reward optimization across modalities, using rule-based rewards where correctness can be verified and model-based judging where deterministic verification is difficult.

---

## 16. Talker post-training

The Talker follows a different four-stage path:

```text
Large-scale multimodal speech training
             ↓
High-quality continual pretraining
+ long-context training
             ↓
Multilingual DPO
             ↓
Speaker fine-tuning
```

### Stage 1

Learn the basic mapping from multimodal/contextual information to speech codec sequences using a very large speech corpus.

### Stage 2

Use cleaner, higher-quality data to reduce hallucination and instability introduced by noisy large-scale data. Long-context training also teaches speech behavior to depend on extended conversational context.

### Stage 3

Multilingual DPO uses preferred vs less-preferred speech outputs to improve multilingual generation and stability.

### Stage 4

Speaker fine-tuning improves particular voices, naturalness, expressiveness, and controllability.

---

# Part 5 — Audio → Text Evaluation

## 17. ASR

For ASR, lower WER is better:

\[
WER = \frac{S + D + I}{N}
\]

Representative paper-reported results discussed in our study:

| Benchmark | Qwen2.5-Omni | Qwen3-Omni |
|---|---:|---:|
| LibriSpeech clean | 1.74 | 1.22 |
| LibriSpeech other | 3.45 | 2.48 |
| CommonVoice15 EN | 7.61 | 6.05 |
| CommonVoice15 ZH | 5.13 | 4.31 |
| FLEURS EN | 3.77 | 2.72 |
| FLEURS ZH | 2.54 | 2.20 |

The system-level improvement is clear, but it should **not** be interpreted as a controlled proof that AuT alone is better than Whisper. Qwen3-Omni changes many variables simultaneously: encoder, architecture, training data, scale, long-context training, and post-training.

---

## 18. Why audio reasoning is different from ASR

ASR asks:

> What words were spoken?

General audio understanding may ask:

> What happened acoustically?

Audio reasoning may ask:

> What can we infer from the speech, emotion, background sounds, and temporal sequence?

Example:

```text
Audio:
"I can't believe you did that!"
[glass breaking]
[person crying]
```

ASR may only produce the spoken sentence. A native audio model can potentially reason over the emotional tone, breaking sound, and subsequent crying as well.

The core idea is:

\[
\boxed{\text{Audio understanding} \neq \text{ASR}}
\]

---

## 19. Music understanding

The paper evaluates tasks such as genre, mood/theme, instrument recognition, music tagging, and general music understanding.

Representative improvements discussed in our study include:

| Benchmark | Qwen2.5-Omni | Qwen3-Omni |
|---|---:|---:|
| RUL-MuchoMusic | 47.3 | 52.0 |
| GTZAN genre accuracy | 81.7 | 93.0 |
| MTG Genre micro-F1 | 32.5 | 39.0 |
| MTG Mood/Theme | 8.9 | 21.0 |
| MTG Instrument | 22.6 | 40.5 |
| MagnaTagATune | 30.1 | 44.3 |

These results reinforce the motivation for a general audio encoder rather than an ASR-specific representation.

---

# Part 6 — X → Speech Evaluation

## 20. What speech evaluation measures

A good speech-generation model needs both:

1. **Content fidelity** — say the correct words.
2. **Speaker/acoustic fidelity** — preserve voice identity and natural speech characteristics.

These are different axes. Low WER/CER does not guarantee good speaker similarity, and vice versa.

---

## 21. Zero-shot speech generation

Representative SEED-TTS content-consistency results discussed in our study:

| Model | Chinese ↓ | English ↓ |
|---|---:|---:|
| Qwen2.5-Omni | 1.42 | 2.33 |
| Qwen3-Omni | 1.07 | 1.39 |

The improvement indicates more stable speech generation, with fewer omissions, repetitions, and hallucinated continuations.

Autoregressive speech instability can look like:

```text
Desired:
"The meeting starts at ten tomorrow."

Failure:
"The meeting starts at ten ten ten tomorrow tomorrow..."
```

---

## 22. Multilingual and cross-lingual generation

Qwen3-Omni supports speech generation across multiple languages and is evaluated on multilingual content consistency and speaker similarity.

Cross-lingual voice cloning is especially challenging:

```text
Reference speaker in language L1
        │
        ├─ speaker identity
        │
Target content in language L2
        │
        ▼
Speech in L2 with identity from L1
```

The model must disentangle speaker identity from source-language pronunciation while producing correct phonology and rhythm in the target language.

The evaluation shows strong results in several cross-lingual directions, but not uniform superiority in every language pair. This is important: strong overall speech generation should not be simplified into “best on every language.”

---

## 23. Evaluation limitation

The Talker architecture is designed to use rich conversational and multimodal context:

```text
conversation history
+ audio/video context
+ semantic response
        ↓
      Talker
        ↓
context-aware speech
```

However, available quantitative speech-generation benchmarks are still heavily TTS-like. Therefore, current evaluation does not fully measure all of the context-sensitive behavior the architecture is intended to support.

---

# Part 7 — Non-Degradation Across Modalities

## 24. The central question

When text, vision, and audio share one model, their training gradients can conflict:

\[
\nabla L_{text}, \quad \nabla L_{vision}, \quad \nabla L_{audio}
\]

A major concern is negative transfer: improving audio could damage text reasoning or vision.

The paper performs controlled comparisons intended to test whether the omni model preserves language and vision capability while adding audio.

### Main observations

1. Adding vision does not meaningfully damage language capability.
2. Adding audio does not meaningfully damage language or vision capability.
3. Audio training is reported to improve some vision benchmarks, including MMMU and OCR-related tasks.

The paper reports the empirical effect, but it does not fully establish the causal mechanism.

---

## 25. Possible positive transfer

A plausible interpretation is that synchronized modalities provide complementary supervision.

```text
           semantic event
               ▲
        ┌──────┼──────┐
        │      │      │
      image   audio   text
```

For audiovisual examples, different signals describe the same underlying event. This may encourage stronger shared concepts such as object, action, event, temporal relation, and physical interaction.

This is interpretation, not a mechanism conclusively demonstrated by the paper.

---

## 26. Why MoE is conceptually attractive

The Thinker is an MoE model. Different tokens may route through different subsets of experts:

```text
Audio token  → E1 + E3
Vision token → E2 + E3
Text token   → E4 + E3
```

This offers a possible balance between specialization and shared knowledge. However, the paper does not establish that MoE is the cause of non-degradation, so this should remain an architectural hypothesis rather than a proven conclusion.

---

# Part 8 — What Qwen3-Omni Still Does Not Solve

## 27. Streaming is not the same as full duplex

Qwen3-Omni supports streaming input and streaming output. That does **not** automatically imply full-duplex conversational interaction.

### Streaming interaction

```text
User speaking
██████████████
              ↓
Assistant speaking
              █████████████
```

### Full-duplex interaction

```text
time ---------------------------------------->

User: ██████████       ███████
                     ↑ interruption

AI:          ███████████████
                   ↓
             stop / adapt
```

A genuinely full-duplex system needs simultaneous listening and speaking plus an interaction policy.

---

## 28. Turn detection

A simple endpoint detector might estimate:

\[
P(\text{speech ended})
\]

But a conversational speech LLM needs something closer to:

\[
P(\text{assistant should respond now}\mid
\text{audio, prosody, semantics, dialogue context, model state})
\]

Example:

```text
"I think we should..."
       [pause]
"...try the other model."
```

Silence alone is insufficient because the pause may be hesitation rather than a completed turn.

---

## 29. Barge-in

Suppose the assistant says:

```text
AI: "The easiest way to configure this is—"
User: "No, I already tried that."
```

The system must decide:

- Is this genuine user speech?
- Is it background speech?
- Is it the assistant hearing its own output?
- Should audio playback stop?
- Should semantic generation be canceled?
- Should the Thinker replan?

This is an interaction-control problem beyond low-latency synthesis.

---

## 30. Listening while speaking

With an open microphone during assistant playback, the observed signal may be:

\[
x(t)=\text{user}(t)+\text{assistant output}(t)+\text{environment}(t)
\]

A production system may need:

- acoustic echo cancellation
- speaker separation
- self-speech awareness
- streaming acoustic modeling
- interruption classification

Qwen3-Omni's core paper is not centered on solving this complete problem.

---

## 31. Prepare before committing

A low-latency spoken LLM should not necessarily wait for an endpoint before beginning all computation.

```text
User speech ------------------------------------------------>
Understanding     ------------------------------------------>
Reasoning                    ------------------------------->
Turn policy                         WAIT → COMMIT
Talker                                      --------------->
Speech                                         ------------>
```

The model can continuously understand and prepare a response while the user is speaking. Then the endpoint decision becomes a **commit decision** rather than the beginning of reasoning.

This can reduce perceived latency dramatically.

---

## 32. Why 234 ms is not the whole conversation latency

Perceived conversational latency is better modeled as:

\[
T_{conversation}
=
T_{turn\ decision}
+
T_{model\ response}
+
T_{first\ audio}
\]

For example, if semantic endpointing adds 600 ms and speech generation requires 234 ms to first packet, the user may wait approximately 834 ms.

Therefore reducing endpoint/turn-decision latency can matter as much as optimizing synthesis latency.

---

## 33. A possible full-duplex extension

A natural extension of the architecture is:

```text
Microphone
    │
    ▼
Streaming AuT
    │
    ▼
Streaming Thinker
    │
    ├── semantic reasoning
    ├── response preparation
    └── turn-state representation
            │
            ▼
      Interaction Controller
       ├── wait?
       ├── speak?
       ├── interrupt?
       └── continue?
            │
            ▼
          Talker
            │
            ▼
      MTP + Code2Wav
            │
            ▼
           Speech
```

The missing abstraction is an explicit **interaction policy** coordinating perception, reasoning, and speaking.

---

# Final Synthesis

## 34. Qwen audio-model evolution

```text
Qwen-Audio
   │
   │ audio understanding
   ▼
Qwen2-Audio
   │
   │ more general audio-language interaction
   ▼
Qwen2.5-Omni
   │
   │ Thinker + Talker
   │ multimodal understanding + native speech
   ▼
Qwen3-Omni
   │
   ├─ AuT
   ├─ MoE Thinker
   ├─ MoE Talker
   ├─ TM-RoPE
   ├─ 12.5 Hz audio representations
   ├─ RVQ + MTP
   └─ causal Code2Wav
   │
   ▼
Next frontier:
full-duplex interaction
```

The research progression can be summarized as:

\[
\boxed{\text{Audio Understanding}}
\rightarrow
\boxed{\text{Audio-Language Interaction}}
\rightarrow
\boxed{\text{Multimodal Understanding + Speech}}
\rightarrow
\boxed{\text{Real-Time Omni Interaction}}
\rightarrow
\boxed{\text{Full-Duplex Interaction}}
\]

---

## 35. Interview / Research Cheat Sheet

### Why AuT instead of Whisper?

To build a general audio representation useful beyond transcription, including music, acoustic events, prosody, and broader audio understanding.

### Why continuous input embeddings?

They can preserve acoustic evidence that would be lost when audio is compressed into a transcript.

### Why 12.5 Hz?

It reduces sequence length and autoregressive workload while maintaining a practical temporal granularity of about 80 ms.

### Why Thinker–Talker?

It separates semantic reasoning from speech realization and provides a clean intervention point for tools, safety, RAG, and business logic.

### Why RVQ?

Multiple codebooks increase acoustic representation capacity.

### Why MTP?

A small module can predict within-frame residual codebooks while the larger Talker focuses on long-range temporal sequence modeling.

### Why causal Code2Wav?

It allows each codec frame to become audible without waiting for future frames.

### Does streaming mean full duplex?

No. Full duplex additionally requires turn-taking, interruption handling, simultaneous listening/speaking, self-echo management, and an interaction policy.

### What is the next research frontier?

Move from optimizing **speech generation latency** to optimizing **interaction latency and turn-taking policy**.

---

## 36. Final Mental Model

```text
PERCEPTION
════════════════════════════════════

Audio → AuT (12.5 Hz) ──┐
Vision ─────────────────┼─→ TM-RoPE → Thinker
Text ───────────────────┘


REASONING
════════════════════════════════════

MoE Thinker
30B-A3B

understanding
reasoning
tools
multimodal fusion

        │
        ▼
semantic response / text


SPEAKING
════════════════════════════════════

MoE Talker
3B-A0.3B
    │
    ▼
primary RVQ token
    │
    ▼
MTP
    │
    ▼
residual codebooks
    │
    ▼
causal Code2Wav
    │
    ▼
Speech

~80 ms / codec frame


NEXT FRONTIER
════════════════════════════════════

streaming perception
+
streaming reasoning
+
streaming speech
        │
        ▼
interaction policy
        │
        ├─ turn completion
        ├─ barge-in
        ├─ wait / speak
        ├─ self-echo handling
        └─ full duplex
```

## One-sentence takeaway

**Qwen3-Omni turns the Thinker–Talker idea into a modular, MoE-based, streaming-native multimodal architecture in which perception, reasoning, temporal speech modeling, acoustic refinement, and waveform reconstruction are deliberately separated by timescale and responsibility.**

---

## References

- Qwen3-Omni Technical Report, arXiv:2509.17765.
- These notes consolidate the complete Qwen3-Omni study discussion. Paper-reported results are separated from our architectural interpretations and proposed full-duplex extensions.