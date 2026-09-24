---
title: "Chapter 17: Text-to-Speech — Complete Study Notes"
date: 2026-09-24 00:00:00 +0000
categories: [Speech, Text-to-Speech]
tags: [tts, encodec, vector-quantization, rvq, vall-e, zero-shot, speech-synthesis]
description: "A guided study of zero-shot TTS, neural audio codecs, residual vector quantization, codec training, VALL-E, streaming implications, and speech evaluation."
math: true
---

- **Book:** Speech and Language Processing, Daniel Jurafsky and James H. Martin
- **Source:** Uploaded 17.pdf, Chapter 17, draft dated August 19, 2026
- **Study dates:** September 23–24, 2026
- **Scope:** Consolidated notes from our guided study, with corrected concept checks. This is a study reference, not a verbatim transcript.
- **Draft limitation:** Section 17.6, “Spoken Language Models,” contains only “TBD.” No substantive material for that section is present in the supplied PDF.
- **Related study:** [Chapter 16 — Automatic Speech Recognition](https://app.notion.com/p/3e4a4c7adafe81319b27c3051131f63b)

## 1. Executive summary

Text-to-speech (TTS) generates a speech waveform corresponding to requested text. In the zero-shot setting studied here, a short recording supplies the desired speaker’s voice, even if that speaker was absent from training.

The chapter’s central approach is conditional language modeling over discrete audio tokens. A pretrained neural audio codec provides the audio representation. A TTS language model predicts new audio token IDs, and codebook lookup followed by the codec decoder converts them into speech.

The three roles must stay distinct:

| Component | Input | Output and purpose |
|---|---|---|
| Codec encoder | Existing waveform | Continuous audio feature vectors |
| Quantizer / RVQ | Continuous encoder vectors | Discrete codebook IDs and their reconstructed vectors |
| TTS language model | Text and reference-audio conditioning | New audio token IDs for the requested speech |
| Codebook lookup and codec decoder | Generated audio IDs | Reconstructed vectors, then a waveform |

**Core generation pipeline:** Target text + reference speech → TTS language model → generated audio token IDs → codebook lookup and summation → codec decoder → waveform.

The reference waveform is first encoded and quantized to obtain its audio prompt.

## 2. Section 17.1 — TTS overview and zero-shot synthesis

### 2.1 What TTS must generate

ASR maps speech to text. TTS maps text to speech. Text alone does not uniquely specify a waveform: the same sentence can be spoken with different voices, speeds, emotions, and intonation.

In the chapter’s zero-shot setup, the inputs are:

- Target text specifying what to say.
- A short reference recording, perhaps three seconds, demonstrating the desired voice.

Example: Alice’s reference says “I like coffee,” while the target text says “Good morning.” The desired output says **“Good morning” in a voice resembling Alice’s**, not a replay of “I like coffee.”

### 2.2 Why it is called zero-shot

The system can generalize to an unseen speaker without speaker-specific training or fine-tuning. The model’s weights remain fixed during synthesis; the reference recording is conditioning input.

| Operation | What changes? | Do model weights change? |
|---|---|---|
| Change the target sentence | Text input | No |
| Change Alice’s reference to Bob’s | Audio conditioning | No |
| Fine-tune on a speaker | Learned parameters through training | Yes |

“Zero-shot” does not mean there is no reference audio. It refers here to producing an unseen voice without updating parameters for that speaker.

## 3. Section 17.2 — Neural audio codecs

A codec compresses audio and reconstructs an approximation of it. In this chapter, its discrete representation is also useful as the vocabulary for audio language modeling.

### 3.1 Encoder and decoder

The encoder reduces the temporal resolution of the waveform using strided convolutions. The EnCodec architecture shown in the chapter includes convolutional blocks, residual units, and an LSTM. The decoder upsamples reconstructed latent vectors using a corresponding decoding network.

For the chapter’s example:

- Waveform sampling rate: 24,000 samples/second.
- Encoder frame rate: 75 vectors/second.
- Illustrative vector dimension: D = 256.

The temporal downsampling factor is:

$$
\frac{24{,}000}{75}=320.
$$

A frame step therefore corresponds to approximately 13.33 ms. This is a temporal stride, not a claim that the encoder’s receptive field is only 13.33 ms.

The chapter uses D = 256 to explain the mechanism; it is not a universal value for every codec configuration.

### 3.2 Vector quantization

A learned codebook is a table of vectors. For each encoder vector, the quantizer chooses a nearby codebook vector and stores its index.

Let the encoder vector be z and codebook entries be e[i]. Nearest-neighbor quantization can be written as:

$$
q=\operatorname*{arg\,min}_{i}\|z-e[i]\|_2^2,
\qquad
\hat z=e[q].
$$

Here q is an integer token ID. It is neither a transcript word nor a continuous feature vector.

Toy example:

| Item | Value |
|---|---|
| Encoder output z | [0.9, 0.2] |
| Codebook entry 0 | [0.0, 0.0] |
| Codebook entry 1 | [1.0, 0.0] |
| Codebook entry 2 | [-1.0, 1.0] |
| Selected ID | 1 |
| Reconstructed vector | [1.0, 0.0] |

The reconstructed vector approximates the original. Their difference is quantization error.

### 3.3 Residual vector quantization (RVQ)

RVQ uses successive codebooks to approximate the error left by earlier stages.

For one encoder frame, define:

$$
r_0=z.
$$

At stage k:

$$
q_k=\operatorname*{arg\,min}_{i}\|r_{k-1}-e_k[i]\|_2^2,
$$

$$
v_k=e_k[q_k],
\qquad
r_k=r_{k-1}-v_k.
$$

The reconstructed feature vector is:

$$
\hat z=\sum_{k=1}^{K}v_k
      =\sum_{k=1}^{K}e_k[q_k].
$$

All selected vectors have dimension D. We **sum** them; we do not concatenate them.

Example:

$$
z=[0.9,0.2], \qquad v_1=[1.0,0.0],
$$

$$
r_1=z-v_1=[-0.1,0.2].
$$

The second codebook approximates [-0.1, 0.2], the remaining error, rather than independently approximating the original vector.

An ID tuple such as [42, 17, 8] describes one frame using three different codebooks. The IDs do not describe three consecutive frames. An ID is interpreted together with its codebook index.

### 3.4 Tensor shapes and axes

Use these conventions throughout these notes:

- B: batch size.
- N: waveform sample count.
- T: number of encoder frames.
- D: encoder-vector dimension.
- K: number of RVQ codebooks.

| Representation | Shape without batch | Shape with batch |
|---|---|---|
| Mono waveform | [N] | [B, N] |
| Continuous encoder vectors | [T, D] | [B, T, D] |
| Audio token IDs | [T, K] | [B, T, K] |
| Looked-up vectors before summation | [T, K, D] | [B, T, K, D] |
| Quantized vectors after summation | [T, D] | [B, T, D] |

These are conceptual axis conventions; an implementation may use channel-first layouts.

For one second at 24 kHz, with 75 encoder frames, D = 256, and K = 8:

| Step | Shape |
|---|---|
| Waveform | [24000] |
| Encoder output | [75, 256] |
| RVQ IDs | [75, 8] |
| Summed codebook vectors | [75, 256] |
| Decoded waveform | [24000] |

There are 75 × 8 = 600 token IDs, but only 75 time frames.

Additional checks:

- 100 frames and 4 codebooks → [100, 4], containing 400 IDs.
- 100 frames and 8 codebooks → [100, 8], containing 800 IDs.
- Summing eight 256-dimensional vectors gives one 256-dimensional vector, not a 2048-dimensional vector.

The draft has an inconsistent prose description of the axes near Equation 17.5. These notes consistently use rows for time and columns for codebooks, matching [T, 8] and the indexing C[t, k].

## 4. Section 17.2.4 — Codec training

### 4.1 Reconstruction task

The codec is trained as an autoencoder:

- Input: a waveform x.
- Processing: encoder → quantizer → decoder.
- Output: reconstructed waveform x-hat.
- Training target: the same original waveform x.

A recording saying “Hello” has that same recording as its target. It is not trained to substitute another sentence. This reconstruction task does not require text transcripts.

### 4.2 Reconstruction loss

A simple waveform reconstruction objective is:

$$
L_{\mathrm{reconstruction}}
=\sum_{n=1}^{N}(x_n-\hat x_n)^2.
$$

Here n indexes waveform samples, not codec frames. The chapter also describes comparing frequency-domain representations such as mel-spectrograms, using L1 and/or L2 differences.

Reconstruction losses encourage fidelity to the particular input, including its speech content and acoustic properties.

### 4.3 Adversarial loss

A discriminator learns to distinguish original audio from reconstructed audio. The codec acts as the generator and learns to produce reconstructions that the discriminator judges to be real.

| Component | Goal |
|---|---|
| Discriminator | Distinguish real recordings from reconstructions |
| Codec / generator | Produce convincing reconstructions |

Realism alone is insufficient: speech can sound realistic while saying the wrong words. This motivates combining adversarial and reconstruction objectives.

### 4.4 Non-differentiable quantization and STE

Nearest-codebook selection is discrete. A small change to an encoder vector often leaves the selected entry unchanged, so ordinary gradients through this choice do not provide useful training feedback.

The straight-through estimator (STE) uses:

- **Forward pass:** Actual quantization.
- **Backward pass:** Approximate the quantizer’s derivative by an identity mapping.

Thus:

$$
\frac{\partial L}{\partial z}
\approx
\frac{\partial L}{\partial z_q}.
$$

This approximation allows reconstruction feedback to reach the encoder. It does not mean quantization is skipped in the forward pass, and it does not by itself specify how to update the codebook.

### 4.5 Quantizer learning and combined loss

The chapter discusses initializing codewords using k-means and using a quantization-related loss to keep encoder/residual vectors and their codebook approximations close.

Its schematic combined objective is:

$$
L=
\lambda_{\mathrm{rec}}L_{\mathrm{reconstruction}}
+\lambda_{\mathrm{adv}}L_{\mathrm{GAN}}
+\lambda_{\mathrm{VQ}}L_{\mathrm{VQ}}.
$$

The coefficients balance the objectives. Exact codebook update rules and detailed loss implementations should be checked for the particular codec; this study covered the mechanism rather than reproducing a training implementation.

## 5. Section 17.3 — VALL-E: two-stage audio-token generation

### 5.1 Distinguish codec training from TTS language-model training

The codec learns to reconstruct existing audio. The TTS language model learns a conditional distribution over audio tokens corresponding to text.

For paired text and speech, the pretrained codec converts the speech into a code matrix C with shape [T, 8]. The language model learns to predict these codes conditioned on the text and the relevant prompt/context.

A general conditional negative log-likelihood objective is:

$$
L_{\mathrm{TTS}}=-\log p(C\mid x,\text{conditioning}).
$$

This expression is explanatory notation, not a verbatim reproduction of the draft’s factorization.

### 5.2 Why two stages?

RVQ has a hierarchy: the first codebook provides a coarse approximation, and later codebooks refine residual information. VALL-E uses this structure to split generation into:

1. An autoregressive (AR) model for codebook 1.
2. A non-autoregressive (NAR) model applied successively to codebooks 2–8.

The first codebook is not a text transcript or guaranteed to contain only semantic information; it is an acoustic representation.

### 5.3 Stage 1: first codebook across the entire utterance

Stage 1 generates codebook-1 tokens sequentially over time, conditioned on text, audio prompt, and previously generated first-codebook tokens.

For 100 output frames, it generates 100 audio IDs, excluding the end-of-sequence token.

| Frame | Codebook 1 | Codebook 2 | Remaining codebooks |
|---|---|---|---|
| 1 | Generated first | Pending | Pending |
| 2 | Generated second | Pending | Pending |
| 3 | Generated third | Pending | Pending |
| … | … | Pending | Pending |
| 100 | Generated last | Pending | Pending |

**Stage 1 fills the first column, not the first row.**

It does not generate all eight tokens of frame 1 before moving to frame 2.

### 5.4 Stage 2: remaining codebooks

Once the first-codebook sequence is complete, the NAR model fills the remaining codebooks in order.

| Pass | Output | Time-axis generation |
|---|---|---|
| AR pass | Codebook 1 for all frames | Sequential |
| NAR pass 1 | Codebook 2 for all frames | Parallel |
| NAR pass 2 | Codebook 3 for all frames | Parallel |
| … | … | … |
| NAR pass 7 | Codebook 8 for all frames | Parallel |

For each NAR pass, generation is parallel across time, while codebooks are generated successively. Each pass conditions on available earlier codebook sequences and prompt information.

This uses a NAR model repeatedly; “seven passes” does not imply seven independently trained NAR models.

### 5.5 Attention to later frames

When predicting codebook 2 at frame 50, the model can use codebook 1 at frame 80 because that entire first-codebook sequence has already been generated.

**Later in audio time does not necessarily mean unavailable in computation.**

Stage 1 cannot depend on ungenerated future tokens in its own first-codebook sequence. Stage 2 can condition on completed earlier-codebook sequences across the utterance.

### 5.6 Final decoding

After completing [T, 8] IDs:

1. Look up each ID in its corresponding codebook.
2. Sum the eight vectors for each frame.
3. Feed the resulting [T, D] sequence into the codec decoder.
4. Reconstruct the waveform.

The language model predicts the new acoustic token sequence. The codec decoder does not read the target sentence to decide which words to say.

## 6. Streaming implications for conversational speech systems

This section records our engineering interpretation of the two-stage inference procedure, rather than a separate streaming algorithm proposed by the chapter.

In the procedure studied:

1. Stage 1 completes the first-codebook sequence.
2. Stage 2 fills the remaining codebooks.
3. The decoder reconstructs the audio.

An autoregressive first stage does **not** automatically make the complete system capable of immediately streaming playable speech.

For a 100-frame example:

| Stage 1 progress | Is Stage 2 ready in this procedure? |
|---|---|
| 1 first-codebook token | No |
| 50 first-codebook tokens | No |
| All 100 first-codebook tokens | Yes, once the sequence is complete |

The first internal token is not the same as the first audible sample. End-to-end streaming requires a generation and decoding procedure that can produce valid audio chunks without waiting for the complete utterance.

This is a limitation of the described schedule, not a claim that all RVQ systems or all neural codecs are inherently non-streaming.

For a spoken assistant, distinguish:

- Time to first internal token.
- Time to first playable audio.
- Total synthesis latency.

These quantities can differ substantially.

## 7. Section 17.4 — TTS evaluation

TTS quality has multiple dimensions: perceptual quality, spoken content, and voice match.

| Metric | What it checks | Method and caveat |
|---|---|---|
| MOS | Perceived speech quality | Listeners rate samples, usually on a 1–5 scale |
| CMOS | Relative quality of two systems | Compare the same sentence; the chapter describes a -3 to +3 scale |
| ASR-based WER | Intelligibility and intended-word accuracy | Transcribe synthesized audio and compare with target text |
| Speaker similarity | Match to the reference voice | Compare generated and reference audio using speaker verification |

WER is conventionally:

$$
\mathrm{WER}=\frac{S+D+I}{N_{\mathrm{ref}}},
$$

where S, D, and I are substitutions, deletions, and insertions, and N-ref is the number of reference words.

ASR-based WER is influenced by the recognizer as well as by the synthesized speech. A low value does not guarantee naturalness or speaker fidelity.

Examples:

- Correct words, robotic delivery: WER may be low while human quality ratings are poor.
- Correct words, Bob’s voice instead of Alice’s: speaker similarity is the direct check.
- Realistic speech, incorrect words: realism alone misses the content error.

Compare systems using matched sentences and appropriate listener protocols; the chapter discusses statistical comparison of ratings.

## 8. Section 17.5 — Other speech tasks

| Task | Question | Example |
|---|---|---|
| Speaker diarization | Who spoke when? | Speaker A: 0–5 s; Speaker B: 5–9 s |
| Speaker verification | Is this the claimed speaker? | Does this voice match Alice’s enrolled recording? |
| Speaker identification | Which known speaker is this? | Choose among speakers in a database |
| Language identification | What language is spoken? | Identify the language of a recording |
| Wake-word detection | Was the activation phrase spoken? | Detect a phrase that activates an assistant |

Diarization can use anonymous speaker labels without knowing names. A common pipeline detects speech, extracts speaker embeddings, and clusters segments; the chapter also mentions end-to-end frame-level speaker labeling.

Verification checks an identity claim, usually through a score and decision threshold. Identification instead selects among multiple candidates.

Wake-word systems emphasize low latency and small computational footprint, often on-device. Wake-word detection is distinct from endpoint detection: it identifies an activation phrase rather than deciding when a user’s turn has ended.

## 9. Section 17.6 — Spoken language models

The supplied draft contains only “TBD” under this heading. We did not invent a missing chapter section or treat the heading as substantive coverage of full-duplex speech LLMs.

Our discussion of streaming was an implication of the VALL-E inference schedule, not a complete treatment of spoken language models.

## 10. Corrected concept checks

These retain the useful learning points while omitting repeated attempts.

| Question | Correct answer |
|---|---|
| Why is zero-shot TTS called zero-shot despite reference audio? | It generalizes to an unseen speaker without speaker-specific weight updates |
| What changes when switching Alice’s voice to Bob’s? | The reference conditioning, not the model weights |
| Which component turns continuous encoder vectors into discrete IDs? | The quantizer |
| Is audio token 42 a word? | No; it is a codebook index |
| Does quantization preserve the encoder vector exactly? | Generally no; it approximates it |
| What does RVQ’s second stage approximate? | The residual error left by stage 1 |
| How are selected codebook vectors combined? | Summation |
| What is the ID shape for 100 frames and 4 codebooks? | [100, 4] |
| What is the vector shape after summing 8 codebooks with D = 256 over 75 frames? | [75, 256] |
| Who predicts audio tokens for new target text? | The TTS language model |
| Does the codec decoder read the target text to choose the words? | No; it reconstructs audio from the acoustic representation |
| Reference says “I like coffee”; target says “Good morning.” What should be spoken? | “Good morning,” in the reference speaker’s voice |
| What is the codec reconstruction target? | The original input recording |
| Is quantization still performed in the STE forward pass? | Yes |
| Can realistic audio contain incorrect words? | Yes |
| What does VALL-E Stage 1 generate? | Codebook 1 for the entire utterance, sequentially over time |
| How many Stage-1 audio IDs for 100 frames? | 100, excluding EOS |
| Can Stage 2 at frame 50 use codebook 1 at frame 80? | Yes; that codebook sequence is complete |
| When does Stage 2 start in the studied procedure? | After the complete first-codebook sequence |
| Does an AR first stage guarantee immediately playable streaming audio? | No |
| Which metric directly detects the wrong requested voice? | Speaker similarity |
| Anonymous speaker turns with timestamps: what task? | Diarization |
| Match a voice against an enrolled identity: what task? | Verification |

## 11. Rapid-review checklist

- [x] Explain zero-shot conditioning versus fine-tuning.
- [x] Separate encoder, quantizer, TTS language model, and decoder.
- [x] Interpret an audio token as a codebook ID.
- [x] Compute an RVQ residual and sum codebook vectors.
- [x] Distinguish time, codebook, and feature dimensions.
- [x] Explain codec reconstruction, adversarial training, and STE.
- [x] Describe VALL-E’s AR first column and successive NAR columns.
- [x] Explain why later-time tokens may already be available in Stage 2.
- [x] Distinguish first-token latency from first-audio latency.
- [x] Separate word accuracy, quality, and voice similarity.
- [x] Distinguish diarization from verification.
- [x] Recognize the draft’s missing Section 17.6 content.

## 12. Compact interview answer

A codec-based TTS system uses a neural audio codec to represent speech as discrete tokens. The codec encoder downsamples waveforms into latent vectors, and residual vector quantization represents each frame using several codebook IDs. A TTS language model predicts new IDs conditioned on target text and reference speech. In zero-shot synthesis, the reference supplies voice information without updating model weights. VALL-E generates the first codebook autoregressively over the utterance, then generates the remaining codebooks with successive non-autoregressive passes. Retrieved codebook vectors are summed and decoded into a waveform. Evaluation separately measures perceptual quality, word accuracy, and speaker similarity; streaming capability must be assessed across the entire generation-and-decoding pipeline.

## Source and scope notes

Primary source: Jurafsky and Martin, Speech and Language Processing, Chapter 17, “Text-to-Speech,” August 19, 2026 draft, supplied as 17.pdf. Relevant anchors include Sections 17.1–17.5 and Figures 17.3, 17.6, and 17.7.

Toy vectors, worked shape examples, corrected questions, and streaming implications are explanatory additions from our study. These notes preserve the study’s technical substance without reproducing the chapter or its historical bibliography.

