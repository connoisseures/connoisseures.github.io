---
title: "Speech and Language Processing Chapter 16: Automatic Speech Recognition"
date: 2026-09-23 00:00:00 +0000
categories: [Speech LLM, ASR]
tags: [speech-recognition, cnn, encoder-decoder, whisper, hubert, ctc, rnnt, streaming, word-error-rate, endpointing]
description: Complete Chapter 16 study notes covering convolutional front ends, encoder-decoder ASR, HuBERT, CTC, RNN-T, word error rate, and practical endpoint detection, with equations and corrected concept checks.
math: true
---

**Book:** Speech and Language Processing, 3rd edition draft  
**Authors:** Daniel Jurafsky and James H. Martin  
**Source studied:** Uploaded 16.pdf, Chapter 16, draft dated August 19, 2026  
**Study dates:** September 22–23, 2026  
**Scope:** Sections 16.1–16.7, worked examples, corrected concept checks, and an applied extension on endpoint detection.

These are explanatory study notes, not a reproduction of the chapter. Numerical examples are illustrative unless explicitly attributed to the chapter. Model dimensions and dataset details describe the studied draft or examples, not universal settings or current benchmark rankings.

## Executive summary

Automatic speech recognition (ASR) maps audio to text. Audio usually contains many more time steps than the corresponding text, and a transcript normally does not provide frame-level timing.

- Convolutional front ends learn local acoustic patterns and can reduce sequence length.
- Attention-based encoder–decoder models generate text using both encoded audio and previous text.
- HuBERT learns audio representations by predicting automatically generated cluster IDs at masked positions.
- CTC trains from paired audio and transcripts without manually labeled frame-to-symbol alignments.
- RNN-T combines audio representations and output-token history, and can support streaming when its encoder permits it.
- Word error rate (WER) counts substitutions, deletions, and insertions relative to the reference length.
- A blank output is not evidence by itself that the user has finished speaking.

## 1. The ASR task — §16.1

The recognition goal is:

$$
\hat{Y}=\arg\max_Y P(Y\mid X)
$$

Here, X is audio and Y is a candidate text sequence.

### What changes the difficulty?

| Dimension | Easier example | Harder example |
|---|---|---|
| Vocabulary | Fixed yes/no commands | Open-ended conversation |
| Speaking style | Carefully read speech | Hesitations, restarts, casual conversation |
| Channel and noise | Close microphone in a quiet room | Far-field, reverberation, traffic, competing speech |
| Speaker coverage | Speakers represented in training | Underrepresented accents, dialects, or ages |

A model that succeeds on audiobooks may fail on assistant requests because the speaking style, recording conditions, vocabulary, and speaker distribution differ.

**Background speech versus environmental noise:** another voice may cause speaker confusion or mixed transcripts; non-speech noise can mask cues that distinguish speech sounds.

Noise augmentation helps with acoustic robustness, but adding traffic noise to an audiobook does not create spontaneous speaking style. Representative conversational data and appropriate channel, speaker, and noise coverage remain important.

### Datasets introduced in the chapter

| Dataset | Main setting |
|---|---|
| LibriSpeech | Read audiobooks |
| Switchboard | Telephone conversations between strangers |
| CALLHOME | Informal telephone conversations, often friends or family |
| CHiME-6 | Dinner-party speech in real homes with distant microphones |
| AMI | Meetings |
| CORAAL | Sociolinguistic interviews with African American speakers |
| HKUST | Mandarin telephone conversations |
| AISHELL-1 | Mandarin read speech |
| Common Voice | Crowdsourced multilingual read speech |
| FLEURS | Multilingual read speech based on parallel text |

LibriSpeech “clean” and “other” were formed using recognition error rates by speaker; they are not simply clean-audio versus noisy-audio labels. Benchmark performance is specific to its data and evaluation protocol.

## 2. Convolutional front ends — §16.2

A convolution slides learned weights over local input windows. The same weights are reused at every position.

For a scalar example, use input [1, 2, 3, 4, 5] and kernel [1, 0, -1], without padding and with stride 1:

$$
z_1=1(1)+2(0)+3(-1)=-2
$$

The next two windows also produce -2. This kernel measures a local difference; training learns kernels useful for speech.

Neural-network “convolution” usually implements cross-correlation, without reversing the kernel. Learned weights make this naming convention unproblematic.

### Channels and tensor shapes

For an ordinary dense 1D convolution (groups = 1):

| Tensor | Shape | Interpretation |
|---|---|---|
| Input | [B, C_in, T] | Batch, input channels, time |
| Weights | [C_out, C_in, k] | Output channels, input channels, kernel width |
| Bias | [C_out] | One bias per output channel |
| Output | [B, C_out, T_out] | Learned features at output positions |

With 80 Mel channels, width 3, and 256 output channels, each output kernel has shape [80, 3]. Each output scalar combines 240 input values. All 256 kernels together produce a 256-dimensional vector per output position.

For stride s, padding p, and dilation 1:

$$
Z_{b,o,t}=a_o+\sum_{c=0}^{C_{\mathrm{in}}-1}\sum_{r=0}^{k-1}
W_{o,c,r}X_{b,c,ts+r-p}
$$

Out-of-range input values are zero under zero padding. The kernel slides along time, while summing over channels.

### Output length

For dilation 1 and symmetric padding:

$$
T_{\mathrm{out}}=\left\lfloor\frac{T+2p-k}{s}\right\rfloor+1
$$

| Input | Output channels | Width | Padding | Stride | Output |
|---|---:|---:|---:|---:|---|
| [2, 80, 100] | 256 | 3 | 1 | 2 | [2, 256, 50] |
| [2, 80, 100] | 128 | 3 | 1 | 2 | [2, 128, 50] |
| [2, 80, 100] | 128 | 3 | 1 | 1 | [2, 128, 100] |

Changing the number of kernels changes output channels, not time length. Stride controls how far the window moves; stride 2 usually roughly halves the sequence, with exact rounding determined by the formula.

### Receptive field, compute, and causality

With 25 ms feature windows and 10 ms frame spacing, three adjacent frames span about 45 ms of waveform, not 30 ms.

Halving the sequence length reduces a full self-attention score matrix from T² to T²/4 entries. This does not imply a fourfold speedup for the entire model.

A centered width-3 kernel uses positions t-1, t, and t+1. It needs future context. A causal convolution uses only current and past positions. Feature extraction itself can also introduce buffering or lookahead.

## 3. Attention-based encoder–decoder ASR — §16.3

The basic information flow is: acoustic features, convolutional subsampling, audio encoder, and text decoder with cross-attention.

### Audio encoder

An illustrative shape trace is:

$$
[B,80,100]\rightarrow[B,256,50]\rightarrow[B,50,256]
$$

The final transformation transposes the convolution output into a Transformer-friendly layout. The encoder produces:

$$
H_{\mathrm{enc}}\in\mathbb{R}^{B\times50\times256}
$$

These 50 positions are continuous audio representations, not predicted words. Encoder attention can contextualize each position using other audio positions.

### Text decoder

Using whole words only for illustration:

| Available text | Next target |
|---|---|
| SOS | turn |
| SOS turn | left |
| SOS turn left | EOS |

Actual models can use characters or subword tokens. For U output tokens including EOS:

$$
P(Y\mid X)=\prod_{u=1}^{U}P(y_u\mid y_{<u},X)
$$

The decoder uses both prior text and encoded audio. Audio and text lengths need not match.

### Self-attention versus cross-attention

| Layer | Information accessed |
|---|---|
| Encoder self-attention | Audio representations within the input segment |
| Decoder causal self-attention | Current and preceding decoder input positions |
| Decoder cross-attention | Audio encoder outputs |

“Self” means the same sequence; it does not necessarily mean text.

For one cross-attention head:

$$
Q=H_{\mathrm{dec}}W_Q,\qquad K=H_{\mathrm{enc}}W_K,\qquad V=H_{\mathrm{enc}}W_V
$$

$$
\operatorname{Attention}(Q,K,V)=
\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

The softmax runs over audio positions. For one text position and 50 audio positions, omitting batch and head axes:

| Tensor | Shape |
|---|---|
| Q | [1, d_k] |
| K | [50, d_k] |
| V | [50, d_v] |
| Attention scores/weights | [1, 50] |
| Retrieved vector | [1, d_v] |

### Whisper-style front end and OWSM

The chapter illustrates an audio context of 30 seconds and a 10 ms feature hop: approximately 3000 feature frames. Two convolutional layers, with strides 1 and 2, reduce this to approximately 1500 encoder positions, spaced 20 ms apart. Positional information is added before the Transformer.

Channel counts and model dimensions vary by model configuration. “Audio token” in this front-end discussion means a continuous representation, not necessarily a discrete codebook ID.

Whisper also uses task and language tokens for its multitask format. OWSM is presented as an open Whisper-style system with publicly available training resources.

### Training and teacher forcing

For the reference “turn left,” shift the text by one position:

| Decoder input | Target |
|---|---|
| SOS | turn |
| turn | left |
| left | EOS |

Causal masking prevents access to future decoder inputs. During teacher forcing, the history contains correct reference tokens:

$$
\mathcal{L}_{\mathrm{CE}}
=-\sum_{u=1}^{U}\log P(y_u^*\mid y_{<u}^*,X)
$$

If the model predicts “right” instead of “left,” it receives a loss, but the next training position still receives the reference “left.” At inference, no reference exists, so selected generated tokens form the history.

This mismatch is called exposure bias. The chapter also mentions mixing predicted and reference history as a possible training variation; this is distinct from ordinary teacher forcing.

For a correct token probability of 0.8, negative log loss is about 0.22; for probability 0.1 it is about 2.30. End-to-end training propagates gradients through decoder, encoder, and trainable front end.

### Inference and language-model rescoring

- Greedy decoding chooses the most probable next token.
- Beam search retains several partial candidates.
- External language-model rescoring combines ASR and text-model scores for an n-best list, with tuned weighting and an appropriate length adjustment.

The single best local choice need not yield the best complete sequence.

### Autoregressive is not the same as streaming

Causal text decoding restricts future text access. Streaming audio processing restricts future audio requirements.

An encoder requiring all 30 seconds cannot provide that full-segment encoding after only one second. Encoding the first second separately can provide a provisional result, but it is a different input and can require revision.

## 4. HuBERT self-supervised pretraining — §16.4

HuBERT learns from untranscribed audio through **cluster, mask, predict**.

### Two paths during pretraining

| Path | Processing | Output |
|---|---|---|
| Target creation | Original audio → acoustic features → k-means | Cluster ID per target position |
| Prediction | Audio → CNN → masked features → Transformer | Distribution over cluster IDs |

For example:

| Position | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| Transformer input | Feature | Feature | MASK | MASK | Feature |
| Target ID | 12 | 12 | 7 | 7 | 35 |

Targets are derived from unmasked audio. Masking the model input does not erase the target.

Cluster IDs are automatically induced acoustic categories, not written words and not guaranteed one-to-one phoneme labels.

### Why mask spans?

Visible features could allow a shortcut: classify the local frame without learning much context. Masking encourages inference from surrounding speech. Consecutive frames are similar, so masking spans makes simple local copying less effective.

The Transformer’s representation is projected and compared with learned class embeddings using cosine similarity:

$$
p(c\mid\widetilde{X},t)=
\frac{\exp(\operatorname{sim}(Ah_t,e_c)/\tau)}
{\sum_{c'=1}^{C}\exp(\operatorname{sim}(Ah_t,e_{c'})/\tau)}
$$

Here A is the learned projection, e_c is the embedding of cluster c, and the chapter uses temperature 0.1. For masked positions M:

$$
\mathcal{L}_{\mathrm{mask}}=-\sum_{t\in M}\log p(z_t\mid\widetilde{X},t)
$$

**Notation correction:** Eq. 16.13 in the uploaded draft omits the minus sign. The negative log-likelihood above is minimized; the figure’s negative-log labels are consistent with it.

### Improve targets through reclustering

The chapter’s described configuration uses:

| Stage | Clustered features | Number of clusters |
|---|---|---:|
| First | 39-dimensional MFCCs | 100 |
| Second | 768-dimensional hidden representations from Transformer layer 6 | 500 |

The important improvement is the learned feature representation, not merely a larger cluster count. Reclustering uses a subsample of data in the illustrated second stage.

### K-means reminder

Assign each vector to its nearest centroid:

$$
z_t=\arg\min_{j\in\{1,\ldots,K\}}\|v_t-\mu_j\|_2^2
$$

Then update each centroid:

$$
\mu_j=\frac{1}{|S_j|}\sum_{v\in S_j}v
$$

Repeat assignment and update. Centroids need not be actual observed vectors. The squared Euclidean distance is the sum of squared coordinate differences; the Euclidean norm itself includes a square root.

### Front end and ASR fine-tuning

The chapter describes seven CNN layers with strides [5, 2, 2, 2, 2, 2, 2]. Their product is 320. At 16 kHz, this gives a nominal output spacing of 20 ms; exact output length also depends on kernels and padding.

For ASR, replace the cluster-prediction head with a text-symbol classifier including CTC blank, and fine-tune on paired audio/transcripts with CTC loss. The configuration described in the chapter freezes the CNN and updates the Transformer and new head.

**Pretraining needs audio only; supervised ASR fine-tuning needs audio paired with transcripts.**

## 5. CTC — §16.5

Connectionist Temporal Classification handles long audio sequences, short transcripts, and unknown frame-to-symbol alignment.

A transcript “cat” tells us the symbol order, not where each sound begins or ends. Saying “cat” slowly changes timing, not its ordinary written transcript.

### Collapse rule

At each encoder position, predict a text symbol or blank, represented here by ∅.

1. Merge consecutive identical symbols.
2. Remove blanks.
3. Do not merge again after blank removal.

| Path | After collapse |
|---|---|
| c c ∅ a a ∅ t t | cat |
| a a | a |
| a ∅ a | aa |
| b o o k | bok |
| b b o ∅ o k k | book |

Blank means no text emission at that position. It is not a word space and does not necessarily mean acoustic silence.

### Path versus transcript probability

CTC assumes output labels are conditionally independent given the audio:

$$
P(A\mid X)=\prod_{t=1}^{T}p_t(a_t\mid X)
$$

The transcript probability sums all paths that collapse to Y:

$$
P(Y\mid X)=\sum_{A:B(A)=Y}P(A\mid X)
$$

Remember: **multiply within a path; add across valid paths.**

A contextual encoder can inspect surrounding audio. Conditional output independence does not mean isolated-frame input processing.

### Greedy decoding versus transcript search

Greedy decoding chooses the highest-probability symbol at each time step and collapses the resulting path. It finds a best path under the factorization, but not necessarily the best transcript.

Illustrative aggregated probabilities:

| Transcript | Individual path probabilities | Total |
|---|---|---:|
| cat | 0.30, 0.05 | 0.35 |
| cap | 0.20, 0.18, 0.12 | 0.50 |

The highest individual path gives “cat,” while the larger transcript total gives “cap.” The remaining probability mass can belong to other transcripts.

Beam search approximates transcript search by maintaining promising prefixes and aggregating compatible path probabilities. An external language model can supply explicit text-history preferences; it is useful in many configurations but not a mathematical requirement for CTC.

### Training and dynamic programming

During training, the transcript is already given:

$$
\mathcal{L}_{\mathrm{CTC}}=-\log P(Y_{\mathrm{reference}}\mid X)
$$

Unknown alignment does not mean unknown transcript. Dynamic programming computes the exact alignment sum for a fixed target without enumerating all paths, tracking progress through the target and blank states.

| Stage | Transcript | Alignment |
|---|---|---|
| Training | Given | Marginalized over valid paths |
| Inference | Must be predicted | Also unknown |

### CTC head versus autoregressive decoder

A standard CTC head predicts:

$$
p_t=\operatorname{softmax}(Wh_t+b)
$$

It does not receive the previously predicted letter. Once encoder states are available, head predictions can be computed in parallel.

An autoregressive decoder explicitly receives preceding text tokens. This is the core distinction, even though both can use contextual audio representations.

### Joint CTC/attention training

A shared encoder can feed both a CTC head and an attention-based decoder:

$$
\mathcal{L}=\lambda\mathcal{L}_{\mathrm{AED}}+(1-\lambda)\mathcal{L}_{\mathrm{CTC}}
$$

Both losses train the shared representations. Inference may combine their scores and optionally a language-model score.

## 6. RNN-T and streaming — §16.5.4

RNN-Transducer introduces explicit output-token history into an alignment-based recognizer.

| Component | Input | Role |
|---|---|---|
| Audio encoder | Audio | Acoustic representation |
| Predictor | Previous non-blank output tokens | Text-history representation |
| Joint network | Encoder and predictor states | Distribution over tokens and blank |

A generic formulation is:

$$
p(k\mid t,u)=\operatorname{softmax}
\left(J(h_t^{\mathrm{enc}},g_u^{\mathrm{pred}})\right)_k
$$

Here t indexes encoder positions and u counts emitted non-blank tokens. The predictor summarizes the first u output tokens.

### Two kinds of progress

| Emission | Audio index | Text history |
|---|---|---|
| Non-blank token | Stay at current position | Append token; advance u |
| Blank | Advance t | Unchanged |

Example:

| Event | Transcript afterward |
|---|---|
| Initial blank | Empty |
| Emit turn | turn |
| Emit blank | turn |
| Emit left | turn left |

**Blank = advance audio, add no text.** If the transcript is “hello,” a blank leaves it “hello.” It adds no word, letter, or space.

RNN-T does not use CTC’s merge-adjacent-duplicates rule: every non-blank token emission contributes to the output sequence. It can emit multiple tokens at one audio position.

Training sums over valid transducer alignments. These paths live on an audio/text grid and differ from CTC’s one-label-per-frame paths.

### Streaming requirements

Streaming requires an encoder whose future-audio dependence is bounded:

- Causal encoder: current and past audio.
- Limited-lookahead encoder: bounded future audio, adding delay.
- Full-segment encoder: must wait for the required segment.

Autoregressive token generation and streaming audio input are separate properties. Neither CTC nor RNN-T automatically makes a full-context encoder causal.

Avoid treating the chapter’s model-family accuracy comparisons as universal rankings: performance depends on training data, architecture, decoding, latency constraints, and evaluation.

## 7. ASR evaluation — §16.6

Word error rate compares reference and hypothesis using a minimum-edit alignment.

$$
\mathrm{WER}=100\%\times\frac{S+D+I}{N}
$$

S is substitutions, D deletions, I insertions, and N the number of reference words.

| Error | Reference | Hypothesis |
|---|---|---|
| Substitution | turn left | turn right |
| Deletion | turn left now | turn left |
| Insertion | turn left | please turn left |

### Worked examples

| Example | Counts | WER |
|---|---|---|
| Four reference words; one substitution | S=1, D=0, I=0, N=4 | 25% |
| turn left now → turn left | D=1, N=3 | 33.3% |
| turn left → please turn left | I=1, N=2 | 50% |
| Exact match of a nonempty reference | No errors | 0% |

For “turn left at the next street” → “turn right at the next street please,” there is one substitution and one insertion over six reference words: 33.3%.

The denominator is reference length, not hypothesis length. Insertions can make WER exceed 100%. Empty-reference handling requires a specified evaluation convention.

### Corpus WER

$$
\mathrm{WER}_{\mathrm{corpus}}
=100\%\times\frac{\sum_i(S_i+D_i+I_i)}{\sum_i N_i}
$$

Sum counts before dividing. A simple average of sentence WERs gives short and long sentences equal weight and is not the usual corpus WER.

Character error rate uses analogous character-level edits. Sentence error rate is the fraction of sentences with at least one word error.

### Normalization

Apply the same task-appropriate rules to both reference and hypothesis. Numbers, contractions, case, punctuation, spelling variants, and fillers can affect scores.

“Two dollars” and a numeric currency form can convey the same spoken content. However, removing “um” is not appropriate when evaluating verbatim transcription that requires fillers. Record normalization rules with results.

### Statistical significance

A test WER improvement, such as 10.0% to 9.8%, is not by itself proof of a reliable gain.

The chapter introduces the Matched-Pair Sentence Segment Word Error test (MAPSSWE). For matching segments, define:

$$
Z_i=E_{A,i}-E_{B,i}
$$

Compute the mean difference and sample variance:

$$
\bar Z=\frac{1}{n}\sum_{i=1}^{n}Z_i,\qquad
s_Z^2=\frac{1}{n-1}\sum_{i=1}^{n}(Z_i-\bar Z)^2
$$

The standardized statistic is:

$$
W=\frac{\bar Z}{s_Z/\sqrt{n}}
$$

Interpretation depends on the test assumptions, including sufficiently independent segments and an appropriate approximation for the sample size. Neighboring word errors are correlated, so treating every word as an independent outcome can be misleading. The chapter discusses this problem for word-level uses of McNemar’s test.

### Limits of WER

WER weights every word error equally. Confusing “left” and “right” may change an action, while a filler error may have little task impact.

For assistant evaluation, supplement WER with task/intent correctness, latency, and slices by accent, speaker group, channel, noise, and speaking style. These are practical extensions, not replacements for documenting WER.

## 8. Applied extension: endpoint detection and turn-taking

This section extends the chapter to conversational systems and our study discussion.

| Component | Question |
|---|---|
| ASR | What words has the user said? |
| VAD | Is speech present in this interval? |
| Endpoint detector | Has the user finished their contribution? |

An RNN-T blank is neither a silence label nor an end-of-turn decision. Blanks can occur while speech is ongoing.

Consider:

> Set a timer for … [pause] … ten minutes.

During the pause, “set a timer for” is linguistically incomplete. An endpoint detector should combine acoustic evidence with linguistic and conversational context rather than treat a blank as completion.

With equal silence:

| Utterance | Linguistic evidence |
|---|---|
| Set a timer for… | Likely continuation |
| Set a timer for ten minutes. | Stronger evidence of completion |

Completeness is evidence, not a guarantee: users can append details or self-correct.

### Latency versus cutoff trade-off

| Silence threshold | Typical effect, all else equal |
|---|---|
| Shorter | Faster response, greater premature-cutoff risk |
| Longer | Slower response, lower premature-cutoff risk |

A longer threshold gives a user more time to continue but delays responses after genuinely completed turns. A contextual endpoint model can make more informed decisions than a fixed silence threshold alone.

## 9. Corrected concept checks

These answers consolidate the study without repeating unsuccessful attempts.

1. **Why might LibriSpeech performance not transfer to assistants?** Noise, competing speakers, channel differences, spontaneous speaking style, vocabulary, and speaker-distribution mismatch.
2. **Does noise augmentation solve all domain mismatch?** No; it does not automatically supply missing conversational or speaker diversity.
3. **Input [2,80,100], 256 kernels, width 3, padding 1, stride 2: output?** [2,256,50].
4. **Change only kernels to 128: output?** [2,128,50].
5. **Then change stride to 1: output?** [2,128,100].
6. **How does the decoder access audio?** Cross-attention; Q comes from the decoder, K and V from the encoder.
7. **Does causal text decoding guarantee streaming audio?** No; inspect encoder and feature-extraction context requirements.
8. **Teacher forcing after a wrong prediction: what token is fed next?** The correct reference token.
9. **HuBERT pretraining target?** Automatically generated acoustic cluster IDs.
10. **Why mask spans?** Encourage prediction from contextual speech rather than visible local features.
11. **Does HuBERT ASR fine-tuning require transcripts?** Yes, in the supervised setup studied.
12. **Does “cat” supply exact sound boundaries?** No; it provides text and order, not alignment.
13. **CTC collapse of b b o ∅ o k k?** book.
14. **CTC training: best path or sum of valid paths?** Sum all valid paths for the known reference.
15. **Within one CTC path, combine probabilities how?** Multiply across frames.
16. **Does the CTC head consume previous predicted letters?** No. An autoregressive text decoder does.
17. **Which RNN-T component receives output-token history?** The predictor.
18. **What does an RNN-T blank do?** Advance audio without changing the transcript.
19. **Does a blank alone mean the user has finished?** No.
20. **Missing reference word: insertion or deletion?** Deletion.
21. **One error in four reference words: WER?** 25%.
22. **Perfect transcript match: WER?** 0%.
23. **Longer silence threshold: typical trade-off?** Fewer premature cutoffs, but slower responses.

## 10. Rapid-review checklist

- [ ] Separate batch, channel, and time dimensions in convolution.
- [ ] Derive output time length from kernel, stride, and padding.
- [ ] Explain Q/K/V sources in cross-attention.
- [ ] Distinguish teacher forcing from inference history.
- [ ] Explain HuBERT’s target-creation and masked-prediction paths.
- [ ] Apply CTC’s collapse order correctly.
- [ ] Distinguish path probability from transcript probability.
- [ ] Explain why CTC needs transcripts but not frame boundaries.
- [ ] Distinguish CTC from autoregressive decoding.
- [ ] Recall: RNN-T encoder = audio; predictor = token history.
- [ ] Recall: RNN-T blank = advance audio, no new text.
- [ ] Check encoder causality separately from output generation.
- [ ] Compute WER using reference length and total dataset counts.
- [ ] Keep blank, silence, and end-of-turn decisions separate.

## Source and scope notes

Primary source: Jurafsky and Martin, Speech and Language Processing, 3rd edition draft, Chapter 16, “Automatic Speech Recognition,” uploaded 16.pdf dated August 19, 2026. Section references throughout refer to that supplied version.

The notes preserve our complete conceptual walkthrough and examples, while correcting mathematical notation and avoiding overgeneralizations. The dynamic-programming discussion is conceptual; we did not implement the CTC forward-backward recurrence. Endpoint detection is explicitly an applied extension. Incorrect learner answers are replaced by corrected explanations suitable for later review.
