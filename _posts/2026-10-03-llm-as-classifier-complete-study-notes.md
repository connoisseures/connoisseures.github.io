---
title: "Using LLMs as Classifiers: Complete Study Notes"
date: 2026-10-03 22:06:00 -0700
categories: [LLM, Interview Preparation]
tags: [classification, fine-tuning, lora, calibration, evaluation, model-serving]
description: A complete guide to LLM classification heads, label generation, training, calibration, data quality, evaluation, and production serving.
math: true
---
Study date: October 3, 2026  
Status: Core conceptual study and applied interview exercise completed.

## 1. Problem formulation

A text classifier maps an input to one or more task labels. Examples include intent detection, support routing, and judging whether a conversational turn is complete.

| Transcript | Example label |
|---|---|
| Set a timer for ten minutes. | Complete |
| Set a timer for… | Incomplete |
| I want to book a flight to… | Incomplete |

For speech endpointing, text alone can be ambiguous. Audio, pause duration, prosody, and conversational context may provide additional evidence. This study focused on the text classification mechanism.

Two main approaches are to generate a label with the existing language-model head or attach a dedicated classification head.

## 2. Dedicated classification head and tensor shapes

Let the Transformer produce contextual hidden states:

$$
H=\operatorname{LLM}(x)\in\mathbb{R}^{B\times T\times d}.
$$

Select or pool a sequence representation:

$$
h\in\mathbb{R}^{B\times d}.
$$

A linear classifier produces logits:

$$
z=hW+b,\qquad W\in\mathbb{R}^{d\times C},\quad b\in\mathbb{R}^{C}.
$$

For hidden size 4,096 and three classes:

$$
[B,4096]\,[4096,3]+[3]\longrightarrow[B,3].
$$

The 50,000-token vocabulary does not determine the classification head's output dimension. The task's three classes do. This uses the mathematical convention above; a linear-layer implementation may store the transposed weight matrix.

### Choosing a sequence representation

In a standard causal decoder, each token can attend only to itself and earlier positions. The last non-padding token can incorporate information from the full preceding input; the first token cannot access later tokens.

For `Set | a | timer | PAD | PAD`, select the hidden state at `timer`, not a padding position. Correct attention masks and position handling are still required. Other model families may use a designated classification token or pooling; the pooling strategy should match the architecture and training.

## 3. Single-label training

For mutually exclusive classes, apply softmax:

$$
p_c=\frac{\exp(z_c)}{\sum_j\exp(z_j)}.
$$

Example classes: 0 = timer, 1 = music, 2 = weather. For “Play some jazz,” the target is 1. If probabilities are [0.2, 0.5, 0.3], cross-entropy is:

$$
\mathcal{L}=-\log p_y=-\log(0.5)\approx0.693.
$$

Training raises the correct class's probability relative to alternatives. In numerical implementations, use a stable loss operating on logits.

## 4. Frozen features, full fine-tuning, and LoRA

| Approach | Parameters updated |
|---|---|
| Zero-shot or few-shot prompting | None |
| Frozen Transformer plus classification head | Head weights and bias |
| Full fine-tuning | All trainable model parameters, including the relevant output head |
| LoRA adaptation | Added adapter parameters; base weights remain frozen; a task head can also be trained |

With a frozen Transformer, training changes the mapping from hidden representations to class scores. It does not teach the Transformer new representations.

$$
z=h_{\text{fixed}}(x)W+b.
$$

Adapting the Transformer allows the representation to change too:

$$
z=h_\theta(x)W+b.
$$

For example, “The assistant cuts me off mid-sentence” and “The assistant waits too long after I finish” both concern timing, but correspond to early and late endpointing. Adaptation may make that distinction easier for the head to separate. Improvement is not guaranteed and must be checked on held-out data.

LoRA expresses an effective weight as:

$$
W_{\text{effective}}=W_{\text{frozen}}+\Delta W,
$$

where the update is parameterized through low-rank factors.

## 5. Classification by generating labels

A prompt can map labels to output codes:

```text
Classify the user's intent:
A = Set a timer
B = Play music
C = Check weather

User: Play some jazz
Answer:
```

The existing vocabulary head predicts the answer tokens. No new classification head is required. Adding examples to the prompt is few-shot prompting: the context changes, but weights do not.

If A, B, and C are each single tokens in this context, compare their logits. With scores 1, 3, and 0, B wins. Renormalizing over those allowed tokens gives:

$$
p(B\mid x,\text{allowed labels})=\frac{e^3}{e^1+e^3+e^0}\approx0.844.
$$

This is conditional on the allowed label set, not proof that the input actually belongs to a supported class. Constrained decoding can ensure valid output syntax, but cannot ensure correct classification.

### Multi-token labels

Suppose labels tokenize as:

```text
MUSIC          -> [MUSIC]
ACCOUNT_ACCESS -> [ACCOUNT, _ACCESS]
ACCOUNT_DELETE -> [ACCOUNT, _DELETE]
```

The first token cannot distinguish the two account classes. Score the entire label using the chain rule:

$$
\log P(y\mid x)=\sum_{t=1}^{m}\log P(y_t\mid x,y_{<t}).
$$

Longer labels introduce more probability factors, so label wording and tokenization affect comparisons. Use consistent completion boundaries when scoring complete outputs. Distinct class codes can simplify scoring, but verify their tokenization in the actual prompt context.

## 6. Fine-tuning a model to generate labels

Example training pair:

```text
Prompt: Classify as TIMER, MUSIC, or WEATHER.
        User: Play some jazz.
        Answer:
Target: MUSIC
```

The answer-token objective is:

$$
\mathcal{L}=-\sum_{t=1}^{m}\log P_\theta(y_t\mid x,y_{<t}).
$$

Typically, prompt positions are masked out of the loss so training scores the desired answer. However, answer predictions depend on the prompt computation. Gradients can flow through that computation into trainable Transformer parameters.

**Loss masking and parameter freezing are separate choices.** Computing loss only on labels does not restrict updates to the output head. A frozen Transformer stays fixed because its parameters are frozen, not because prompt tokens lack direct loss terms.

## 7. Unsupported inputs and abstention

If the only labels are TIMER, MUSIC, and WEATHER, “Send a message to Alice” has no correct supported label. Forcing a choice can produce a confident mistake.

- An OTHER class represents requests outside the supported categories.
- Abstention represents insufficient confidence to make a decision.

These are different mechanisms. OTHER also needs representative examples; it is not a guarantee of detecting every unfamiliar input. High softmax confidence alone cannot establish correctness.

## 8. Single-label versus multi-label classification

| Task | Output activation | Typical loss |
|---|---|---|
| Exactly one mutually exclusive class | Softmax | Multiclass cross-entropy |
| Zero or more applicable labels | Independent sigmoids | Binary cross-entropy across labels |

“Set a timer and play some jazz” may have target [1, 1, 0] for TIMER, MUSIC, WEATHER.

$$
p_c=\sigma(z_c)=\frac{1}{1+e^{-z_c}}.
$$

Outputs might be [0.95, 0.92, 0.03]; they need not sum to one. Each label is selected with a threshold, potentially specific to that class. Independent output sigmoids do not mean real-world labels are statistically unrelated.

## 9. Metrics, imbalance, and thresholds

If 990 of 1,000 requests are normal and ten require escalation, predicting normal every time gives 99% accuracy and zero escalation recall.

$$
\operatorname{Precision}=\frac{TP}{TP+FP},\qquad
\operatorname{Recall}=\frac{TP}{TP+FN}.
$$

Precision asks how many predicted positives are correct. Recall asks how many actual positives are detected.

Escalate when:

$$
p(\text{escalation}\mid x)\geq\tau.
$$

On a fixed set of scores, lowering the threshold increases or preserves recall, and increases or preserves the number of false positives. Precision often falls, but is not mathematically required to decrease at every threshold step.

When missing an escalation is more costly than extra review, a lower threshold may fit the objective. Select thresholds on validation data using error costs and operational capacity. For multiclass systems, specify how class-specific thresholds, competing classes, and abstention interact.

Evaluate per-class precision and recall, confusion matrices, and macro-F1 alongside overall accuracy. Macro-F1 gives each class equal weight by averaging class-level F1 scores.

## 10. Calibration

A well-calibrated escalation score near 0.8 means approximately 80% of comparable scored requests truly require escalation. For top-label confidence, calibration relates predicted confidence to the frequency of correct predictions.

A model can be accurate yet overconfident. Temperature scaling adjusts softmax probabilities:

$$
p_c=\operatorname{softmax}(z/T)_c,\qquad T>0.
$$

Fit T on held-out calibration/validation data while keeping model weights fixed.

- T greater than one softens the distribution.
- T below one sharpens it.
- A common positive T preserves class order and argmax.
- Threshold-based acceptance may change even though argmax does not.

Calibration does not repair incorrect class rankings and can degrade under distribution shift. The probability vectors used conversationally to illustrate softening were not an exact single-temperature pair; this note retains the exact mathematical property instead.

## 11. Data splits and leakage

Split according to the intended generalization claim:

| Goal | Split principle |
|---|---|
| New conversations | Keep a conversation in one split |
| New users | Keep a user in one split |
| Future traffic | Hold out later traffic and control duplicates |
| Augmented examples | Keep originals and their paraphrases together |

Near-duplicates across training and test data inflate evaluation. Select few-shot examples from training data, tune prompts and thresholds on validation data, and reserve the final test set for evaluation.

## 12. Label quality, rare intents, and hard negatives

A rare important intent with 40% recall is missing 60% of its true examples. High overall accuracy can hide that failure. Deployment depends on explicit quality requirements and error costs, not accuracy alone.

Before changing architectures, inspect false negatives and which classes receive them. Check:

1. Label definitions and consistency at class boundaries.
2. Diverse positive coverage for the rare intent.
3. Hard negatives: similar-looking examples that actually belong to another intent.
4. Missing context needed to distinguish the labels.

Example boundary:

| Request | Intent |
|---|---|
| Cancel my subscription. | Cancel subscription |
| I never authorized this subscription. | Unauthorized subscription |

If essentially identical requests receive conflicting labels, clarify guidelines and adjudicate disagreements. More data collected with the same inconsistent process will not reliably fix the ambiguity. If the input lacks necessary evidence, obtain context or support ambiguity.

Oversampling repeats rare-class examples more often. It increases their influence but creates no new linguistic diversity and may overfit a small set. Collecting varied real examples addresses coverage. Keep validation and test distributions representative of production; training rebalancing can also affect probability calibration.

## 13. Serving and classifier cascades

A cascade can use a smaller classifier first and route uncertain requests to a larger LLM:

1. Run the smaller classifier.
2. Accept predictions satisfying a validated confidence rule.
3. Route remaining inputs to the larger model or another fallback.

Raising the small model's acceptance threshold sends more requests to the larger model. This typically raises serving cost. Escalated requests incur both stages' computation, so measure end-to-end latency, including tail latency and queueing.

The large model must be evaluated on the difficult routed subset; superiority is not automatic. Compare overall quality, coverage, routing rate, cost, and latency.

## 14. Production monitoring

Monitor operational performance and quality together:

- Latency, cost, traffic volume, and escalation rate.
- Per-class precision and recall on fresh representative labeled traffic.
- Unsupported requests and changing class frequencies.
- Input length, terminology changes, and truncation.

Average confidence staying at 0.95 does not establish stable accuracy. New labels or other reliable outcome evidence are needed to measure correctness.

## 15. Applied interview exercise

### Requirements

Classify support messages into 20 mutually exclusive intents, with 50,000 labeled examples, a fixed label set, and a tight latency budget.

### Baseline

Start with a smaller pretrained text model plus a 20-class classification head. Fine-tune using supervised data and assess whether its quality meets requirements. A smaller model typically reduces inference cost and latency, but verify performance under realistic load.

### Evaluation and iteration

Use leakage-resistant splits; measure macro-F1, per-class precision and recall, and p95 latency. If an important rare intent has 40% recall, inspect its errors before deployment. Improve label consistency, positive coverage, and hard negatives. Consider training rebalancing without distorting production-oriented evaluation. Validate fallbacks and decision thresholds.

### Consolidated interview answer

> For 20 fixed intents, 50,000 labeled examples, and tight latency requirements, I would start with a smaller pretrained text model and a classification head. I would prevent leakage through appropriate data splits and evaluate per-class precision and recall, macro-F1, and production-like latency. If an important rare intent has poor recall, I would inspect false negatives, label consistency, positive coverage, and hard negatives before changing architectures. I would address imbalance during training while keeping evaluation representative of production, validate confidence thresholds and fallbacks, and monitor quality using newly labeled traffic.

## 16. Concept-check answer key

| Check | Correct answer |
|---|---|
| Head shape for d = 4,096 and C = 3 under z = hW + b | [4096, 3] |
| Can the first token of a causal decoder incorporate later tokens? | No |
| Do token embeddings update when the Transformer is frozen? | No |
| Does few-shot prompting update model weights? | No |
| Does 0.98 confidence guarantee a supported, correct class? | No |
| Sports and politics may both apply: softmax or sigmoids? | Independent sigmoids |
| Missing positives is costly: raise or lower the positive threshold? | Generally lower it |
| Does positive scalar temperature scaling change argmax? | No |
| Can label-only loss update a trainable Transformer? | Yes |
| Frozen Transformer: what does head training change? | Mapping from representations to class scores |
| How should original examples and paraphrases be split? | Keep each group in one split |
| Can a shared first label token distinguish two labels? | No |
| Higher small-model acceptance threshold: more large-model requests? | Yes |
| Does stable confidence establish stable accuracy? | No |
| Does more inconsistently labeled data reliably fix disagreement? | No |
| Does oversampling add linguistic diversity? | No; it increases existing examples' influence |

## 17. Completion and next steps

The conceptual study and applied interview exercise are complete. A hands-on training and evaluation implementation remains optional and was not performed in this study.

These notes consolidate the conversation, including clarified distinctions and corrections. No empirical model benchmark was run.
