---
title: "Embedding Learning and Embeddings in LLMs: Complete Study Notes"
date: 2026-10-03 16:25:00 -0700
categories: [LLM, Embeddings]
tags: [embedding-learning, contrastive-learning, retrieval, reranking, llm, interview-preparation]
description: "A complete study of token embeddings, LLM gradient updates, weight tying, contrastive learning, retrieval evaluation, and production serving."
math: true
---

**Study date:** October 3, 2026  
**Source:** Our guided study conversation.  
**Scope:** Embedding fundamentals, LLM training, contrastive retrieval training, evaluation, and production serving.

## 1. What an embedding is

An embedding represents an item, such as a token, user, product, query, or document, as a vector. The training objective determines which information the representation preserves.

Token IDs are arbitrary category identifiers. Assigning cat = 0, dog = 1, and apple = 2 does not imply meaningful numerical distances or an ordering among those concepts. Integer IDs are valid inputs to an embedding lookup, but should not be interpreted as scalar semantic features.

A vocabulary of V tokens with embedding dimension D uses a trainable table:

$$
E \in \mathbb{R}^{V \times D}, \qquad e_i = E[i].
$$

Tokenization produces integer IDs. The embedding layer maps each ID to a vector. The ID remains fixed; the vector is learned. Individual coordinates usually have no explicit human-assigned meaning.

## 2. How embeddings learn

Embeddings are often initialized randomly and optimized jointly with a downstream model.

For “The cat ___” with target “sleeps”:

1. Look up the input embeddings.
2. Run the prediction model.
3. Compute a loss against the correct target.
4. Backpropagate into the model and embedding table.
5. Update trainable parameters.

With basic gradient descent:

$$
E[i] \leftarrow E[i] - \eta \frac{\partial \mathcal{L}}{\partial E[i]}.
$$

Words appearing in similar contexts receive related learning signals, which can lead to similar embeddings. However, reducing prediction loss does not guarantee that their vectors become close: prediction loss directly rewards correct predictions, not embedding proximity.

**Interview answer:** Words used in similar contexts need to support similar predictions. Shared model parameters can encourage related representations, but prediction loss does not explicitly enforce geometric closeness.

## 3. Three representations to distinguish

| Representation | Produced by | Meaning |
|---|---|---|
| Input token embedding | Lookup in a trainable table | Token identity |
| Contextual token representation | Transformer layers | A token in its available context |
| Sentence/document embedding | Pooling or selection, optionally projection | A whole text or chunk |

The token “bank,” assuming the same token ID, starts with the same lookup vector in “river bank” and “bank account.” Transformer layers can distinguish the meanings using context. A causal LLM can attend only to the current and preceding positions.

The embedding table is a **parameter**. Lookup outputs and contextual representations are **activations** computed for a particular input.

## 4. Tensor shapes and parameter sharing

Let vocabulary size V = 50,000, hidden dimension D = 768, batch size B = 2, and sequence length T = 10.

| Tensor | Shape |
|---|---|
| Embedding table | [50,000, 768] |
| Input token IDs | [2, 10] |
| Lookup output X | [2, 10, 768] |
| Transformer output H | [2, 10, 768] |

$$
X = E[\text{input\_ids}], \qquad H = \operatorname{Transformer}(X).
$$

Repeated occurrences of a token share one row. If token ID 7 appears at three positions:

$$
X_{1,2}=E[7], \qquad X_{1,5}=E[7], \qquad X_{2,3}=E[7].
$$

The lookup-path gradient contributions accumulate:

$$
\frac{\partial \mathcal{L}}{\partial E[7]}
=
\frac{\partial \mathcal{L}}{\partial X_{1,2}}
+
\frac{\partial \mathcal{L}}{\partial X_{1,5}}
+
\frac{\partial \mathcal{L}}{\partial X_{2,3}}.
$$

This equation describes the lookup contributions; a tied output projection adds another gradient path. Loss averaging is already reflected in the gradients.

If “cat” occurs 20 times and D = 64, its embedding row still contains **64 trainable parameters**, not 20 × 64. Its contextual activations may differ by occurrence.

## 5. Next-token prediction trains LLM embeddings

For “The cat sleeps”:

| Input position | Target |
|---|---|
| The | cat |
| cat | sleeps |

Using a column-vector convention and omitting output bias:

$$
h_t \in \mathbb{R}^{D}, \qquad
W_{\text{out}} \in \mathbb{R}^{V \times D},
\qquad z_t = W_{\text{out}}h_t \in \mathbb{R}^{V}.
$$

$$
p_t = \operatorname{softmax}(z_t), \qquad
\mathcal{L}_t = -\log p_t[y_t].
$$

At the “cat” position:

$$
\mathcal{L}_t = -\log P(\text{sleeps} \mid \text{The cat}).
$$

Gradients flow through the output projection and transformer into input embeddings. Since the prediction can depend on both “The” and “cat,” both rows can receive gradients.

The correct next token comes directly from text, making this self-supervised learning. Human-provided target vectors and a separate embedding-training stage are unnecessary.

## 6. Weight tying

Some LLMs share input and output embedding weights:

$$
W_{\text{out}} = E.
$$

The same row has two roles:

- Input: represent token i using E[i].
- Output: score token i using its dot product with the contextual state.

$$
z_{t,i} = E[i]^\top h_t.
$$

This is a dot product, not necessarily cosine similarity.

Sharing two V × D matrices saves one such matrix:

$$
50{,}000 \times 768 = 38{,}400{,}000
$$

parameters in our example.

With a tied matrix and standard full-vocabulary softmax, tokens absent from the input can receive gradients through the output projection, because all vocabulary tokens participate in prediction.

## 7. From token states to text embeddings

An LLM produces H with shape [B, T, D]. Retrieval typically needs one vector per sequence, with shape [B, D].

**Mean pooling** averages non-padding states. With mask m:

$$
e = \frac{\sum_{t=1}^{T} m_t h_t}{\sum_{t=1}^{T} m_t}.
$$

**Last-token pooling** selects the last non-padding state. In a causal LLM, that position can attend to the full preceding sequence, whereas earlier positions have partial context.

Pooling gives the desired shape but does not guarantee useful retrieval geometry. A next-token objective does not directly optimize cosine similarity between relevant query–document pairs.

## 8. Contrastive learning for retrieval

For each query, train relevant documents to score higher than irrelevant documents.

$$
q = f_\theta(\text{query}), \qquad
d = g_\phi(\text{document}).
$$

The encoders may share weights. Let s be similarity, d+ a relevant document, and d- negative documents:

$$
\mathcal{L}_q =
-\log
\frac{\exp(s(q,d^+)/\tau)}
{\exp(s(q,d^+)/\tau)+\sum_{j=1}^{K}\exp(s(q,d_j^-)/\tau)}.
$$

The positive is the correct class among candidate documents; similarities act as logits after temperature scaling.

Similarity is the **score**. Contrastive loss is the **optimization objective**.

If a positive and two negatives have equal scores, the positive probability is 1/3. Training encourages a higher positive score relative to the negatives.

### Embedding collapse

If an objective only rewards positive-pair similarity without negatives or another anti-collapse mechanism, mapping every text to the same vector can satisfy it:

$$
f(\text{query}) = f(\text{password guide}) = f(\text{recipe}) = [1,0,0].
$$

All cosine similarities are 1, so retrieval cannot distinguish documents. Negatives discourage this trivial solution. Other learning methods can prevent collapse without explicit negatives, but need additional mechanisms.

## 9. In-batch negatives

A batch contains matching pairs (q1, d1), (q2, d2), and (q3, d3). Encode and normalize them:

$$
Q,D \in \mathbb{R}^{3 \times d}, \qquad S=QD^\top \in \mathbb{R}^{3 \times 3}.
$$

The diagonal contains labeled positives. Off-diagonal entries are treated as negatives in the simple one-positive setup.

$$
\mathcal{L}_i =
-\log \frac{\exp(S_{ii}/\tau)}
{\sum_{j=1}^{3}\exp(S_{ij}/\tau)}.
$$

This reuses documents already encoded in the batch.

### False negatives

Another query’s positive document may also answer the current query. For example, “Reset my password” and “I forgot my password” may have interchangeable relevant guides.

Possible responses:

- Treat all known relevant documents as positives.
- Mask ambiguous pairs out of the loss.
- Deduplicate examples that create accidental negatives.

Do not force genuinely relevant pairs apart.

### Hard versus easy negatives

For “How do I reset my password?”:

| Document | Category |
|---|---|
| Another valid reset guide | Positive; a false negative if mislabeled |
| Change a username | Hard negative, if it does not answer password reset |
| Banana bread recipe | Easy negative |

Easy negatives allow broad topic discrimination. Hard negatives require distinguishing intent within the same topic.

Mine hard negatives from highly ranked results of an existing retriever, then check relevance. A high-scoring document may be an alternative valid answer.

For “Cancel my subscription,” “Change my subscription plan” is a useful hard negative only if it does not also explain cancellation.

## 10. Dot product, cosine similarity, and normalization

$$
q^\top d = \|q\|\|d\|\cos\theta,
\qquad
\operatorname{cosine}(q,d)=\frac{q^\top d}{\|q\|\|d\|}.
$$

For q = [1, 0], d1 = [2, 0], and d2 = [20, 0]:

| Score | d1 | d2 |
|---|---:|---:|
| Dot product | 2 | 20 |
| Cosine similarity | 1 | 1 |

Normalize nonzero vectors:

$$
\hat q=\frac{q}{\|q\|}, \qquad
\hat d=\frac{d}{\|d\|}, \qquad
\hat q^\top\hat d=\cos\theta.
$$

After normalization, both document vectors become [1, 0]. A larger original norm gives no advantage. Only direction affects the score.

Neither scoring method is universally better; use a method consistent with training. Vector norms can carry information in models trained to use them.

## 11. Temperature

$$
P(d_j\mid q) =
\frac{\exp(s(q,d_j)/\tau)}
{\sum_k \exp(s(q,d_k)/\tau)}, \qquad \tau>0.
$$

Lower temperature sharpens probabilities; higher temperature makes them more uniform.

For positive score 0.8 and negative score 0.6:

| Temperature | Scaled scores | Positive probability |
|---|---|---:|
| 1.0 | 0.8, 0.6 | 0.550 |
| 0.1 | 8, 6 | 0.881 |

A positive temperature does not change ranking for fixed embeddings. If the negative has the higher score, lowering temperature makes the wrong answer more confident and raises the positive-label loss.

Training gradients also change. For a score s_j:

$$
\frac{\partial \mathcal{L}}{\partial s_j}
=
\frac{p_j-\mathbf{1}[j=+]}{\tau}.
$$

A smaller temperature can intensify correction of confident mistakes, but may also amplify label noise. It does not universally increase every gradient; easy correct predictions can saturate.

## 12. Training data and relevance

| Source | Example | Limitation |
|---|---|---|
| Human judgments | Query with annotated relevant document | Cost |
| Search interactions | Query and clicked result | Noise and position bias |
| Existing pairs | Question and accepted answer | Domain mismatch |
| Synthetic queries | LLM-generated query for a document | Artificial or ungrounded queries |

No click does not automatically imply irrelevance: the user may not have seen the result, found another answer, or read enough from the snippet. Higher positions receive more attention.

Use relevance judgments, examination-aware treatment of interactions, or stronger evidence of irrelevance. Even a viewed-but-unclicked result is uncertain.

**Similarity is not identical to relevance.** “Can I use this device in the rain?” can be answered by “The device is not water-resistant; avoid moisture.” The wording differs but the answer is relevant.

For support search, question–relevant-document pairs match the task more directly than question–paraphrase pairs.

## 13. Adapting an LLM into an embedding model

Training steps:

1. Tokenize queries and documents.
2. Compute contextual token states.
3. Pool while excluding padding.
4. Optionally project and normalize.
5. Compute contrastive loss.
6. Backpropagate into trainable parameters.

| Method | Updated parameters | Trade-off |
|---|---|---|
| Full fine-tuning | All parameters | Flexible; higher training-memory requirements |
| LoRA | Adapter parameters in selected layers | Lower trainable-parameter and optimizer-state cost |
| Frozen LLM plus projection | Projection only | Limited by the frozen representation |

When the LLM is frozen, its token embedding table does not update. A trainable projection can still reshape the final retrieval vectors.

## 14. Projection, compression, and evaluation

Example:

$$
h\in\mathbb{R}^{4096}, \quad
W\in\mathbb{R}^{256\times4096}, \quad
b\in\mathbb{R}^{256}, \quad
u=Wh+b, \quad e=\frac{u}{\|u\|}.
$$

For float32 vectors, excluding index and metadata overhead:

| Dimension | Bytes per document | Decimal GB for 1 million documents |
|---|---:|---:|
| 4,096 | 16,384 | 16.384 |
| 256 | 1,024 | 1.024 |

This is 16× less vector storage. Compression may remove useful distinctions.

Evaluate original and projected embeddings on held-out queries with relevance labels, using the same corpus, candidate count, and scoring setup. Compare Recall@K, ranking quality, latency, storage, and difficult slices.

Start with exact search when isolating compression effects; approximate search can introduce separate errors. A Recall@100 drop from 95% to 75% is a substantial quality loss, not preserved quality. Try intermediate dimensions and reassess.

## 15. Bi-encoder retrieval and cross-encoder reranking

| Aspect | Bi-encoder | Cross-encoder |
|---|---|---|
| Processing | Query and document separately | Query and document text jointly |
| Output | Vectors compared by similarity | Pair relevance score |
| Reusable document computation | Yes | No general query-independent pair score |
| Typical role | Retrieve across a large corpus | Rerank a small candidate set |

A single-vector retriever can miss detailed conditions or negation. A cross-encoder can attend across both texts to judge whether the document answers the complete query.

For “Can I cancel my annual subscription and get a refund?”, cancellation instructions and monthly refund policies are related, but annual cancellation and refund eligibility is the full match.

Reranking is optional: measure whether quality gains justify cost.

### Serving one million support documents

1. Offline: encode documents or chunks and build a vector index.
2. Online: encode the query.
3. Retrieve the top K candidates.
4. Optionally score query–candidate text pairs with a cross-encoder.
5. Return the highest-ranked results.

The cross-encoder consumes query and document **text**, not merely their precomputed embeddings. Running it over one million documents per query requires one million pair evaluations. Reranking 100 candidates requires only 100. Previously seen pair scores can be cached, but new queries still need computation.

## 16. Retrieval metrics and candidate count

$$
\operatorname{Recall@K} =
\frac{\text{relevant documents retrieved in top K}}
{\text{total relevant documents}}.
$$

Retrieving 3 of 4 relevant documents gives Recall@K = 0.75.

| Metric | Question answered |
|---|---|
| Recall@K | How many relevant documents reached the candidate set? |
| Precision@K | What fraction of candidates are relevant? |
| MRR | How early is the first relevant result? |
| NDCG@K | Are highly relevant results ranked near the top? |

A reranker cannot recover documents absent from its candidate set. Candidate recall limits downstream retrieval coverage.

Increasing K from 100 to 500 can improve recall but requires five times as many cross-encoder pair evaluations. Wall-clock latency need not grow exactly fivefold because of batching. For a fixed exact ordering, larger top-K sets cannot reduce recall; approximate-search behavior also depends on its settings.

## 17. Updating documents and models

**Document text changes, model stays fixed:** Recompute the changed document’s embedding and replace its index entry. For chunked documents, update affected chunks, add new chunks, and remove deleted chunks. Chunk-boundary changes may affect more than one chunk. Keep text and vector versions consistent.

**Encoder weights change:** Old vectors may no longer be compatible, even if dimensions remain identical. Typically re-embed the corpus, build a new index, and switch compatible query-encoder/index versions together. Reusing old vectors requires explicit compatibility validation.

## 18. Consolidated interview answer

For one million support documents and 50,000 question–relevant-document pairs:

> I would fine-tune a pretrained encoder with a contrastive objective using relevant documents as positives, in-batch negatives, and checked hard negatives. I would pool contextual states, optionally project them, and normalize them for cosine scoring. I would avoid false negatives and split evaluation data from training. Offline, I would compute document embeddings and build an index. Online, I would encode queries, retrieve candidates, and optionally rerank query–document text pairs with a cross-encoder. I would evaluate Recall@K, ranking quality, latency, and memory. Document edits trigger targeted re-embedding; model updates require compatible encoder and index versions.

## 19. Concept checks and corrected takeaways

| Question | Correct takeaway |
|---|---|
| Why not use token IDs as semantic numbers? | IDs are arbitrary; numerical distances do not encode category relationships. |
| Does low prediction loss guarantee nearby embeddings? | No; it rewards correct predictions rather than proximity directly. |
| What happens with only positive attraction? | All vectors can collapse to the same representation without other constraints. |
| Where is “bank” disambiguated? | Transformer contextual processing, not the token lookup. |
| Are repeated token embeddings independent? | No; occurrences share one table row and accumulate gradients. |
| Twenty occurrences with dimension 64: how many row parameters? | 64. |
| Are human target embeddings needed for LLM pretraining? | No; the next token provides supervision. |
| What does weight tying save in the example? | 50,000 × 768 = 38.4 million parameters. |
| Does pooling alone ensure good retrieval? | No; the objective must produce useful retrieval geometry. |
| Should alternative valid answers be pushed away? | No; they are positives or ambiguous pairs, not reliable negatives. |
| Why use hard negatives? | They distinguish fine-grained intent within a topic. |
| Can larger norms help after unit normalization? | No; every normalized vector has norm 1. |
| Can temperature fix ranking for fixed vectors? | No; it changes probabilities and training gradients. |
| Can reranking recover a missing candidate? | No. |
| Is an unclicked result certainly irrelevant? | No; examination and other biases matter. |
| Does projection-only training change the frozen token table? | No. |
| How is compression validated? | Held-out relevance metrics plus serving cost, under controlled comparisons. |
| What is precomputed? | Document vectors; query vectors and uncached cross-encoder scores are computed online. |

## 20. Final distinction

An **input token embedding** is a learned parameter vector selected by a token ID. A **document embedding** is an activation computed from an entire text or chunk, usually from contextual states through pooling and optional projection. Retrieval training shapes these computed vectors so relevant query–document pairs score higher than irrelevant pairs.

**Study status:** Core study complete. These notes consolidate the full session rather than reproduce every conversational turn.
