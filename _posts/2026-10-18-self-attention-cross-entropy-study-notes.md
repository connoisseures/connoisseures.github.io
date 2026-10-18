---
title: "Self-Attention and Cross-Entropy: Interview Study Notes"
date: 2026-10-18 14:26:00 -0700
categories: [LLM, Interview Preparation]
tags: [self-attention, causal-mask, cross-entropy, softmax, ffn, pytorch, numpy]
description: "Interview study notes covering causal self-attention implementations, code corrections, attention scaling, FFN roles, cross-entropy derivations, and next-token prediction."
math: true
---

Study completed: 2026-10-18  
Project: 2026_TT_prep

## 1. Scope and core takeaways

This study combined live coding, debugging, oral questions, and loss derivations:
- Single-head causal self-attention in PyTorch and NumPy.
- Stable softmax, tensor shapes, masking, and attention scaling.
- The FFN's role and why it needs a nonlinear activation.
- Categorical cross-entropy derived from maximum likelihood.
- Softmax–cross-entropy gradients and next-token language modeling.

## 2. Self-attention equations and shapes

Input X has shape [B, T, D]. B is batch size, T is sequence length, and D is model width. Projection matrices have shapes [D, d_k], [D, d_k], and [D, d_v].

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V
$$

Q and K have shape [B, T, d_k]; V has shape [B, T, d_v].

$$
S=\frac{QK^\top}{\sqrt{d_k}},\qquad
A=\operatorname{softmax}(S+M),\qquad O=AV
$$

S and A have shape [B, T, T]; O has shape [B, T, d_v]. The transpose swaps only the final two dimensions of K.

Each score row corresponds to one query; columns correspond to keys. Softmax normalizes over the last dimension, the key positions. The causal mask is zero for allowed positions and negative infinity for future positions:

$$
M_{ij}=\begin{cases}0&j\le i\\-\infty&j>i.\end{cases}
$$

These examples implement the attention operation without padding, dropout, residual connections, normalization, or an output projection.

## 3. Review of the original code

The original attempt correctly introduced Q/K/V projections and subtracting the maximum for stable exponentials. Corrections:

1. Divide exponentials by their sum, not their mean. Dividing by the mean makes the row sum equal to the number of entries.
2. Divide scores by sqrt(d_k) before softmax. Scaling probabilities afterward changes their sum and does not implement scaled dot-product attention.
3. Mask future scores with negative infinity before softmax.
4. Use nn.Linear or nn.Parameter; nn.parameters(d, d) is not a valid parameter constructor.
5. Initialize trainable weights once in __init__, not on each forward call.
6. Assign reshape and transpose results; these operations do not replace the original variable.
7. PyTorch transpose swaps two dimensions; permute reorders several dimensions. NumPy transpose can accept a full axes tuple.
8. Correct the spelling of transpose.
9. Keep PyTorch tensor operations in PyTorch for autograd and device support.
10. A single head needs no explicit head dimension. The implementations below use d_k = d_v = D.

For multiple heads, the corresponding PyTorch transformation would be:

```python
q = q.reshape(B, T, H, d_k).permute(0, 2, 1, 3)
# [B, H, T, d_k]
```

## 4. Correct PyTorch implementation

```python
import math
import torch
import torch.nn as nn

def softmax(x):
    shifted = x - x.amax(dim=-1, keepdim=True)
    exp_x = shifted.exp()
    return exp_x / exp_x.sum(dim=-1, keepdim=True)

class SingleHeadCausalSelfAttention(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)

    def forward(self, X):
        B, T, D = X.shape
        q = self.q_proj(X)                 # [B, T, D]
        k = self.k_proj(X)                 # [B, T, D]
        v = self.v_proj(X)                 # [B, T, D]

        scores = q @ k.transpose(-2, -1)   # [B, T, T]
        scores = scores / math.sqrt(D)

        future = torch.ones(
            T, T, dtype=torch.bool, device=X.device
        ).triu(diagonal=1)
        scores = scores.masked_fill(future, float("-inf"))

        weights = softmax(scores)          # [B, T, T]
        return weights @ v                 # [B, T, D]
```

Each query has at least its own position available, so the causal mask alone does not create fully masked rows. Additional padding masks require care: a row containing only negative infinity makes this softmax undefined.

## 5. Stable softmax and amax versus max

Subtracting the row maximum leaves softmax unchanged:

$$
\frac{e^{x_i-m}}{\sum_j e^{x_j-m}}
=\frac{e^{x_i}}{\sum_j e^{x_j}},\qquad m=\max_j x_j.
$$

All shifted values are nonpositive, avoiding exponential overflow. For [2, 5, 3], the shifted values are [-3, 0, -2].

```python
x.amax(dim=-1, keepdim=True)         # Maximum values
x.max(dim=-1, keepdim=True)          # Values and indices
x.max(dim=-1, keepdim=True).values   # Maximum values
```

The first and third expressions return the same maximum values. keepdim=True retains a size-one final axis for broadcasting. Their gradient handling at tied maxima can differ; they are not identical APIs in every respect.

## 6. NumPy implementation

```python
import numpy as np

def softmax_numpy(x):
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

class SingleHeadCausalSelfAttentionNumpy:
    def __init__(self, d, seed=None):
        rng = np.random.default_rng(seed)
        scale = 1.0 / np.sqrt(d)
        self.Wq = rng.normal(0, scale, size=(d, d))
        self.Wk = rng.normal(0, scale, size=(d, d))
        self.Wv = rng.normal(0, scale, size=(d, d))

    def forward(self, X):
        B, T, D = X.shape
        assert D == self.Wq.shape[0]

        q = X @ self.Wq                    # [B, T, D]
        k = X @ self.Wk                    # [B, T, D]
        v = X @ self.Wv                    # [B, T, D]

        scores = q @ k.swapaxes(-2, -1)    # [B, T, T]
        scores = scores / np.sqrt(D)

        future = np.triu(np.ones((T, T), dtype=bool), k=1)
        scores = np.where(future, -np.inf, scores)

        weights = softmax_numpy(scores)
        return weights @ v                # [B, T, D]

rng = np.random.default_rng(0)
X = rng.normal(size=(2, 4, 8))
attention = SingleHeadCausalSelfAttentionNumpy(d=8, seed=42)
out = attention.forward(X)
print(out.shape)  # (2, 4, 8)
```

This implements a forward pass. NumPy does not automatically provide backpropagation.

### Mixing NumPy and PyTorch

Computing a scalar constant with NumPy is fine when it is independent of trainable tensors:

```python
scores = scores / np.sqrt(d_k)  # d_k is a Python integer
# math.sqrt(d_k) is simpler.
```

Converting a tensor to NumPy disconnects the computation from the original autograd graph:

```python
array = tensor.detach().cpu().numpy()
result = np.exp(array)
new_tensor = torch.from_numpy(result)
# No gradient connection from new_tensor back to tensor.
```

NumPy-to-PyTorch conversion can share CPU memory. Use conversions for preprocessing or output analysis; use PyTorch tensor operations throughout a trainable PyTorch forward pass.

## 7. Why scale attention by sqrt(d_k)?

Assume all components of q and k are mutually independent, have mean zero, and variance one.

$$
q^\top k=\sum_{j=1}^{d_k}q_jk_j.
$$

Each product has mean zero and variance one:

$$
\operatorname{Var}(q_jk_j)
=\mathbb E[q_j^2]\mathbb E[k_j^2]=1.
$$

Therefore:

$$
\operatorname{Var}(q^\top k)=d_k,\qquad
\operatorname{Var}\left(\frac{q^\top k}{\sqrt{d_k}}\right)=1.
$$

Without scaling, score differences typically grow with head dimension. Softmax can become too concentrated and have small derivatives. Scaling stabilizes the score scale; actual learned queries and keys need not satisfy these idealized assumptions exactly.

Your answer correctly identified variance d_k and division by its standard deviation.

## 8. Causal attention checks

### First token in a four-token sequence

Masked scores are [s_11, -infinity, -infinity, -infinity]. Exponentials at masked positions are zero. The attention weights are [1, 0, 0, 0], and the attention output is exactly v_1 before any output projection or residual connection.

You initially did not know this answer.

### Second token

The first two weights may differ, but sum to one. Future positions have weight zero. You answered this correctly.

For example:

$$
A_2=[0.3,0.7,0,0],\qquad o_2=0.3v_1+0.7v_2.
$$

Equal weights occur when the two allowed scores are equal.

## 9. FFN role and nonlinearity

Self-attention exchanges information across token positions. A standard position-wise FFN transforms features within each token independently, using the same learned weights at every position.

$$
\operatorname{FFN}(h_i)=\operatorname{GELU}(h_iW_1+b_1)W_2+b_2.
$$

Example shapes:

$$
[B,T,768]\longrightarrow[B,T,3072]\longrightarrow[B,T,768].
$$

W_1 has shape [768, 3072]; W_2 has shape [3072, 768]. The FFN processes contextual features already gathered by attention; it does not directly mix token positions.

You initially answered that the FFN mixes across positions. After correction, you correctly answered that changing token 1 immediately before the FFN does not affect token 2's output during that operation.

Without GELU, the two affine layers collapse:

$$
(hW_1+b_1)W_2+b_2
=h(W_1W_2)+(b_1W_2+b_2).
$$

You correctly recognized this. The activation enables nonlinear feature transformations.

## 10. Cross-entropy from maximum likelihood

For one example with C classes, p is the predicted categorical distribution and y is a one-hot target.

$$
P(y\mid x)=\prod_{c=1}^{C}p_c^{y_c}.
$$

For p = [0.1, 0.7, 0.2] and y = [0, 1, 0], this probability is 0.7.

Maximum likelihood maximizes the probability of the observed label. Minimizing its negative logarithm yields:

$$
L=-\log P(y\mid x)
=-\log\left(\prod_{c=1}^{C}p_c^{y_c}\right)
=-\sum_{c=1}^{C}y_c\log p_c.
$$

For a one-hot target:

$$
L=-\log p_{\mathrm{true}}.
$$

You correctly selected -log(0.7), rather than log(0.7), as the loss to minimize.

For independent training examples, likelihoods multiply. Taking negative logs gives a sum of example losses; dividing by the number of examples gives the mean loss.

All logs here are natural logarithms.

### Numerical intuition

- Correct-class probability 1/3: loss log(3), approximately 1.099.
- Correct-class probability 0.7: loss approximately 0.357.
- Correct-class probability 1: loss 0, understood as the ideal limiting case for finite-logit softmax.

You initially needed help with the uniform three-class case, then correctly answered that increasing the correct-class probability from 0.3 to 0.9 decreases the loss.

## 11. Cross-entropy expressed using logits

Softmax defines:

$$
p_c=\frac{e^{z_c}}{\sum_j e^{z_j}}.
$$

Substitute into the one-hot loss:

$$
L=-\log\left(\frac{e^{z_{\mathrm{true}}}}{\sum_j e^{z_j}}\right)
=-\left[z_{\mathrm{true}}-\log\sum_j e^{z_j}\right].
$$

Thus:

$$
\boxed{L=\log\sum_j e^{z_j}-z_{\mathrm{true}}}.
$$

You needed a walkthrough for this algebra. The key identities are log(a/b) = log(a) - log(b) and log(exp(z)) = z.

For numerical stability:

$$
\log\sum_j e^{z_j}=m+\log\sum_j e^{z_j-m},\qquad m=\max_j z_j.
$$

Manual PyTorch implementation:

```python
import torch

# logits: [B, C]; targets: [B], dtype=torch.long
m = logits.amax(dim=-1, keepdim=True)
shifted = logits - m
log_probs = shifted - shifted.exp().sum(
    dim=-1, keepdim=True
).log()

rows = torch.arange(logits.shape[0], device=logits.device)
loss = -log_probs[rows, targets].mean()
```

## 12. Gradient of softmax plus cross-entropy

For a normalized target distribution:

$$
L=\log\sum_j e^{z_j}-\sum_j y_jz_j.
$$

Differentiating with respect to z_k:

$$
\frac{\partial L}{\partial z_k}
=\frac{e^{z_k}}{\sum_j e^{z_j}}-y_k
=\boxed{p_k-y_k}.
$$

For p = [0.1, 0.7, 0.2] and y = [0, 1, 0]:

$$
\nabla_zL=[0.1,-0.3,0.2].
$$

A direct gradient descent step on the correct-class logit gives:

$$
z_{\mathrm{true}}\leftarrow z_{\mathrm{true}}-\eta(-0.3)
=z_{\mathrm{true}}+0.3\eta.
$$

You correctly answered that this pushes the correct-class logit up. Positive gradients push the other logits down in this direct-logit interpretation.

In an actual network, optimization updates shared parameters through the chain rule; it does not independently optimize every example's logits. For a mean loss, each example's logit gradient also includes the averaging factor.

## 13. Next-token prediction and target shifting

Text supplies its own labels, so this is self-supervised learning. The reason for shifting is specifically to predict the next token:

$$
P(x_{t+1}\mid x_0,\ldots,x_t).
$$

```text
sequence: [BOS, I, like, tea, EOS]
inputs:   [BOS, I, like, tea]
targets:  [I, like, tea, EOS]
```

At the position containing "like", the model has seen BOS, I, and like, and predicts tea.

Using identical input and target positions exposes the answer to the model and allows copying. A causal mask alone does not prevent this: each position can attend to itself, and its own embedding also enters the residual stream.

You identified self-supervised learning, then correctly answered that the causal mask does not prevent copying the current token.

## 14. Language-model loss implementation

```python
import torch.nn.functional as F

# logits: [B, T, V], produced from the original tokens
# tokens: [B, T], integer token IDs
V = logits.shape[-1]

prediction_logits = logits[:, :-1, :]  # [B, T-1, V]
targets = tokens[:, 1:]                # [B, T-1]

loss = F.cross_entropy(
    prediction_logits.reshape(-1, V),  # [B*(T-1), V]
    targets.reshape(-1),               # [B*(T-1)]
)
```

Softmax normalizes over vocabulary V, not over batch or time. You answered this correctly.

Pass raw logits to F.cross_entropy. It internally computes log-softmax and negative log-likelihood. Applying softmax first makes the function treat probabilities as logits and yields a different objective. You correctly answered not to apply softmax beforehand.

This code assumes no padding. With padding, ignored target positions should be excluded from the loss, for example using ignore_index with appropriately masked labels. If inputs and targets are already shifted before the forward pass, do not shift them a second time.

## 15. Interview recap and review priorities

A compact spoken answer:

"Self-attention projects inputs into queries, keys, and values. I compute QK transpose, divide by the square root of the key dimension, mask future positions, normalize over keys, and multiply by values. Scaling controls score variance. Attention combines tokens, while the position-wise FFN applies nonlinear feature transformations. Cross-entropy comes from the negative log-likelihood of the observed class; with softmax it is log-sum-exp minus the correct logit, and its logit gradient is p minus y. For language modeling, targets are shifted by one token and cross-entropy operates on raw vocabulary logits."

Review priorities based on this session:
- Reproduce attention code without missing masking or placing scaling after softmax.
- Explain why attention mixes positions and the FFN transforms features.
- Derive cross-entropy from categorical likelihood without hints.
- Derive the logits form using logarithm identities.
- Recall that uniform prediction over C classes gives loss log(C).

Independent practice prompts:
1. Write single-head causal attention from memory with all shapes.
2. Derive Var(q dot k) under the stated independence assumptions.
3. Explain why a first-token attention row is [1, 0, 0, 0].
4. Collapse a two-layer FFN without its activation into one affine map.
5. Derive categorical cross-entropy and its p-y gradient.
6. Implement shifted language-model cross-entropy without applying softmax twice.

