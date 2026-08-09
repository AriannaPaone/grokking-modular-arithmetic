# Investigating Grokking in Modular Arithmetic

Mechanistic interpretability of 2-layer transformers trained on **(a + b + c) mod 37**, and a comparison of the circuits learned under causal vs. non-causal attention.

Grokking is the phenomenon where a network first memorises the training set and only much later, after prolonged optimisation, snaps to near-perfect generalisation. This project asks what actually changes inside the model at that transition, and specifically: **does a three-operand sum get computed in one shot, or staged as `(a+b)` then `+c`?**

Our analysis points to a one shot computation. Activation patching rules out staged computation in both models, and the apparent `a+b` signal present in the causal model turns out to be an artefact of the causal mask.

---

## Key findings

**1. Weight decay is the driver, training fraction sets the delay.**
At weight decay 0.01 the model memorises within a few epochs and validation accuracy never improves. At weight decay 1.0–3.0 it generalises reliably across every training fraction tested. Smaller training fractions delay grokking but don't prevent it, at `frac = 0.6` train and validation accuracy rise together with no memorisation plateau at all, suggesting the plateau appears when the training set is small enough to memorise cheaply.

**2. Embeddings become circles at the transition.**
Before grokking, the Fourier energy of the embedding matrix `W_E` is flat and embeddings are unstructured in PCA space. At the transition, energy concentrates at a sparse set of frequencies. Each active frequency defines its own circle in a 2-D subspace, which is the prerequisite for doing modular arithmetic through trigonometric identities.

**3. Layer 0 computes the answer; layer 1 makes it readable.**
The full sum is linearly decodable from the residual stream after layer 0, but the model's own output head can't read it there since logit lens accuracy is 0.125, barely above the 1/37 ≈ 0.027 chance rate. Only after layer 1 does it reach 1.000. Layer 0 produces the answer in a Fourier basis; layer 1 converts the format.

**4. No staged computation, and the evidence for it was an artefact.**
The causal model shows `R² ≈ 1.1` (sum of $R^2$ for $sin$ and $cos$) for target `a+b` at positions `b` and `c`, which looks like intermediate computation. But patching those positions never produces the corrupted answer, and the signal is absent in the non-causal model. It's a byproduct of `b` being the first position that can attend to both `a` and itself.

**5. The causal mask decides where computation localises.**
By step 15,000 the causal model has funnelled everything through the `=` position (patching `=` flips the output completely, `acc = 1.000`). The non-causal model never consolidates: patching `=` has almost no causal effect (0.168), while patching any operand position damages the clean answer. Same algorithm, different spatial organisation, driven purely by the mask.

---

## Setup

| | |
|---|---|
| Task | `(a + b + c) mod 37`, tokenised as `a b c =` |
| Dataset | all 37³ = 50,653 inputs; 20% train for the mechanistic analysis |
| Model | 2-layer transformer, `d_model` = 128, 4 heads, FFN 512, no weight tying |
| Vocab | 38 tokens (residues 0–36, plus `=`) |
| Optimiser | AdamW, lr 1e-3 → 1e-4 cosine, β = (0.9, 0.98), weight decay 1.0 |
| Training | ≥ 15,000 steps; checkpoints every 1,000 steps |
| Variants | causal vs. non-causal attention, identical otherwise |

---

## Interpretability tools

Four tools, each answering a different question. The first three are correlational; only the last establishes causation.

**Fourier energy** — projects each column of `W_E` onto a normalised Fourier basis and measures energy at k = 1…18. Answers: *has the model chosen to represent numbers as points on circles?*

**Linear probes** — fits OLS from the 128-d residual stream to `cos(2πkt/p)` and `sin(2πkt/p)` separately, reporting `R²_cos + R²_sin ∈ [0, 2]`. Answers: *where and when does each quantity (individual operands, partial sums, the full answer) become linearly decodable?*

**Logit lens** — applies the model's own LayerNorm + `W_U` to intermediate residual streams. Distinct from probing: a probe fits a *new* linear map and tests decodability in principle, the logit lens tests whether the representation is already in the format `W_U` expects. High probe `R²` with low lens accuracy means computed but notyetformatted. 

**Activation patching** — corrupts one operand `a → a'`, runs both inputs through layer 0, swaps the residual stream at a single position, and checks whether the output flips to `(a' + b + c) mod p`. Answers: *does this position carry that operand's contribution causally, or is it inert?*

---

## Activation patching results

Effect on the output when the residual stream at one position is swapped for its corrupted counterpart, at step 15,000 (post-consolidation):

| Patched position | Causal: acc(a'+b+c) | Causal: acc(a+b+c) | Non-causal: acc(a'+b+c) | Non-causal: acc(a+b+c) |
|---|---|---|---|---|
| `a` | 0.000 | 1.000 | 0.030 | 0.658 |
| `b` | 0.000 | 1.000 | 0.014 | 0.762 |
| `c` | 0.000 | 1.000 | 0.016 | 0.748 |
| `=` | **1.000** | 0.000 | 0.168 | 0.142 |

The causal model routes everything through `=`. The non-causal model distributes the computation across operand positions and never concentrates it anywhere.

---

## Repository structure

<!-- TODO: replace with your actual layout -->

```
.
├── src/                # model, training loop, config
├── analysis/           # Fourier energy, probes, logit lens, patching
├── notebooks/          # figures and exploration
├── checkpoints/        # saved weights + cached forward passes (not tracked)
├── figures/
└── report.pdf
```

Checkpoints store full model weights plus a cached forward pass on 512 held-out examples, including residual stream activations at every layer and all attention matrices. Every analysis reads from these caches rather than re-running the model.

## Reproducing

Check ['instructions.md](instructions.md)

---

## Open questions

- Are separate attention head components needed to represent the trigonometric products, or can they be collapsed into one?
- Do the same structures emerge across other seeds, moduli, and operand counts?
- Why does the circuit keep drifting under weight decay after accuracy has saturated?

## References

1. Nanda, Chan, Lieberum, Smith, Steinhardt. *Progress measures for grokking via mechanistic interpretability.* [arXiv:2301.05217](https://arxiv.org/abs/2301.05217), 2023.
2. Power, Burda, Edwards, Babuschkin, Misra. *Grokking: Generalization beyond overfitting on small algorithmic datasets.* [arXiv:2201.02177](https://arxiv.org/abs/2201.02177), 2022.

## Authors

Arianna Paone, Damien Bardina — Language Engineering, KTH Royal Institute of Technology, June 2026.

Full write-up in [`report.pdf`](report.pdf).
