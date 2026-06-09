# MNIST Capstone — Project Plan

## Problem Statement

We want to **automatically recognise handwritten digits** for data-entry teams
so that forms are processed faster and with fewer human errors, by building a
model that looks at a 28×28 grayscale image and outputs the correct digit (0–9).

---

## Success Metrics

| Type | Metric | Target |
|------|--------|--------|
| Technical | Test accuracy | ≥ 99% |
| Technical | Inference latency | < 50 ms per image |
| Business | Forms processed without human review | ≥ 95% |

---

## Data Needs

| Item | Detail |
|------|--------|
| Source | `keras.datasets.mnist` (Kaggle: digit-recognizer) |
| Format | 28×28 grayscale PNG, uint8 |
| Size | 70,000 images (60,000 train + 10,000 test) |
| Classes | 10 (digits 0–9), balanced |
| License | Public domain (Yann LeCun et al.) |

---

## Risk Register

| # | Risk | Likelihood | Mitigation |
|---|------|-----------|------------|
| 1 | **Overfitting** — model memorises training data | High | Add Dropout (0.25–0.5), monitor val accuracy |
| 2 | **Confusable digits** — e.g. 4 vs 9, 3 vs 5 | Medium | Analyse confusion matrix; consider per-class augmentation |
| 3 | **Distribution shift** — clean MNIST ≠ messy real handwriting | Medium | Note limitation in report; future work: test on own photos |
| 4 | **Irreproducibility** — different results each run | Low | Fix `np.random.seed(42)` and `tf.random.set_seed(42)` |

---

## Experiment Plan (Phase 2)

Change **one variable at a time** and record validation accuracy:

| Run | Change from baseline | Expected effect |
|-----|----------------------|-----------------|
| 1 | Baseline CNN, 5 epochs | — |
| 2 | Add Dropout(0.25) | Reduce overfitting |
| 3 | Increase epochs to 10 | Higher accuracy |
| 4 | Add data augmentation | Better generalisation |

---

## Timeline

| Day | Phase |
|-----|-------|
| 46 | Phase 1 — Plan |
| 47 | Phase 2 — Build & Train |
| 48 | Phase 3 — Evaluate |
| 49 | Phase 4 — Present |
| 50 | Phase 5 — Package & Submit |

---

*AI assistance used for structure and wording — all decisions are the author's own.*
