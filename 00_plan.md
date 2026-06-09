# MNIST Capstone — Project Plan

**Student:** مهند الحربي  
**Instructor:** المهندس راشد العقيل  
**Course:** AI Foundations Bootcamp — Week 10

---

## Problem Statement

We want to **automatically recognise handwritten digits** for form-processing teams
so that data-entry staff can focus on exceptions, by building a model that classifies
a 28×28 grayscale image into one of ten digit classes (0–9).

---

## Success Metrics

| Type | Metric | Target |
|------|--------|--------|
| Technical | Test accuracy | ≥ 99% |
| Technical | Inference latency (single image, CPU) | < 50 ms |
| Business | Forms processed without human review | ≥ 95% |

---

## Data Needs

| Item | Detail |
|------|--------|
| Source | `keras.datasets.mnist` |
| Format | 28×28 grayscale, uint8 (0–255) |
| Size | 70,000 images — 60,000 train + 10,000 test |
| Classes | 10 balanced classes (0–9) |
| Licence | Public domain / CC0 |

**Split used:** 54,000 train / 6,000 validation / 10,000 test

---

## Risk Register

| # | Risk | Mitigation |
|---|------|-----------|
| 1 | **Overfitting** | Add Dropout (0.25); monitor val loss each epoch |
| 2 | **Confusable digits** (4↔9, 3↔5) | Error analysis on confusion matrix |
| 3 | **Domain gap** (clean MNIST vs real handwriting) | Note as limitation; add data augmentation |
| 4 | **Reproducibility** | Fix np.random.seed(42) and tf.random.set_seed(42) |

---

## Timeline

| Day | Phase | Deliverable |
|-----|-------|-------------|
| 46 | Plan | 00_plan.md |
| 47 | Build & Train | 01_train.py + experiment table |
| 48 | Evaluate | 02_evaluate.py + confusion matrix |
| 49 | Present | report.md |
| 50 | Package | clean repo + requirements.txt + models/mnist_cnn.h5 |
