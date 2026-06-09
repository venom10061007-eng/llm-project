# Project Plan — MNIST Digit Classifier
**Student:** مهند الحربي  
**Instructor:** المهندس راشد العقيل  
**Course:** AI Foundations Bootcamp — Week 10

---

## Problem Statement
We want to **automatically recognise handwritten digits** for **form-processing systems** so that **human effort is reduced**, by building a model that **classifies 28×28 grayscale images into digits 0–9**.

---

## Success Metrics
| Metric | Target |
|--------|--------|
| Test Accuracy | ≥ 99% |
| Inference Latency | < 50 ms per image |
| Business Goal | Read 95% of forms without human review |

---

## Data
- **Source:** MNIST via `keras.datasets.mnist`
- **Format:** 28×28 grayscale images, labels 0–9
- **Size:** 70,000 images (60,000 train + 10,000 test)
- **License:** Public domain

**Split used:**
- Train: 54,000
- Validation: 6,000
- Test: 10,000 (touched only once in Phase 3)

---

## Risks & Mitigations
| Risk | Mitigation |
|------|-----------|
| Overfitting | Add Dropout layer, monitor val_accuracy |
| Confusable digits (4↔9, 3↔5) | Analyse confusion matrix per digit |
| Gap between clean MNIST & real handwriting | Note as limitation in report |
