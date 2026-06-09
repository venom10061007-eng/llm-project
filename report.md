# Recognising Handwritten Digits with a CNN

**Student:** مهند الحربي  
**Instructor:** المهندس راشد العقيل  
**Course:** AI Foundations Bootcamp — Final Capstone (Week 10)  
**Dataset:** MNIST — Digit Recognizer (kaggle.com/c/digit-recognizer)

---

## 1. Introduction

For this capstone project I used the MNIST dataset, which contains 70,000 grayscale images (28×28 pixels) of handwritten digits from 0 to 9. The dataset is perfectly balanced, with approximately 7,000 samples per class.

My goal was to answer three questions:
1. Can a simple Convolutional Neural Network (CNN) read handwritten digits accurately?
2. Which digits does it confuse most often?
3. How does adding Dropout regularisation affect generalisation?

---

## 2. Planning Summary

**Problem statement:** We want to automatically recognise handwritten digits for form-processing teams so that data-entry staff can focus on exceptions, by building a model that classifies a 28×28 grayscale image into one of ten digit classes (0–9).

**Success metrics:**

| Metric | Target |
|--------|--------|
| Test accuracy | ≥ 99% |
| Inference latency (CPU) | < 50 ms |
| Forms auto-processed | ≥ 95% |

**Main risks identified:**
- Overfitting → mitigated with Dropout (0.25)
- Confusable digits (4↔9, 3↔5) → mitigated with error analysis
- Domain gap (clean MNIST vs real handwriting) → noted as limitation

---

## 3. Development Summary

**Data split:** 54,000 training / 6,000 validation / 10,000 test (locked until final evaluation)

**Preprocessing:** Pixels scaled from 0–255 to 0–1. Channel dimension added: (28,28) → (28,28,1). Random seeds fixed at 42 for reproducibility.

**Architecture (final model):**
```
Input (28×28×1)
→ Conv2D(64, 3×3, ReLU)
→ MaxPooling2D
→ Conv2D(128, 3×3, ReLU)
→ MaxPooling2D
→ Flatten
→ Dropout(0.25)
→ Dense(64, ReLU)
→ Dense(10, Softmax)
```

**Experiment table:**

| Experiment | Epochs | Dropout | Val Acc |
|-----------|--------|---------|---------|
| Baseline (32/64 filters) | 5 | 0.0 | 98.53% |
| More epochs (32/64) | 10 | 0.0 | 98.70% |
| Dropout 0.25 (32/64) | 10 | 0.25 | 98.90% |
| **Dropout 0.25 + larger filters (64/128)** | **10** | **0.25** | **99.10%** |

Each experiment changed only ONE variable to isolate its effect.

---

## 4. Key Findings

**Finding 1 — A small CNN achieves high accuracy.**  
The final model reached **99.1% accuracy** on the untouched test set, meeting the ≥ 99% target. Because MNIST is perfectly balanced across all 10 classes, this accuracy figure is trustworthy.

**Finding 2 — Mistakes cluster around visually similar digits.**  
The confusion matrix showed the most errors occurring between digit pairs 4↔9 and 3↔5. Inspecting the misclassified images confirmed these were unusually written or ambiguous digits that even humans might hesitate on.

**Finding 3 — Dropout significantly improved generalisation.**  
The baseline model (no Dropout) showed a growing gap between training accuracy (~99.5%) and validation accuracy (~98.7%) — a classic sign of overfitting. Adding Dropout(0.25) narrowed this gap and pushed validation accuracy above 99%.

**Finding 4 — Larger filters extract richer features.**  
Doubling the filter counts from (32,64) to (64,128) gave a further 0.2% improvement in validation accuracy at the cost of approximately 2× more parameters and slightly longer training time.

> *[Figure 1: Training/validation accuracy and loss curves — see reports/training_curves.png]*

> *[Figure 2: 10×10 confusion matrix on test set — see reports/confusion_matrix.png]*

---

## 5. Limitations

- **Clean data only:** The model was trained and tested exclusively on MNIST's clean, centred, well-sized digits. Performance on messy or rotated real-world handwriting is expected to be lower.
- **No data augmentation:** Adding random rotations (±10°) and shifts (±2 px) would help the model generalise to more varied input.
- **Fixed image size:** The pipeline requires a 28×28 input. Any real-world system would need a preprocessing step to detect and crop individual digits from a document.

---

## 6. What I Would Do Next

1. **Add data augmentation** (small rotations and shifts) and compare accuracy with and without it.
2. **Test on real handwriting** — photograph my own handwritten digits and run them through the model.
3. **Try a deeper architecture** such as ResNet-20 to see if accuracy improves further.
4. **Build a simple web interface** using Flask or Streamlit where a user can draw a digit and get a real-time prediction.

---

## 7. Reflection

**What worked:**  
The CNN architecture came together quickly thanks to the clear structure in the project guide. The experiment table was very useful — by changing only one thing at a time I could clearly see the contribution of each improvement.

**What didn't work initially:**  
The first training run without Dropout overfit noticeably. I could see the training accuracy climbing to 99%+ while validation accuracy plateaued around 98.7%.

**What I would change:**  
I would add early stopping from the beginning to avoid unnecessary epochs, and I would start with data augmentation rather than adding it as an afterthought.

**Biggest learning:**  
A single accuracy number is not enough. The confusion matrix and error analysis told a much richer story about where and why the model fails.

---

*Report prepared by: مهند الحربي*  
*Reviewed by: المهندس راشد العقيل*  
*AI Foundations Bootcamp — Week 10 Capstone*
