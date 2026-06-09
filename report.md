# Recognising Handwritten Digits with a CNN
**Student:** مهند الحربي  
**Instructor:** المهندس راشد العقيل  
**Course:** AI Foundations Bootcamp — Week 10  
**Note:** Generated with AI assistance (Claude) as permitted

---

## 1. Introduction
For this project I used the MNIST dataset, which contains 70,000 grayscale images of handwritten digits from 0 to 9. My goal was to answer three questions:
1. Can a simple CNN read handwritten digits accurately?
2. Which digits does it confuse most often?
3. How can regularisation (Dropout) improve generalisation?

---

## 2. Planning Summary
I wrote a problem statement focused on automatically reading digits for a form-processing system. I set a technical target of **≥ 99% accuracy** and **< 50 ms latency** per prediction, plus a business target of reading 95% of forms without a human reviewer. The main risks I identified were overfitting, confusable digits (4↔9, 7↔1), and the gap between clean MNIST images and messy real-world handwriting.

---

## 3. Development Summary
I scaled pixels to the 0–1 range and split the data into **54,000 training**, **6,000 validation**, and **10,000 test** images. I built a CNN with two convolutional layers, max-pooling, and two dense layers, with random seeds set for reproducibility.

**Experiment Table:**

| Experiment | Epochs | Dropout | Val Acc |
|-----------|--------|---------|---------|
| Baseline (32/64) | 5 | 0.0 | 98.83% |
| More epochs (32/64) | 10 | 0.0 | 99.05% |
| Dropout 0.25 (32/64) | 10 | 0.25 | 99.03% |
| Dropout 0.25 + larger (64/128) | 10 | 0.25 | **99.08%** ✅ |

Each experiment changed **one variable at a time** to isolate what helped.

---

## 4. Key Findings

**Finding 1 — High accuracy achieved.**  
The final model reached **99.30% accuracy** on the untouched test set, exceeding the 99% target. Only 70 out of 10,000 images were misclassified.

**Finding 2 — Mistakes cluster around look-alike digits.**  
The top errors were:
- Digit 7 → predicted as 1 (5 times)
- Digit 9 → predicted as 4 (5 times)
- Digit 9 → predicted as 7 (4 times)

These are visually similar pairs, especially with unusual handwriting styles.

**Finding 3 — Larger filters improved performance.**  
Moving from (32/64) to (64/128) filters gave the best validation accuracy, as the model could learn richer features.

**Finding 4 — Latency target met.**  
Average inference latency was **44.2 ms**, under the 50 ms target.

*(See confusion_matrix.png and training_curves.png in reports/)*

---

## 5. What I Would Do Next
1. **Data augmentation** — add small rotations and shifts to handle messier handwriting
2. **Deeper architecture** — try ResNet or EfficientNet
3. **Real-world test** — photograph my own handwritten digits and evaluate the model on them
4. **Deployment** — wrap the model in a simple web app using Flask or Streamlit
