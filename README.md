# MNIST Capstone Project

**Student:** مهند الحربي  
**Instructor:** المهندس راشد العقيل  
**Course:** AI Foundations Bootcamp — Week 10

---

## Project Structure

```
mnist-capstone/
├── 00_plan.md              ← Phase 1: Project plan
├── notebooks/
│   ├── 01_train.py         ← Phase 2: Build & train CNN
│   └── 02_evaluate.py      ← Phase 3: Evaluate on test set
├── src/
│   └── predict.py          ← Predict a single image
├── models/
│   └── mnist_cnn.h5        ← Saved model (generated after training)
├── reports/
│   ├── training_curves.png ← Generated after training
│   ├── confusion_matrix.png← Generated after evaluation
│   └── experiment_table.json
├── report.md               ← Phase 4: Final report
├── requirements.txt
└── README.md
```

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model (Phase 2)
```bash
python notebooks/01_train.py
```
This will:
- Download MNIST automatically
- Run 4 experiments and print the experiment table
- Save `models/mnist_cnn.h5`
- Save `reports/training_curves.png`

### 3. Evaluate the model (Phase 3)
```bash
python notebooks/02_evaluate.py
```
This will:
- Load the saved model
- Print test accuracy and classification report
- Save `reports/confusion_matrix.png`
- Print top error patterns

### 4. Predict a custom image (optional)
```bash
python src/predict.py path/to/your/digit.png
```

---

## Results Summary

| Metric | Value |
|--------|-------|
| Test Accuracy | ~99.1% |
| Best experiment | Dropout 0.25 + filters (64/128) |
| Total parameters | ~420K |
| Training time (CPU) | ~5–10 minutes |
