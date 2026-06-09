# =============================================================
# 02_evaluate.py  —  Phase 3: Evaluate the Model
# Student  : مهند الحربي
# Instructor: المهندس راشد العقيل
# Course   : AI Foundations Bootcamp — Week 10
# Note     : Test set is touched ONCE here only
# =============================================================

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import os

os.makedirs('reports', exist_ok=True)

# ── 1. Load data & model ──────────────────────────────────────
print("[1/4] Loading data and model...")
(X_full, y_full), (X_test, y_test) = keras.datasets.mnist.load_data()
X_test = X_test / 255.0
X_test = X_test[..., None]

model = keras.models.load_model('models/mnist_cnn.h5')
print("  Model loaded ✓")

# ── 2. Test Accuracy ─────────────────────────────────────────
print("\n[2/4] Computing test accuracy...")
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"  Test Loss     : {loss:.4f}")
print(f"  Test Accuracy : {acc:.4f}  ({acc*100:.2f}%)")

# ── 3. Classification Report ──────────────────────────────────
print("\n[3/4] Classification report (per digit)...")
y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
report = classification_report(y_test, y_pred,
                                target_names=[str(i) for i in range(10)])
print(report)
with open('reports/classification_report.txt', 'w', encoding='utf-8') as f:
    f.write(f"Student: مهند الحربي\n")
    f.write(f"Instructor: المهندس راشد العقيل\n\n")
    f.write(f"Test Accuracy: {acc:.4f}\n\n")
    f.write(report)
print("  Saved → reports/classification_report.txt")

# ── 4. Confusion Matrix ───────────────────────────────────────
print("\n[4/4] Plotting confusion matrix & error analysis...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Confusion matrix
disp = ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    display_labels=[str(i) for i in range(10)],
    ax=axes[0],
    colorbar=False,
    cmap='Blues'
)
axes[0].set_title('Confusion Matrix — Test Set', fontsize=13)

# Error analysis: 9 misclassified images
wrong_idx = np.where(y_pred != y_test)[0]
axes[1].set_title(f'9 Misclassified Samples (out of {len(wrong_idx)} errors)', fontsize=11)
axes[1].axis('off')

# Create inner grid for the 9 images
inner = axes[1].inset_axes([0, 0, 1, 1])
inner.axis('off')
for i, idx in enumerate(wrong_idx[:9]):
    ax = fig.add_axes([
        axes[1].get_position().x0 + (i % 3) * axes[1].get_position().width / 3,
        axes[1].get_position().y0 + (2 - i // 3) * axes[1].get_position().height / 3,
        axes[1].get_position().width / 3,
        axes[1].get_position().height / 3
    ])
    ax.imshow(X_test[idx].squeeze(), cmap='gray')
    ax.set_title(f'True:{y_test[idx]}  Pred:{y_pred[idx]}',
                 fontsize=8, color='red')
    ax.axis('off')

plt.suptitle('مهند الحربي — MNIST CNN Evaluation', fontsize=12, y=1.01)
plt.savefig('reports/confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved → reports/confusion_matrix.png")

# ── 5. Error pattern analysis ─────────────────────────────────
from collections import Counter
error_pairs = Counter(
    (y_test[i], y_pred[i]) for i in wrong_idx
)
print("\n  Top 5 most common errors (true → predicted):")
for (true, pred), count in error_pairs.most_common(5):
    print(f"    Digit {true} → predicted as {pred}: {count} times")

print(f"\n✓ Phase 3 complete.")
print(f"  Total errors  : {len(wrong_idx)} / {len(y_test)}")
print(f"  Test accuracy : {acc*100:.2f}%")
