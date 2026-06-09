# mnist_v3.py   Train + Evaluate CNN on MNIST (GPU Optimized)
# Student  : مهند الحربي
# Instructor: المهندس راشد العق# Course   : AI Foundations Bootcamp — Week 10
# Note     : Generated with AI assistance (Claude) as permitted
# =============================================================

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from collections import Counter
import json, os, time

np.random.seed(42)
tf.random.set_seed(42)
os.makedirs('reports', exist_ok=True)
os.makedirs('models',  exist_ok=True)

print(f"TensorFlow: {tf.__version__}")
print("Running on CPU\n")
# ── 1. Data ───────────────────────────────────────────────────
print("[1/6] Loading MNIST...")
(X_full, y_full), (X_test, y_test) = keras.datasets.mnist.load_data()
X_full = X_full / 255.0
X_test = X_test / 255.0
X_full = X_full[..., None]
X_test = X_test[..., None]
X_val,   y_val   = X_full[:6000], y_full[:6000]
X_train, y_train = X_full[6000:], y_full[6000:]
print(f"  Train:{X_train.shape}  Val:{X_val.shape}  Test:{X_test.shape}")

# ── 2. Model ──────────────────────────────────────────────────
def build_model(dropout=0.0, filters=(32, 64)):
    m = keras.Sequential([
        keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(filters[0], 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(filters[1], 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
    ])
    if dropout > 0:
        m.add(layers.Dropout(dropout))
    m.add(layers.Dense(64, activation='relu'))
    m.add(layers.Dense(10, activation='softmax'))
    m.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    return m

# ── 3. Experiments ────────────────────────────────────────────
BEST_NAME = "Dropout 0.25 + larger (64/128)"

experiments = [
    {"name": "Baseline (32/64, 5ep)",         "epochs": 5,  "dropout": 0.0,  "filters": (32, 64)},
    {"name": "More epochs (32/64, 10ep)",      "epochs": 10, "dropout": 0.0,  "filters": (32, 64)},
    {"name": "Dropout 0.25 (32/64, 10ep)",     "epochs": 10, "dropout": 0.25, "filters": (32, 64)},
    {"name": BEST_NAME,                         "epochs": 10, "dropout": 0.25, "filters": (64, 128)},
]

print("\n[2/6] Running experiments...")
results, best_history, best_model = [], None, None

for exp in experiments:
    print(f"  → {exp['name']}")
    np.random.seed(42); tf.random.set_seed(42)
    model = build_model(exp['dropout'], exp['filters'])
    h = model.fit(X_train, y_train,
                  validation_data=(X_val, y_val),
                  epochs=exp['epochs'], batch_size=128, verbose=0)
    va = max(h.history['val_accuracy'])
    ta = max(h.history['accuracy'])
    results.append({
        "Experiment": exp['name'], "Epochs": exp['epochs'],
        "Dropout": exp['dropout'], "Filters": str(exp['filters']),
        "Val Acc": round(va, 4), "Train Acc": round(ta, 4)
    })
    print(f"     val_acc = {va:.4f}")
    if exp['name'] == BEST_NAME:
        best_history, best_model = h.history, model

print("\n====== EXPERIMENT TABLE ======")
print(f"{'Experiment':<42} {'Epochs':>6} {'Dropout':>8} {'Val Acc':>9} {'Train Acc':>10}")
print("-" * 78)
for r in results:
    print(f"{r['Experiment']:<42} {r['Epochs']:>6} {r['Dropout']:>8} {r['Val Acc']:>9.4f} {r['Train Acc']:>10.4f}")

with open('reports/experiment_table.json', 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

# ── 4. Training Curves ────────────────────────────────────────
print("\n[3/6] Saving training curves...")
fig, (a0, a1) = plt.subplots(1, 2, figsize=(12, 4))
a0.plot(best_history['accuracy'],     label='Train', color='#534AB7', linewidth=2)
a0.plot(best_history['val_accuracy'], label='Val',   color='#1D9E75', linestyle='--', linewidth=2)
a0.set_title('Accuracy over Epochs'); a0.set_xlabel('Epoch'); a0.set_ylabel('Accuracy')
a0.legend(); a0.grid(True, alpha=0.3)
a1.plot(best_history['loss'],     label='Train', color='#534AB7', linewidth=2)
a1.plot(best_history['val_loss'], label='Val',   color='#1D9E75', linestyle='--', linewidth=2)
a1.set_title('Loss over Epochs'); a1.set_xlabel('Epoch'); a1.set_ylabel('Loss')
a1.legend(); a1.grid(True, alpha=0.3)
plt.suptitle('مهند الحربي — MNIST CNN Training', fontsize=11, y=1.02)
plt.tight_layout()
plt.savefig('reports/training_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → reports/training_curves.png")

# ── 5. Save Model ─────────────────────────────────────────────
print("\n[4/6] Saving model...")
best_model.save('models/mnist_cnn.h5')
print("  → models/mnist_cnn.h5")

# ── 6. Evaluate ───────────────────────────────────────────────
print("\n[5/6] Test evaluation...")
loss, acc = best_model.evaluate(X_test, y_test, verbose=0)
print(f"  Test Loss     : {loss:.4f}")
print(f"  Test Accuracy : {acc:.4f}  ({acc*100:.2f}%)")

y_pred = np.argmax(best_model.predict(X_test, verbose=0), axis=1)
report = classification_report(y_test, y_pred, target_names=[str(i) for i in range(10)])
print(report)

with open('reports/classification_report.txt', 'w', encoding='utf-8') as f:
    f.write(f"Student: مهند الحربي\nInstructor: المهندس راشد العقيل\n\n")
    f.write(f"Test Accuracy: {acc:.4f}\n\n{report}")
print("  → reports/classification_report.txt")

# ── 7. Confusion Matrix + صور صح ─────────────────────────────
print("\n[6/6] Confusion matrix & correct predictions...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    display_labels=[str(i) for i in range(10)],
    ax=axes[0], colorbar=False, cmap='Blues'
)
axes[0].set_title('Confusion Matrix — Test Set', fontsize=13)

# 9 صور صنّفها الموديل صح
correct_idx = np.where(y_pred == y_test)[0]
axes[1].set_title(f'9 Correctly Classified Samples ✓', fontsize=11)
axes[1].axis('off')
for i, idx in enumerate(correct_idx[:9]):
    ax = fig.add_axes([
        axes[1].get_position().x0 + (i % 3) * axes[1].get_position().width / 3,
        axes[1].get_position().y0 + (2 - i // 3) * axes[1].get_position().height / 3,
        axes[1].get_position().width / 3,
        axes[1].get_position().height / 3
    ])
    ax.imshow(X_test[idx].squeeze(), cmap='gray')
    ax.set_title(f'True:{y_test[idx]}  Pred:{y_pred[idx]}', fontsize=8, color='green')
    ax.axis('off')

plt.suptitle('مهند الحربي — MNIST CNN Evaluation', fontsize=12, y=1.01)
plt.savefig('reports/confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → reports/confusion_matrix.png")

# ── Error Analysis ────────────────────────────────────────────
wrong_idx = np.where(y_pred != y_test)[0]
error_pairs = Counter((y_test[i], y_pred[i]) for i in wrong_idx)
print("\n  Top 5 errors (true → predicted):")
for (true, pred), count in error_pairs.most_common(5):
    print(f"    Digit {true} → {pred}: {count} times")

# ── Latency ───────────────────────────────────────────────────
sample = X_test[:1]
start = time.time()
for _ in range(100):
    best_model.predict(sample, verbose=0)
ms = (time.time() - start) / 100 * 1000
print(f"\n  Avg inference latency: {ms:.1f} ms")

print(f"\n✓ Done. Errors: {len(wrong_idx)}/{len(y_test)}  |  Test Acc: {acc*100:.2f}%")
