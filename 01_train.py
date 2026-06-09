# =============================================================
# 01_train.py  —  Phase 2: Build & Train the CNN
# Student  : مهند الحربي
# Instructor: المهندس راشد العقيل
# Course   : AI Foundations Bootcamp — Week 10
# Note     : Generated with AI assistance (Claude) as permitted
# =============================================================

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json, os, time

# ── 1. Seeds ─────────────────────────────────────────────────
np.random.seed(42)
tf.random.set_seed(42)
print(f"TensorFlow version: {tf.__version__}")

# ── 2. Load & Preprocess Data ─────────────────────────────────
print("\n[1/5] Loading MNIST data...")
(X_full, y_full), (X_test, y_test) = keras.datasets.mnist.load_data()

# Scale pixels 0-255 → 0-1
X_full = X_full / 255.0
X_test  = X_test  / 255.0

# Add channel dimension: (N,28,28) → (N,28,28,1)
X_full = X_full[..., None]
X_test  = X_test[..., None]

# Split: first 6,000 → validation, rest → training
X_val,   y_val   = X_full[:6000],  y_full[:6000]
X_train, y_train = X_full[6000:],  y_full[6000:]

print(f"  Train : {X_train.shape}")
print(f"  Val   : {X_val.shape}")
print(f"  Test  : {X_test.shape}  (LOCKED until Phase 3)")

# ── 3. Model Builder ─────────────────────────────────────────
def build_model(dropout=0.0, filters=(32, 64)):
    """
    Small CNN for MNIST.
    Args:
        dropout : Dropout rate after Flatten (0 = no dropout)
        filters : Tuple of (conv1_filters, conv2_filters)
    """
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
    m.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return m

# ── 4. Experiment Table ────────────────────────────────────────
# Each row changes ONE variable at a time
experiments = [
    {"name": "Baseline (32/64, 5ep)",          "epochs": 5,  "dropout": 0.0,  "filters": (32, 64)},
    {"name": "More epochs (32/64, 10ep)",       "epochs": 10, "dropout": 0.0,  "filters": (32, 64)},
    {"name": "Dropout 0.25 (32/64, 10ep)",      "epochs": 10, "dropout": 0.25, "filters": (32, 64)},
    {"name": "Dropout 0.25 + larger (64/128)",  "epochs": 10, "dropout": 0.25, "filters": (64, 128)},
]

print("\n[2/5] Running experiments...")
results = []
best_history, best_model = None, None

for exp in experiments:
    print(f"\n  → {exp['name']}")
    np.random.seed(42); tf.random.set_seed(42)
    model = build_model(exp['dropout'], exp['filters'])
    h = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=exp['epochs'],
        batch_size=128,
        verbose=0
    )
    va = max(h.history['val_accuracy'])
    ta = max(h.history['accuracy'])
    results.append({
        "Experiment":  exp['name'],
        "Epochs":      exp['epochs'],
        "Dropout":     exp['dropout'],
        "Filters":     str(exp['filters']),
        "Val Acc":     round(va, 4),
        "Train Acc":   round(ta, 4),
    })
    print(f"     best val_acc = {va:.4f}")
    if exp['name'] == "Dropout 0.25 + larger (64/128)":
        best_history = h.history
        best_model   = model

# Print experiment table
print("\n\n====== EXPERIMENT TABLE ======")
print(f"{'Experiment':<42} {'Epochs':>6} {'Dropout':>8} {'Val Acc':>9} {'Train Acc':>10}")
print("-" * 78)
for r in results:
    print(f"{r['Experiment']:<42} {r['Epochs']:>6} {r['Dropout']:>8} {r['Val Acc']:>9.4f} {r['Train Acc']:>10.4f}")

os.makedirs('reports', exist_ok=True)
with open('reports/experiment_table.json', 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print("\nExperiment table saved → reports/experiment_table.json")

# ── 5. Training Curves ────────────────────────────────────────
print("\n[3/5] Saving training curves...")
fig, (a0, a1) = plt.subplots(1, 2, figsize=(12, 4))

a0.plot(best_history['accuracy'],     label='Train', color='#534AB7', linewidth=2)
a0.plot(best_history['val_accuracy'], label='Val',   color='#1D9E75', linestyle='--', linewidth=2)
a0.set_title('Accuracy over Epochs', fontsize=13)
a0.set_xlabel('Epoch'); a0.set_ylabel('Accuracy')
a0.legend(); a0.grid(True, alpha=0.3)

a1.plot(best_history['loss'],     label='Train', color='#534AB7', linewidth=2)
a1.plot(best_history['val_loss'], label='Val',   color='#1D9E75', linestyle='--', linewidth=2)
a1.set_title('Loss over Epochs', fontsize=13)
a1.set_xlabel('Epoch'); a1.set_ylabel('Loss')
a1.legend(); a1.grid(True, alpha=0.3)

plt.suptitle('مهند الحربي — MNIST CNN Training', fontsize=11, y=1.02)
plt.tight_layout()
plt.savefig('reports/training_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved → reports/training_curves.png")

# ── 6. Save Model ─────────────────────────────────────────────
print("\n[4/5] Saving model...")
os.makedirs('models', exist_ok=True)
best_model.save('models/mnist_cnn.h5')
print("  Saved → models/mnist_cnn.h5")

# ── 7. Quick latency check ────────────────────────────────────
print("\n[5/5] Latency check (single image, CPU)...")
sample = X_test[:1]
start = time.time()
for _ in range(100):
    best_model.predict(sample, verbose=0)
latency_ms = (time.time() - start) / 100 * 1000
print(f"  Average inference latency: {latency_ms:.1f} ms")

print("\n✓ Phase 2 complete. Run 02_evaluate.py next.")
