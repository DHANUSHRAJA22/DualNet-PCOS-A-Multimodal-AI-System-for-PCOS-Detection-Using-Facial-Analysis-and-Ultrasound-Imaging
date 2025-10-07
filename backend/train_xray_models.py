# -*- coding: utf-8 -*-
"""
Train PCOS X-ray classifier (VGG16 only) to match backend/managers/xray_manager.py.

Outputs:
  backend/models/xray/pcos_vgg16.h5
  backend/models/xray/pcos_vgg16.labels.txt

Dataset layout:
  <DATASET_ROOT>/
      infected/        # PCOS-positive
      notinfected/     # PCOS-negative

We fix class index 0 = "non_pcos" (notinfected), 1 = "pcos" (infected).
"""

import os
from pathlib import Path
import numpy as np
import tensorflow as tf

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

AUTOTUNE = tf.data.AUTOTUNE
SEED = 1337
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
VAL_SPLIT = 0.2
LABELS_FILE_LINES = ["non_pcos", "pcos"]  # fixed order

# ------------------------
# Dataset
# ------------------------
def make_datasets(root: Path):
    root = Path(root)
    if not (root / "infected").exists() or not (root / "notinfected").exists():
        raise SystemExit("Expected 'infected/' and 'notinfected/' inside dataset root")

    class_names = ["notinfected", "infected"]

    train_ds = tf.keras.utils.image_dataset_from_directory(
        root,
        labels="inferred",
        label_mode="categorical",
        class_names=class_names,
        validation_split=VAL_SPLIT,
        subset="training",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        root,
        labels="inferred",
        label_mode="categorical",
        class_names=class_names,
        validation_split=VAL_SPLIT,
        subset="validation",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )

    train_ds = train_ds.shuffle(1024, seed=SEED).cache().prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)

    class_counts = np.array([0, 0], dtype=np.int64)
    for _, y in train_ds.unbatch():
        class_counts += tf.cast(tf.round(y), tf.int64).numpy()

    class_counts = class_counts.astype(np.float32)
    cw = {i: float(sum(class_counts) / (2.0 * max(1.0, class_counts[i]))) for i in (0, 1)}
    print(f"[info] class counts (notinfected, infected): {class_counts.tolist()}")
    print(f"[info] class weights: {cw}")
    return train_ds, val_ds, cw

# ------------------------
# Model
# ------------------------
def build_vgg16():
    base = tf.keras.applications.VGG16(
        include_top=False, weights="imagenet", input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = tf.keras.applications.vgg16.preprocess_input(inputs)
    x = base(x, training=False)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(2, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs, name="pcos_vgg16")
    return model, base

def compile_model(model, lr):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(name="acc"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="prec"),
            tf.keras.metrics.Recall(name="rec"),
        ],
    )

# ------------------------
# Train routine
# ------------------------
def train_vgg16(dataset_root: str, out_dir: str, epochs_top=5, epochs_ft=10):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, cw = make_datasets(Path(dataset_root))
    model, base = build_vgg16()

    # Phase 1: train head
    compile_model(model, 1e-3)
    print("\n[phase 1] train classifier head (VGG16 frozen)")
    model.fit(train_ds, validation_data=val_ds,
              epochs=epochs_top, class_weight=cw,
              callbacks=[
                  tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max",
                                                   patience=5, restore_best_weights=True),
                  tf.keras.callbacks.ReduceLROnPlateau(monitor="val_auc", mode="max",
                                                       factor=0.5, patience=2)
              ],
              verbose=1)

    # Phase 2: fine-tune
    print("\n[phase 2] fine-tune last VGG16 block")
    n = len(base.layers)
    cut = int(0.7 * n)
    for i, l in enumerate(base.layers):
        l.trainable = i >= cut
    compile_model(model, 1e-5)
    model.fit(train_ds, validation_data=val_ds,
              epochs=epochs_ft, class_weight=cw,
              callbacks=[
                  tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max",
                                                   patience=5, restore_best_weights=True),
                  tf.keras.callbacks.ReduceLROnPlateau(monitor="val_auc", mode="max",
                                                       factor=0.5, patience=2)
              ],
              verbose=1)

    # Save weights-only file
    weights_out = out_dir / "pcos_vgg16.h5"
    model.save_weights(str(weights_out))
    print(f"[saved] {weights_out}")

    labels_out = out_dir / "pcos_vgg16.labels.txt"
    labels_out.write_text("\n".join(LABELS_FILE_LINES), encoding="utf-8")
    print(f"[saved] {labels_out}")

    # Validation report
    print("\n[eval] final validation metrics:")
    eval_res = model.evaluate(val_ds, verbose=0)
    for k, v in zip(model.metrics_names, eval_res):
        print(f"  {k:>8}: {v:.4f}")

# ------------------------
# Run
# ------------------------
if __name__ == "__main__":
    dataset = r"C:\Users\Bharat ravi\OneDrive\Desktop\DHANUSH\weigts\pcos traiing dataset"
    output = "backend/models/xray"
    train_vgg16(dataset, output)
