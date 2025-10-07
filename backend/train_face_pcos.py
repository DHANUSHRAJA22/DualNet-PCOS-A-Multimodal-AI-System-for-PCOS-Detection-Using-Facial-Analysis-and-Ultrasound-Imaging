# train_vgg16_face_pcos.py
# Trains a VGG16 classifier for PCOS (2 classes) and saves it to backend/models/face

import os
from pathlib import Path
import tensorflow as tf
from tensorflow import keras

# -----------------------------
# EDIT THIS ONLY IF YOUR DATA PATH CHANGES
# -----------------------------
DATA_DIR = r"C:\Users\Bharat ravi\OneDrive\Desktop\DHANUSH\pcos_prediction_vgg_resnet-main\train_directory2"

# -----------------------------
# Output locations (match your backend)
# -----------------------------
OUT_DIR = Path("backend/models/face")
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = OUT_DIR / "vgg16_weights_tf_finetuned_pcos.h5"
LABELS_PATH = OUT_DIR / "face_classifier.labels.txt"

# -----------------------------
# Training config
# -----------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS_STAGE1 = 15
EPOCHS_STAGE2 = 10
SEED = 123

# -----------------------------
# Datasets
# -----------------------------
print("Loading datasets...")
train_raw = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    label_mode="categorical",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="training",
    seed=SEED,
)
val_raw = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    label_mode="categorical",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
)

class_names = train_raw.class_names
num_classes = len(class_names)
print("Class order:", class_names)

# Write labels file so the backend can read them
with open(LABELS_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(class_names))

AUTOTUNE = tf.data.AUTOTUNE

# VGG16 expects inputs preprocessed with its own routine (0..255 -> centered/bgr)
@tf.function
def _preprocess(x, y):
    x = tf.cast(x, tf.float32)
    x = tf.keras.applications.vgg16.preprocess_input(x)
    return x, y

@tf.function
def _augment(x, y):
    x = tf.image.random_flip_left_right(x, seed=SEED)
    x = tf.image.random_brightness(x, max_delta=10.0)  # in "pixel" units for 0..255
    x = tf.image.random_contrast(x, lower=0.9, upper=1.1)
    return x, y

train_ds = (
    train_raw
    .map(_augment, num_parallel_calls=AUTOTUNE)
    .map(_preprocess, num_parallel_calls=AUTOTUNE)
    .shuffle(1000, seed=SEED)
    .prefetch(AUTOTUNE)
)
val_ds = (
    val_raw
    .map(_preprocess, num_parallel_calls=AUTOTUNE)
    .prefetch(AUTOTUNE)
)

# -----------------------------
# Class weights (handle any class imbalance)
# -----------------------------
counts = [len(list((Path(DATA_DIR) / c).glob("**/*.*"))) for c in class_names]
total = sum(counts)
class_weight = {i: total / (num_classes * counts[i]) for i in range(num_classes)}
print("Class counts:", dict(zip(class_names, counts)))
print("Class weights:", class_weight)

# -----------------------------
# Build VGG16 model
# -----------------------------
def build_vgg16_model():
    input_shape = (IMG_SIZE[0], IMG_SIZE[1], 3)
    base = tf.keras.applications.VGG16(include_top=False, weights="imagenet", input_shape=input_shape)
    base.trainable = False  # Stage 1: freeze

    inputs = keras.Input(shape=input_shape)
    x = base(inputs, training=False)
    x = keras.layers.Flatten()(x)                       # VGG-style head
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dropout(0.5)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="pcos_head")(x)
    model = keras.Model(inputs, outputs, name="vgg16_pcos_classifier")
    return model, base

tf.keras.backend.clear_session()
model, base = build_vgg16_model()

metrics = [
    "accuracy",
    tf.keras.metrics.AUC(name="auc"),
    tf.keras.metrics.Precision(name="precision"),
    tf.keras.metrics.Recall(name="recall"),
]

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="categorical_crossentropy",
    metrics=metrics,
)

ckpt = keras.callbacks.ModelCheckpoint(
    filepath=str(MODEL_PATH),
    monitor="val_auc",
    mode="max",
    save_best_only=True,
    verbose=1,
)
early = keras.callbacks.EarlyStopping(
    monitor="val_auc",
    mode="max",
    patience=7,
    restore_best_weights=True,
    verbose=1,
)
reduce = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=3, verbose=1
)

print("\nStage 1 (frozen base)…")
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_STAGE1,
    class_weight=class_weight,
    callbacks=[ckpt, early, reduce],
    verbose=1,
)

# -----------------------------
# Fine-tune: unfreeze Block 5 of VGG16
# -----------------------------
for layer in base.layers:
    layer.trainable = layer.name.startswith("block5")

model.compile(
    optimizer=keras.optimizers.Adam(1e-5),
    loss="categorical_crossentropy",
    metrics=metrics,
)

print("\nStage 2 (fine-tune Block 5)…")
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_STAGE2,
    class_weight=class_weight,
    callbacks=[ckpt, early],
    verbose=1,
)

# -----------------------------
# Final evaluation
# -----------------------------
val_loss, val_acc, val_auc, val_prec, val_rec = model.evaluate(val_ds, verbose=0)
print(f"\n✅ VGG16 saved → {MODEL_PATH}")
print(f"   Val: acc {val_acc:.3f} | AUC {val_auc:.3f} | P {val_prec:.3f} | R {val_rec:.3f}")
print(f"\nLabels written to: {LABELS_PATH}")
