# model.py
import os
import tensorflow as tf
from tensorflow import keras

DATA_DIR = r"C:\Users\Bharat ravi\OneDrive\Desktop\DHANUSH\pcos_prediction_vgg_resnet-main\train_directory2"
IMG_SIZE = (100, 100)
BATCH_SIZE = 32
EPOCHS_STAGE1 = 25
EPOCHS_STAGE2 = 10
BEST_MODEL_PATH = "models/pcos_detector_158.h5"
LABELS_PATH = "models/pcos_detector_158.labels.txt"
SEED = 123

os.makedirs("models", exist_ok=True)

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    label_mode="categorical",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="training",
    seed=SEED,
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    label_mode="categorical",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
)

class_names = train_ds.class_names
with open(LABELS_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(class_names))
print("Labels:", class_names)

AUTOTUNE = tf.data.AUTOTUNE
def normalize(x, y): return tf.cast(x, tf.float32) / 255.0, y
train_ds = train_ds.map(normalize).cache().shuffle(1000).prefetch(AUTOTUNE)
val_ds = val_ds.map(normalize).cache().prefetch(AUTOTUNE)

data_augment = keras.Sequential([
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.05),
    keras.layers.RandomZoom(0.10),
])

base = keras.applications.VGG16(
    include_top=False, weights="imagenet",
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
)
base.trainable = False

inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = data_augment(inputs)
x = base(x, training=False)
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(256, activation="relu")(x)
x = keras.layers.Dropout(0.5)(x)
outputs = keras.layers.Dense(len(class_names), activation="softmax")(x)

model = keras.Model(inputs, outputs)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

callbacks = [
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=1),
    keras.callbacks.ModelCheckpoint(filepath=BEST_MODEL_PATH, monitor="val_loss", save_best_only=True),
]

model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_STAGE1, callbacks=callbacks)

base.trainable = True
for layer in base.layers[:-4]:
    layer.trainable = False
model.compile(optimizer=keras.optimizers.Adam(1e-5), loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_STAGE2, callbacks=callbacks)

print(f"Model saved to {BEST_MODEL_PATH}, labels to {LABELS_PATH}")
