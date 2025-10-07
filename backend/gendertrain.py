import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import os
import datetime

# Paths
DATASET_DIR = r"C:\Users\Bharat ravi\OneDrive\Desktop\DHANUSH\datasets"   # <- put "male" and "female" folders inside here
OUTPUT_DIR = "models"

IMG_SIZE = (100, 100)  # keep same as PCOS models
BATCH_SIZE = 16
EPOCHS = 20

def main():
    # Load dataset
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    class_names = train_ds.class_names
    print("Detected classes:", class_names)

    # Save labels for backend
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    labels_path = os.path.join(OUTPUT_DIR, "gender.labels.txt")
    with open(labels_path, "w") as f:
        for name in class_names:
            f.write(name + "\n")
    print(f"Labels saved to {labels_path}")

    # Normalize
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.map(lambda x, y: (x/255.0, y)).cache().shuffle(500).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (x/255.0, y)).cache().prefetch(buffer_size=AUTOTUNE)

    # Simple CNN (you can swap for MobileNetV2/EfficientNet if dataset is big)
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=IMG_SIZE + (3,)),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(len(class_names), activation='softmax')
    ])

    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    # Callbacks
    checkpoint_path = os.path.join(OUTPUT_DIR, "gender_classifier.h5")
    callbacks = [
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, verbose=1),
        keras.callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True, verbose=1)
    ]

    # Train
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # Save final model
    model.save(checkpoint_path)
    print(f"Model saved to {checkpoint_path}")

if __name__ == "__main__":
    main()
