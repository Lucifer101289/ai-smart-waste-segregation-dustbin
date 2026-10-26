from pathlib import Path
import tensorflow as tf

DATA = Path("data")
TRAIN_DIR = DATA / "train"
VAL_DIR = DATA / "validation"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42

for folder in (TRAIN_DIR / "organic", TRAIN_DIR / "recyclable",
               VAL_DIR / "organic", VAL_DIR / "recyclable"):
    if not folder.exists():
        raise SystemExit(
            f"Missing folder: {folder}. Create the dataset folders described in README.md "
            "and add labelled images before training."
        )

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
    label_mode="binary", seed=SEED
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
    label_mode="binary", seed=SEED, shuffle=False
)
class_names = train_ds.class_names
print("Class order:", class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.08),
    tf.keras.layers.RandomZoom(0.1),
])

base = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,), include_top=False,
    weights="imagenet"
)
base.trainable = False

inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
x = augmentation(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x * 255.0)
x = base(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
model = tf.keras.Model(inputs, outputs)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

model.fit(train_ds, validation_data=val_ds, epochs=8)
model.save("waste_model.keras")
print("Saved model to waste_model.keras")
