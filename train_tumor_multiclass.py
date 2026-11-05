import tensorflow as tf
import numpy as np
import json
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight

# =========================
# CONFIG
# =========================
IMG_SIZE = 224
BATCH = 32
EPOCHS_HEAD = 10
EPOCHS_FINE = 15

DATA_DIR = "/content/dataset/tumor_multiclass"
MODEL_PATH = "/content/tumor_multiclass.keras"
CLASS_INDEX_PATH = "/content/class_indices.json"

# =========================
# DATA GENERATORS
# =========================
train_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True
)

val_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train = train_gen.flow_from_directory(
    DATA_DIR + "/train",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH,
    class_mode="categorical"
)

val = val_gen.flow_from_directory(
    DATA_DIR + "/val",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH,
    class_mode="categorical",
    shuffle=False
)

# =========================
# SAVE CLASS INDICES
# =========================
with open(CLASS_INDEX_PATH, "w") as f:
    json.dump(train.class_indices, f)

print("Saved class indices:", train.class_indices)

# =========================
# CLASS WEIGHTS
# =========================
weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train.classes),
    y=train.classes
)
class_weight = dict(zip(np.unique(train.classes), weights))
print("Class weights:", class_weight)

# =========================
# MODEL
# =========================
base = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base.trainable = False

x = GlobalAveragePooling2D()(base.output)
x = Dense(256, activation="relu")(x)
x = Dropout(0.4)(x)
out = Dense(train.num_classes, activation="softmax")(x)

model = Model(base.input, out)

# =========================
# PHASE 1 – TRAIN HEAD
# =========================
model.compile(
    optimizer=Adam(1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train,
    validation_data=val,
    epochs=EPOCHS_HEAD,
    class_weight=class_weight,
    callbacks=[
        EarlyStopping(patience=4, restore_best_weights=True),
        ReduceLROnPlateau(patience=2, factor=0.3)
    ]
)

# =========================
# PHASE 2 – FINE TUNING
# =========================
base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=Adam(1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train,
    validation_data=val,
    epochs=EPOCHS_FINE,
    class_weight=class_weight,
    callbacks=[
        EarlyStopping(patience=4, restore_best_weights=True),
        ModelCheckpoint(MODEL_PATH, save_best_only=True)
    ]
)

model.save(MODEL_PATH)
print("✅ Multiclass model saved:", MODEL_PATH)
