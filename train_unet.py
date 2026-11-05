import os
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow.keras import layers, models, backend as K
import albumentations as A

# ============================================================
# CONFIG
# ============================================================
IMG_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 40
LR = 1e-4

PROJECT_DIR = Path(__file__).resolve().parent
BASE_DIR = PROJECT_DIR / "dataset" / "brain_tumor_unet"
IMG_DIR = BASE_DIR / "images"
MASK_DIR = BASE_DIR / "masks"

MODEL_PATH = PROJECT_DIR / "models" / "unetpp_fast_brain_tumor.keras"
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

tf.keras.mixed_precision.set_global_policy("mixed_float16")

# ============================================================
# BRAIN EXTRACTION (SKULL STRIPPING)
# ============================================================
def extract_brain(img):
    img = (img * 255).astype(np.uint8)
    _, thresh = cv2.threshold(
        img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, 2)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return img / 255.0

    mask = np.zeros_like(img)
    largest = max(contours, key=cv2.contourArea)
    cv2.drawContours(mask, [largest], -1, 255, -1)

    brain = cv2.bitwise_and(img, img, mask=mask)
    return brain / 255.0

# ============================================================
# DATA GENERATOR
# ============================================================
augment = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.RandomBrightnessContrast(p=0.3),
])

def data_generator(img_dir, mask_dir, batch_size):
    names = sorted(os.listdir(img_dir))
    while True:
        np.random.shuffle(names)
        for i in range(0, len(names), batch_size):
            imgs, masks = [], []
            for name in names[i:i+batch_size]:
                img = cv2.imread(os.path.join(img_dir, name),
                                 cv2.IMREAD_GRAYSCALE)
                mask = cv2.imread(os.path.join(mask_dir, name),
                                  cv2.IMREAD_GRAYSCALE)

                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))

                img = img.astype(np.float32) / 255.0
                img = extract_brain(img)
                mask = (mask > 0).astype(np.float32)

                aug = augment(image=img, mask=mask)
                img = aug["image"]
                mask = aug["mask"]

                imgs.append(img[..., np.newaxis])
                masks.append(mask[..., np.newaxis])

            yield np.array(imgs), np.array(masks)

# ============================================================
# LOSS FUNCTIONS
# ============================================================
def dice_coef(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    return (2. * intersection + 1) / (K.sum(y_true) + K.sum(y_pred) + 1)

def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
    return -K.mean(
        alpha * y_true * K.pow(1 - y_pred, gamma) * K.log(y_pred)
    )

def total_loss(y_true, y_pred):
    return focal_loss(y_true, y_pred) + (1 - dice_coef(y_true, y_pred))

# ============================================================
# U-NET++ (NESTED U-NET)
# ============================================================
def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    return x

def build_unetpp():
    inputs = layers.Input((IMG_SIZE, IMG_SIZE, 1))

    x00 = conv_block(inputs, 64)
    p0 = layers.MaxPooling2D()(x00)

    x10 = conv_block(p0, 128)
    p1 = layers.MaxPooling2D()(x10)

    x20 = conv_block(p1, 256)

    x01 = conv_block(
        layers.Concatenate()([x00, layers.UpSampling2D()(x10)]), 64
    )

    x11 = conv_block(
        layers.Concatenate()([x10, layers.UpSampling2D()(x20)]), 128
    )

    x02 = conv_block(
        layers.Concatenate()([x00, x01, layers.UpSampling2D()(x11)]), 64
    )

    output = layers.Conv2D(
        1, 1, activation="sigmoid", dtype="float32"
    )(x02)

    return models.Model(inputs, output)

# ============================================================
# TRAIN
# ============================================================
image_names = {name for name in os.listdir(IMG_DIR) if os.path.isfile(IMG_DIR / name)}
mask_names = {name for name in os.listdir(MASK_DIR) if os.path.isfile(MASK_DIR / name)}
if not image_names or image_names != mask_names:
    raise ValueError("The U-Net dataset must contain matching, non-empty image and mask filenames.")
train_gen = data_generator(str(IMG_DIR), str(MASK_DIR), BATCH_SIZE)
steps = max(1, len(image_names) // BATCH_SIZE)

model = build_unetpp()
model.compile(
    optimizer=tf.keras.optimizers.Adam(LR),
    loss=total_loss,
    metrics=[dice_coef]
)

model.summary()

model.fit(
    train_gen,
    steps_per_epoch=steps,
    epochs=EPOCHS
)

model.save(MODEL_PATH)
print("✅ Model saved at:", MODEL_PATH)
