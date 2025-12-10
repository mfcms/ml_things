import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import shutil
from pathlib import Path

DATA_DIR = "ml_things/parrots/Birds dataset"
IMG_SIZE = 224
BATCH_SIZE = 8
EPOCHS_HEAD = 25
EPOCHS_FINE = 10
CLASSES = ["amazon green parrot", "gray parrot", "macaw", "white parrot"]
NUM_CLASSES = len(CLASSES)

TMP_DIR = Path("ml_things/parrots/Birds dataset")
TMP_DIR.mkdir(exist_ok=True)
for split in ["train", "val", "test"]:
    (TMP_DIR / split).mkdir(parents=True, exist_ok=True)
    for cls in CLASSES:
        (TMP_DIR / split / cls).mkdir(exist_ok=True)

image_paths = []
labels = []

for cls in CLASSES:
    cls_dir = Path(DATA_DIR) / cls
    for img_path in cls_dir.iterdir():
        if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
            image_paths.append(str(img_path))
            labels.append(cls)

image_paths, labels = shuffle(image_paths, labels, random_state=42)

X_temp, X_test, y_temp, y_test = train_test_split(
    image_paths, labels, test_size=0.15, stratify=labels, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.1765, stratify=y_temp, random_state=42  # 15% от исходного
)

def copy_files(file_list, label_list, target_dir):
    for fp, lbl in zip(file_list, label_list):
        dst = TMP_DIR / target_dir / lbl / Path(fp).name
        shutil.copy(fp, dst)

copy_files(X_train, y_train, "train")
copy_files(X_val, y_val, "val")
copy_files(X_test, y_test, "test")

print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],
    fill_mode='nearest'
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    str(TMP_DIR / "train"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_gen = val_test_datagen.flow_from_directory(
    str(TMP_DIR / "val"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

test_gen = val_test_datagen.flow_from_directory(
    str(TMP_DIR / "test"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

base_model = EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.trainable = False  # Заморозка

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
output = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(model.summary())

callbacks = [
    EarlyStopping(patience=8, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(factor=0.5, patience=4, verbose=1),
    ModelCheckpoint("best_parrot_model.h5", save_best_only=True, verbose=1)
]

print("\n Этап 1: Обучение головы (30 эпох)...")
history1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_HEAD,
    callbacks=callbacks,
    verbose=1
)

print("\n  Этап 2: Fine-tuning...")
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),  # Маленький LR!
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history2 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_HEAD + EPOCHS_FINE,
    initial_epoch=EPOCHS_HEAD,
    callbacks=callbacks,
    verbose=1
)

print("\n Оценка на тестовом наборе...")
test_loss, test_acc = model.evaluate(test_gen, verbose=0)
print(f" Тестовая точность: {test_acc:.4f} ({test_acc*100:.1f}%)")

test_gen.reset()
y_pred = model.predict(test_gen)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_gen.classes

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

print("\n Classification Report:")
print(classification_report(y_true, y_pred_classes, target_names=CLASSES))

cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=CLASSES, yticklabels=CLASSES, cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("Confusion matrix saved as 'confusion_matrix.png'")

acc = history1.history['accuracy'] + history2.history['accuracy']
val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
loss = history1.history['loss'] + history2.history['loss']
val_loss = history1.history['val_loss'] + history2.history['val_loss']

epochs_range = range(len(acc))
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.axvline(x=EPOCHS_HEAD-1, color='gray', linestyle='--', label='Start Fine-tuning')
plt.legend()
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.axvline(x=EPOCHS_HEAD-1, color='gray', linestyle='--')
plt.legend()
plt.title('Training and Validation Loss')
plt.savefig('training_history.png')
print("Training history saved as 'training_history.png'")