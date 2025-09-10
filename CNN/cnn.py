"""
Improved Cat vs. Dog Classification
-----------------------------------
Enhancements:
- Deeper CNN architecture
- Stronger augmentation
- EarlyStopping & ReduceLROnPlateau
- Evaluation on Train, Val, Test
- Saves best model
"""

import os
import random
import logging
from typing import Tuple, List

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.losses import SparseCategoricalCrossentropy
from keras.optimizers import Adam
from keras.regularizers import L2
from keras.models import Sequential
from keras import layers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use("Agg")
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, ConfusionMatrixDisplay
)

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
tf.get_logger().setLevel(logging.ERROR)

IMAGE_SIZE: Tuple[int, int] = (160, 160)  # upgraded resolution
BATCH_SIZE: int = 64
TRAIN_RATIO: float = 0.7
VAL_RATIO: float = 0.15
SEED: int = 42

CAT_DIR: str = "./cats_set"
DOG_DIR: str = "./dogs_set"

TRAIN_CSV: str = "train_indices.csv"
VAL_CSV: str = "val_indices.csv"
TEST_CSV: str = "test_indices.csv"

os.makedirs("results", exist_ok=True)

# ---------------------------------------------------------------------
# Dataset splitting
# ---------------------------------------------------------------------
def _extract_id(filename: str) -> str:
    parts = filename.split(".")
    if len(parts) >= 2:
        return parts[1]
    return filename

def split_dataset(cat_dir: str, dog_dir: str,
                  train_ratio: float = TRAIN_RATIO,
                  val_ratio: float = VAL_RATIO,
                  seed: int = SEED):
    cat_files: List[str] = [f for f in os.listdir(cat_dir) if f.lower().endswith((".jpg", ".jpeg"))]
    dog_files: List[str] = [f for f in os.listdir(dog_dir) if f.lower().endswith((".jpg", ".jpeg"))]

    random.seed(seed)
    random.shuffle(cat_files)
    random.shuffle(dog_files)

    def partition(files: List[str]):
        n = len(files)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        return files[:n_train], files[n_train:n_train+n_val], files[n_train+n_val:]

    cat_train, cat_val, cat_test = partition(cat_files)
    dog_train, dog_val, dog_test = partition(dog_files)

    def build_records(file_list: List[str], label: str):
        return [{"class": label, "id": _extract_id(f)} for f in file_list]

    train_records = build_records(cat_train, "cat") + build_records(dog_train, "dog")
    val_records = build_records(cat_val, "cat") + build_records(dog_val, "dog")
    test_records = build_records(cat_test, "cat") + build_records(dog_test, "dog")

    return pd.DataFrame(train_records), pd.DataFrame(val_records), pd.DataFrame(test_records)

def save_split_indexes(df_train, df_val, df_test):
    df_train.to_csv(TRAIN_CSV, index=False)
    df_val.to_csv(VAL_CSV, index=False)
    df_test.to_csv(TEST_CSV, index=False)

# ---------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------
def build_dataset_from_index(df: pd.DataFrame,
                             cat_dir: str,
                             dog_dir: str,
                             batch_size: int = BATCH_SIZE) -> tf.data.Dataset:
    class_to_label = {"cat": 0, "dog": 1}
    file_paths, labels = [], []
    for _, row in df.iterrows():
        cls, id_str = row["class"], str(row["id"])
        filename = f"{cls}.{id_str}.jpg"
        if cls == "cat":
            file_paths.append(os.path.join(cat_dir, filename))
        else:
            file_paths.append(os.path.join(dog_dir, filename))
        labels.append(class_to_label[cls])

    paths_tensor = tf.constant(file_paths)
    labels_tensor = tf.constant(labels, dtype=tf.int64)
    ds = tf.data.Dataset.from_tensor_slices((paths_tensor, labels_tensor))

    def _load_image(path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, IMAGE_SIZE)
        return image, label

    ds = ds.map(_load_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(len(file_paths), seed=SEED)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

# ---------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------
def draw_histogram(classes, df_train, df_val, df_test):
    counts = {cls: 0 for cls in classes}
    for df in (df_train, df_val, df_test):
        for cls in classes:
            counts[cls] += len(df[df["class"] == cls])

    plt.figure(figsize=(8, 5))
    plt.bar(counts.keys(), counts.values(), width=0.5, edgecolor="black", color="cyan")
    plt.title("Distribution of Images per Class")
    plt.xlabel("Class")
    plt.ylabel("Number of Samples")
    plt.tight_layout()
    plt.savefig("results/class_distribution.png")
    plt.close()

# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------
def make_model(num_classes: int) -> Sequential:
    data_augmentation = Sequential([
        layers.RandomFlip("horizontal", input_shape=(*IMAGE_SIZE, 3)),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomContrast(0.2),
    ])

    model = Sequential([
        data_augmentation,
        layers.Rescaling(1./255),

        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),

        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),

        layers.Conv2D(256, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),

        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(256, activation="relu", kernel_regularizer=L2(0.0005)),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss=SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model

# ---------------------------------------------------------------------
# Training & Evaluation
# ---------------------------------------------------------------------
def train_and_evaluate():
    if not (os.path.isfile(TRAIN_CSV) and os.path.isfile(VAL_CSV) and os.path.isfile(TEST_CSV)):
        df_train, df_val, df_test = split_dataset(CAT_DIR, DOG_DIR)
        save_split_indexes(df_train, df_val, df_test)
    else:
        df_train = pd.read_csv(TRAIN_CSV)
        df_val = pd.read_csv(VAL_CSV)
        df_test = pd.read_csv(TEST_CSV)

    classes = ["cat", "dog"]
    input_train = build_dataset_from_index(df_train, CAT_DIR, DOG_DIR)
    input_val = build_dataset_from_index(df_val, CAT_DIR, DOG_DIR)
    input_test = build_dataset_from_index(df_test, CAT_DIR, DOG_DIR)

    draw_histogram(classes, df_train, df_val, df_test)

    model = make_model(len(classes))

    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True, monitor="val_loss"),
        ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6, monitor="val_loss"),
        ModelCheckpoint("results/best_model.keras", save_best_only=True, monitor="val_loss")
    ]

    history = model.fit(input_train, validation_data=input_val, epochs=100, verbose=1, callbacks=callbacks)

    # accuracy/loss curves
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="train")
    plt.plot(history.history["val_accuracy"], label="val")
    plt.title("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="val")
    plt.title("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/training_curves.png")
    plt.close()

    # evaluation function
    def evaluate_dataset(ds, split_name, out_file):
        y_true, y_pred = [], []
        for img_batch, lab_batch in ds:
            y_true.extend(lab_batch.numpy())
            preds = np.argmax(model.predict(img_batch, verbose=0), axis=1)
            y_pred.extend(preds)
        print(f"{split_name} Accuracy: {100*accuracy_score(y_true, y_pred):.2f}%")
        print(f"{split_name} Precision: {100*precision_score(y_true, y_pred, average='weighted'):.2f}%")
        print(f"{split_name} Recall: {100*recall_score(y_true, y_pred, average='weighted'):.2f}%")
        print(f"{split_name} F1: {100*f1_score(y_true, y_pred, average='weighted'):.2f}%")
        cm = confusion_matrix(y_true, y_pred, normalize="true")
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes).plot(cmap="Blues")
        plt.title(f"Confusion Matrix - {split_name}")
        plt.savefig(out_file)
        plt.close()

    evaluate_dataset(input_train, "Training", "results/conf_matrix_train.png")
    evaluate_dataset(input_val, "Validation", "results/conf_matrix_val.png")
    evaluate_dataset(input_test, "Test", "results/conf_matrix_test.png")

if __name__ == "__main__":
    train_and_evaluate()
