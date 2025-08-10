#!/usr/bin/env python3
"""
train_kitchenware_cnn.py

Complete script to train a CNN (from scratch) on a dataset described by:
- train.csv  (image id, class)
- test.csv   (image id)
- sample_submission.csv
- images/    (all images named <id>.<ext>)

Outputs:
- best_model.keras  (best checkpoint by val_loss)
- final_model.keras (final model after training)
- submission.csv    (predicted labels for test.csv in sample_submission format)
- training plots PNGs

Author: ChatGPT (adapted for your dataset)
"""

import os
import random
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # safe for headless environments
import matplotlib.pyplot as plt
import math
from collections import Counter

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, callbacks, optimizers
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

# ----------------- Config -----------------
SEED = 42
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 30
VAL_SPLIT = 0.20
L2_REG = 1e-4
DROPOUT_RATE = 0.3
AUTOTUNE = tf.data.AUTOTUNE
DATA_DIR = Path(".")
IMAGES_DIR = DATA_DIR / "images"
TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"
SAMPLE_SUB = DATA_DIR / "sample_submission.csv"
BEST_MODEL_PATH = "best_model.keras"
FINAL_MODEL_PATH = "final_model.keras"
PLOT_HISTORY = "training_history.png"
CM_VAL = "confusion_val.png"

# ----------------- Reproducibility -----------------
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ----------------- Helpers -----------------
def find_image_path(img_id, images_dir=IMAGES_DIR):
    """Try common extensions for an image id and return the existing path or None."""
    for ext in (".jpg", ".jpeg", ".png", ".bmp"):
        p = images_dir / (str(img_id) + ext)
        if p.exists():
            return str(p)
    # sometimes CSV ids already include extension
    p = images_dir / str(img_id)
    if p.exists():
        return str(p)
    return None

def infer_column_names(df):
    """Return (id_col, label_col or None) - tries common names."""
    cols = [c.lower() for c in df.columns]
    id_col = None
    label_col = None
    for c in df.columns:
        if c.lower() in ("id", "image_id", "img_id", "image", "filename", "file"):
            id_col = c
            break
    if id_col is None:
        id_col = df.columns[0]  # fallback

    for c in df.columns:
        if c.lower() in ("label", "class", "target"):
            label_col = c
            break
    # If no obvious label column, return None for label_col
    return id_col, label_col

def plot_history(history, out=PLOT_HISTORY):
    plt.figure(figsize=(12,5))
    # accuracy
    plt.subplot(1,2,1)
    plt.plot(history.history.get('accuracy', []), label='train_acc')
    plt.plot(history.history.get('val_accuracy', []), label='val_acc', linestyle='--')
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    # loss
    plt.subplot(1,2,2)
    plt.plot(history.history.get('loss', []), label='train_loss')
    plt.plot(history.history.get('val_loss', []), label='val_loss', linestyle='--')
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out)
    plt.close()

def plot_confusion(y_true, y_pred, classes, out=CM_VAL, title="Confusion matrix"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(max(6, len(classes)*0.8), max(4, len(classes)*0.5)))
    import seaborn as sns
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap="Blues")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()

# ----------------- Load CSVs -----------------
if not TRAIN_CSV.exists():
    raise FileNotFoundError(f"{TRAIN_CSV} not found in working dir.")
if not TEST_CSV.exists():
    raise FileNotFoundError(f"{TEST_CSV} not found in working dir.")
if not IMAGES_DIR.exists():
    raise FileNotFoundError(f"{IMAGES_DIR} folder not found in working dir.")

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

train_id_col, train_label_col = infer_column_names(train_df)
test_id_col, _ = infer_column_names(test_df)

if train_label_col is None:
    raise RuntimeError("Could not find label column in train.csv. Expected column named 'label'/'class'/'target' etc.")

print(f"Detected train id column: '{train_id_col}', label column: '{train_label_col}'")
print(f"Detected test id column: '{test_id_col}'")

# Build full filepaths and filter missing images
train_df['filepath'] = train_df[train_id_col].apply(lambda x: find_image_path(x))
missing_train = train_df['filepath'].isna().sum()
if missing_train > 0:
    print(f"Warning: {missing_train} train images not found in {IMAGES_DIR}. These rows will be dropped.")
train_df = train_df.dropna(subset=['filepath']).reset_index(drop=True)

test_df['filepath'] = test_df[test_id_col].apply(lambda x: find_image_path(x))
missing_test = test_df['filepath'].isna().sum()
if missing_test > 0:
    print(f"Warning: {missing_test} test images not found in {IMAGES_DIR}. These rows will be dropped.")
test_df = test_df.dropna(subset=['filepath']).reset_index(drop=True)

# Map labels to integer indices
classes = sorted(train_df[train_label_col].unique().tolist())
num_classes = len(classes)
class_to_idx = {c:i for i,c in enumerate(classes)}
idx_to_class = {i:c for c,i in class_to_idx.items()}
train_df['label_idx'] = train_df[train_label_col].map(class_to_idx)

print(f"Found {len(train_df)} training images spanning {num_classes} classes.")
print("Class distribution:", Counter(train_df['label_idx'].values))

# ----------------- Train/Validation split (stratified) -----------------
train_paths = train_df['filepath'].tolist()
train_labels = train_df['label_idx'].tolist()

train_paths, val_paths, train_labels, val_labels = train_test_split(
    train_paths, train_labels,
    test_size=VAL_SPLIT,
    stratify=train_labels,
    random_state=SEED
)

print(f"Training samples: {len(train_paths)}, Validation samples: {len(val_paths)}")

# ----------------- Build tf.data pipelines -----------------
def preprocess_image(path, label=None, img_size=IMG_SIZE):
    """Read, decode, resize, scale to [0,1]."""
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)  # supports jpeg; if png exists decode will still work
    image = tf.image.convert_image_dtype(image, tf.float32)  # [0,1]
    image = tf.image.resize(image, img_size)
    if label is None:
        return image
    else:
        return image, label

# Augmentation layer (Keras preprocessing)
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal", seed=SEED),
    layers.RandomRotation(0.08, seed=SEED),
    layers.RandomZoom(0.08, seed=SEED),
    layers.RandomTranslation(0.12, 0.12, seed=SEED)
], name="data_augmentation")

def make_dataset(paths, labels=None, batch_size=BATCH_SIZE, training=False):
    if labels is None:
        ds = tf.data.Dataset.from_tensor_slices(paths)
        ds = ds.map(lambda p: preprocess_image(p, label=None), num_parallel_calls=AUTOTUNE)
        ds = ds.batch(batch_size)
        ds = ds.prefetch(AUTOTUNE)
        return ds
    else:
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        ds = ds.shuffle(len(paths), seed=SEED) if training else ds
        ds = ds.map(lambda p, l: (preprocess_image(p, l)), num_parallel_calls=AUTOTUNE)
        if training:
            ds = ds.map(lambda x,y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
        ds = ds.batch(batch_size)
        ds = ds.prefetch(AUTOTUNE)
        return ds

train_ds = make_dataset(train_paths, train_labels, training=True)
val_ds = make_dataset(val_paths, val_labels, training=False)
test_paths = test_df['filepath'].tolist()
test_ids = test_df[test_id_col].tolist()
test_ds = make_dataset(test_paths, labels=None, batch_size=BATCH_SIZE)

# ----------------- Class weights -----------------
# helps with class imbalance
cw = class_weight.compute_class_weight(class_weight='balanced', classes=np.arange(num_classes), y=np.array(train_labels))
class_weight_dict = {i: float(w) for i,w in enumerate(cw)}
print("Class weights:", class_weight_dict)

# ----------------- Model building (BN before activation, GAP) -----------------
def build_cnn(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), num_classes=num_classes, l2_reg=L2_REG, dropout=DROPOUT_RATE):
    l2 = regularizers.l2(l2_reg)
    inp = layers.Input(shape=input_shape)
    x = inp
    for filters in [32, 64, 128, 256]:
        x = layers.Conv2D(filters, (3,3), padding="same", activation=None, kernel_regularizer=l2)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D((2,2))(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation=None, kernel_regularizer=l2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs=inp, outputs=out, name="KitchenwareCNN")
    return model

model = build_cnn()
model.compile(optimizer=optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ----------------- Callbacks -----------------
cb_early = callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1)
cb_ckpt = callbacks.ModelCheckpoint(BEST_MODEL_PATH, monitor="val_loss", save_best_only=True, verbose=1)
cb_rlp = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1)
cbs = [cb_early, cb_ckpt, cb_rlp]

# ----------------- Train -----------------
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=cbs,
    class_weight=class_weight_dict,
    verbose=1
)

model.save(FINAL_MODEL_PATH)

# ----------------- Plot training curves -----------------
plot_history(history, out=PLOT_HISTORY)

# ----------------- Evaluate on validation set and print metrics -----------------
def predict_on_dataset(model, ds, true_labels=None):
    proba = model.predict(ds, verbose=1)
    preds = np.argmax(proba, axis=1)
    if true_labels is None:
        return preds, proba
    else:
        return preds, proba

# Build val dataset without batching shuffling in order to extract true labels
val_ds_unbatched = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
val_ds_unbatched = val_ds_unbatched.map(lambda p,l: (preprocess_image(p,l)), num_parallel_calls=AUTOTUNE)
val_ds_unbatched = val_ds_unbatched.batch(BATCH_SIZE).prefetch(AUTOTUNE)

y_pred_val, proba_val = predict_on_dataset(model, val_ds_unbatched)
y_true_val = np.array(val_labels)

print("\n--- Validation metrics ---")
print_metrics_flag = True
acc_val = accuracy_score(y_true_val, y_pred_val)
prec, rec, f1, _ = precision_recall_fscore_support(y_true_val, y_pred_val, average='weighted', zero_division=0)
print(f"Val Accuracy: {acc_val:.4f}")
print(f"Val Precision (weighted): {prec:.4f}")
print(f"Val Recall (weighted): {rec:.4f}")
print(f"Val F1 (weighted): {f1:.4f}")
print("\nClassification report (validation):")
print(classification_report(y_true_val, y_pred_val, target_names=classes, zero_division=0))

# Confusion matrix plot
plot_confusion(y_true_val, y_pred_val, classes, out=CM_VAL, title="Confusion Matrix - Validation")

# ----------------- Predict test set and write submission -----------------
test_preds, test_proba = predict_on_dataset(model, test_ds)
test_pred_labels = [idx_to_class[int(p)] for p in test_preds]

# Prepare submission using sample_submission template if present
if SAMPLE_SUB.exists():
    sub_df = pd.read_csv(SAMPLE_SUB)
    id_col, _ = infer_column_names(sub_df)
    # If sample has a label column use the second column name, else create 'label' column
    label_col = None
    for c in sub_df.columns:
        if c != id_col:
            label_col = c
            break
    if label_col is None:
        label_col = "label"
        sub_df[label_col] = ""
    # Build new submission keyed by id_col
    # Map test_df ids to predicted labels
    pred_map = dict(zip(test_ids, test_pred_labels))
    # Fill predictions for rows where id is present in test_df
    sub_df[label_col] = sub_df[id_col].apply(lambda x: pred_map.get(x, ""))
    out_sub = "submission.csv"
    sub_df.to_csv(out_sub, index=False)
    print(f"Saved submission to {out_sub} (based on sample_submission.csv)")
else:
    # fallback simple submission
    out = pd.DataFrame({"id": test_ids, "label": test_pred_labels})
    out.to_csv("submission.csv", index=False)
    print("Saved submission.csv")

print("Done.")
