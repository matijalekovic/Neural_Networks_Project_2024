# rice_mlp_cv.py
import itertools
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg' if you have Qt installed
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
warnings.filterwarnings("ignore")

# -------------------------
# Config / Hyperparam grid
# -------------------------
CSV_PATH = "Rice.csv"  # update path if needed
RANDOM_STATE = 42
NUM_EPOCHS = 80
BATCH_SIZE = 32
KFOLD = 5

# hyperparameter grid to search (at least 3 hyperparameters as required)
LRS = [0.001, 0.005, 0.01]
HIDDEN_SIZES = [32, 64, 128]
DROPOUTS = [0.2, 0.3, 0.5]

# -------------------------
# Utility functions & model
# -------------------------
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_size, num_classes, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
    def forward(self, x):
        return self.net(x)

def train_epoch(model, optimizer, criterion, X, y, batch_size):
    model.train()
    idx = np.random.permutation(len(X))
    losses = []
    accs = []
    for i in range(0, len(X), batch_size):
        batch_idx = idx[i:i+batch_size]
        xb = torch.tensor(X[batch_idx], dtype=torch.float32)
        yb = torch.tensor(y[batch_idx], dtype=torch.long)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        preds = out.argmax(dim=1).numpy()
        accs.append((preds == yb.numpy()).mean())
    return np.mean(losses), np.mean(accs)

def eval_model(model, criterion, X, y, batch_size=1024):
    model.eval()
    with torch.no_grad():
        xb = torch.tensor(X, dtype=torch.float32)
        yb = torch.tensor(y, dtype=torch.long)
        out = model(xb)
        loss = criterion(out, yb).item()
        preds = out.argmax(dim=1).numpy()
        acc = (preds == y).mean()
    return loss, acc, preds

# -------------------------
# Load & preprocess
# -------------------------
df = pd.read_csv(CSV_PATH)
# Provided columns: AREA,PERIMETER,MAJORAXIS,MINORAXIS,ECCENTRICITY,CONVEX_AREA,EXTENT,CLASS
# If headerless or different, adjust accordingly
feature_cols = [c for c in df.columns if c.strip().upper() != "CLASS"]
X = df[feature_cols].values
y_raw = df["CLASS"].values

le = LabelEncoder()
y = le.fit_transform(y_raw)
class_names = le.classes_

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split (final held-out test set)
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

# Plot histogram of class distribution (full dataset)
plt.figure(figsize=(6,4))
sns.countplot(x=y_raw)
plt.title("Class distribution (all samples)")
plt.xlabel("Class")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -------------------------
# Grid search with Stratified K-Fold
# -------------------------
skf = StratifiedKFold(n_splits=KFOLD, shuffle=True, random_state=RANDOM_STATE)
best_mean_val_acc = -1
best_config = None
results = []

total_configs = len(LRS)*len(HIDDEN_SIZES)*len(DROPOUTS)
print(f"Testing {total_configs} hyperparameter combinations...")

start_all = time.time()
for lr, hs, dp in itertools.product(LRS, HIDDEN_SIZES, DROPOUTS):
    val_accs = []
    fold_idx = 0
    for train_idx, val_idx in skf.split(X_trainval, y_trainval):
        fold_idx += 1
        X_tr, X_val = X_trainval[train_idx], X_trainval[val_idx]
        y_tr, y_val = y_trainval[train_idx], y_trainval[val_idx]

        model = MLP(input_dim=X.shape[1], hidden_size=hs, num_classes=len(class_names), dropout=dp)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        best_val_acc_fold = 0.0
        # train for NUM_EPOCHS, track best val accuracy for this fold
        for epoch in range(NUM_EPOCHS):
            train_loss, train_acc = train_epoch(model, optimizer, criterion, X_tr, y_tr, BATCH_SIZE)
            val_loss, val_acc, _ = eval_model(model, criterion, X_val, y_val)
            if val_acc > best_val_acc_fold:
                best_val_acc_fold = val_acc
                best_model_state = {k: v.cpu().clone() for k,v in model.state_dict().items()}
        val_accs.append(best_val_acc_fold)
    mean_val_acc = np.mean(val_accs)
    results.append(((lr, hs, dp), mean_val_acc))
    print(f"lr={lr}, hidden={hs}, dropout={dp} -> mean val acc: {mean_val_acc:.4f}")
    if mean_val_acc > best_mean_val_acc:
        best_mean_val_acc = mean_val_acc
        best_config = (lr, hs, dp)
end_all = time.time()
print(f"Grid search finished in {end_all - start_all:.1f}s")
print("Best config:", best_config, "with mean val acc:", best_mean_val_acc)

# -------------------------
# Train final model on trainval with best config, track curves
# -------------------------
best_lr, best_hs, best_dp = best_config
model = MLP(input_dim=X.shape[1], hidden_size=best_hs, num_classes=len(class_names), dropout=best_dp)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=best_lr)

train_losses, val_losses = [], []
train_accs, val_accs = [], []

# split trainval into internal train/val for plotting curves (stratified)
X_tr, X_val, y_tr, y_val = train_test_split(X_trainval, y_trainval, test_size=0.2, stratify=y_trainval, random_state=RANDOM_STATE)

for epoch in range(NUM_EPOCHS):
    t_loss, t_acc = train_epoch(model, optimizer, criterion, X_tr, y_tr, BATCH_SIZE)
    v_loss, v_acc, _ = eval_model(model, criterion, X_val, y_val)
    train_losses.append(t_loss)
    train_accs.append(t_acc)
    val_losses.append(v_loss)
    val_accs.append(v_acc)
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train acc {t_acc:.4f} | Val acc {v_acc:.4f}")

# -------------------------
# Evaluation: train set & test set
# -------------------------
train_loss, train_acc, train_preds = eval_model(model, criterion, X_tr, y_tr)
test_loss, test_acc, test_preds = eval_model(model, criterion, X_test, y_test)

print("\nFinal results:")
print(f"Train acc: {train_acc:.4f}, Test acc: {test_acc:.4f}")

# classification report on test set
print("\nClassification report (test):")
print(classification_report(y_test, test_preds, target_names=class_names))

# confusion matrices
cm_train = confusion_matrix(y_tr, train_preds)
cm_test = confusion_matrix(y_test, test_preds)

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
sns.heatmap(cm_train, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix - Train (internal split)")
plt.xlabel("Predicted")
plt.ylabel("True")

plt.subplot(1,2,2)
sns.heatmap(cm_test, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix - Test")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()

# Plot training/validation curves
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(train_losses, label="train loss")
plt.plot(val_losses, label="val loss")
plt.title("Loss vs Epochs")
plt.xlabel("Epoch")
plt.legend()

plt.subplot(1,2,2)
plt.plot(train_accs, label="train acc")
plt.plot(val_accs, label="val acc")
plt.title("Accuracy vs Epochs")
plt.xlabel("Epoch")
plt.legend()
plt.tight_layout()
plt.show()

# Print metrics table (accuracy, precision, recall, f1) for test
prec, rec, f1, _ = precision_recall_fscore_support(y_test, test_preds, average='weighted', zero_division=0)
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Precision (weighted): {prec:.4f}")
print(f"Test Recall/Sensitivity (weighted): {rec:.4f}")
print(f"Test F1-score (weighted): {f1:.4f}")

# Save results summary
print("\nBest hyperparameters found:")
print(f"Learning rate: {best_lr}, Hidden size: {best_hs}, Dropout: {best_dp}")
