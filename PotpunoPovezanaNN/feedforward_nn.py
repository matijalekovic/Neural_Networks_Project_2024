import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns

# ============================
# CONFIGURATION
# ============================
# Hyperparameters influence how the network learns and generalizes.
# We will tune:
# - Number of neurons in the first hidden layer: affects model capacity and ability to capture complexity.
# - Activation function in the first hidden layer: introduces non-linearity essential for learning complex patterns.
# - Learning rate: controls step size during weight updates, impacting convergence speed and stability.

CSV_PATH = "Rice.csv"
EPOCHS = 200           # Maximum number of iterations over the dataset to train
BATCH_SIZE = 32        # Mini-batches speed up training and provide noise that can help generalization
PATIENCE = 10          # Early stopping patience to prevent overfitting
KFOLD = 5              # Cross-validation folds for robust hyperparameter selection
RANDOM_STATE = 42      # For reproducibility of splits and shuffling

NEURONS_OPTIONS = [32, 64, 128]
ACTIVATIONS = {
    "relu": nn.ReLU(),      # Rectified Linear Unit (ReLU) allows efficient gradient flow and sparsity
    "tanh": nn.Tanh(),      # Hyperbolic tangent outputs in [-1,1], can center data but saturates for large inputs
    "sigmoid": nn.Sigmoid() # Sigmoid outputs in [0,1], prone to vanishing gradients but historically popular
}
LR_OPTIONS = [0.001, 0.005, 0.01]

# ============================
# L-INFINITY NORM LOSS FUNCTION (CRITERION)
# ============================
class LInfLoss(nn.Module):
    """
    The L∞ norm loss measures the maximum absolute error across all predicted class probabilities
    and true labels for a given sample.
    This loss function focuses on the worst-case prediction error,
    encouraging the network to minimize the largest deviation from correct classification.

    Unlike traditional cross-entropy loss that measures average error,
    the L∞ norm loss is robust to outliers but can be more challenging to optimize.
    """

    def __init__(self):
        super().__init__()

    def forward(self, outputs, targets):
        # Convert targets to one-hot encoding for comparison with outputs
        targets_onehot = torch.zeros_like(outputs)
        targets_onehot.scatter_(1, targets.unsqueeze(1), 1.0)

        # Convert raw network outputs to probabilities using softmax activation
        probs = torch.softmax(outputs, dim=1)

        # Return maximum absolute difference across all classes and samples
        return torch.max(torch.abs(probs - targets_onehot))

# ============================
# FULLY CONNECTED NEURAL NETWORK ARCHITECTURE
# ============================
class MLP(nn.Module):
    """
    Multilayer Perceptron (MLP) is a classic feedforward neural network model.
    It consists of stacked layers of neurons, where each neuron computes
    a weighted sum of inputs plus bias, followed by a nonlinear activation function.

    Here, we use:
    - First hidden layer: user-defined number of neurons and activation function (hyperparameters)
    - Second hidden layer: fixed 64 neurons with ReLU activation to add depth and representation power
    - Output layer: linear neurons equal to number of classes, providing raw scores (logits)

    No softmax in the output layer because the loss function applies softmax internally.

    The network learns by adjusting weights to minimize the loss between predicted outputs and true labels.
    """

    def __init__(self, input_dim, hidden1_neurons, hidden1_activation, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden1_neurons)
        self.act1 = hidden1_activation
        self.fc2 = nn.Linear(hidden1_neurons, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        # Input flows forward through the network:
        # Linear transform -> Activation -> next layer, etc.
        x = self.act1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)   # Output logits (raw scores for each class)
        return x

# ============================
# DATA PREPROCESSING
# ============================
df = pd.read_csv(CSV_PATH)

# Neural networks generally require numerical input features scaled to similar ranges
features = ["AREA","PERIMETER","MAJORAXIS","MINORAXIS","ECCENTRICITY","CONVEX_AREA","EXTENT"]
X = df[features].values

# Encode categorical target labels as integers for classification
y_raw = df["CLASS"].values
le = LabelEncoder()
y = le.fit_transform(y_raw)
class_names = le.classes_

# Standardize features to zero mean and unit variance to stabilize and speed up training
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split dataset into training+validation and test subsets
# Test set is held out to evaluate final generalization performance
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

# Visualize class distribution to understand data balance and possible class imbalance issues
plt.figure(figsize=(6,4))
sns.countplot(x=y_raw)
plt.title("Class Distribution")
plt.show()

# ============================
# TRAINING WITH EARLY STOPPING
# ============================
def train_with_early_stopping(model, optimizer, criterion, X_train, y_train, X_val, y_val, max_epochs=EPOCHS, patience=PATIENCE):
    """
    Train the neural network using mini-batch gradient descent.
    Early stopping halts training when validation accuracy stops improving,
    which helps prevent overfitting (model memorizing training data).

    During each epoch:
    - The network learns by adjusting weights to reduce loss (here L∞ loss).
    - Validation accuracy guides early stopping.
    """

    best_val_acc = 0
    best_state = None
    patience_counter = 0

    for epoch in range(max_epochs):
        model.train()
        idx = np.random.permutation(len(X_train))

        for i in range(0, len(X_train), BATCH_SIZE):
            batch_idx = idx[i:i+BATCH_SIZE]
            xb = torch.tensor(X_train[batch_idx], dtype=torch.float32)
            yb = torch.tensor(y_train[batch_idx], dtype=torch.long)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()       # Backpropagation: compute gradients
            optimizer.step()      # Gradient descent step: update weights

        # Evaluate validation accuracy (model generalization on unseen data)
        model.eval()
        with torch.no_grad():
            xb_val = torch.tensor(X_val, dtype=torch.float32)
            yb_val = torch.tensor(y_val, dtype=torch.long)
            val_outputs = model(xb_val)
            val_preds = val_outputs.argmax(dim=1).numpy()
            val_acc = accuracy_score(y_val, val_preds)

        # Save best model weights and apply early stopping if no improvement
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_state)
    return best_val_acc

# ============================
# CROSS-VALIDATION FOR HYPERPARAMETER OPTIMIZATION
# ============================
skf = StratifiedKFold(n_splits=KFOLD, shuffle=True, random_state=RANDOM_STATE)
best_config = None
best_val_mean_acc = -1

# Cross-validation divides data into folds to ensure that hyperparameter tuning
# generalizes across different subsets of data, preventing overfitting on a single split.
print("Starting hyperparameter search over neurons, activation, and learning rate...")
for neurons in NEURONS_OPTIONS:
    for act_name, act_fn in ACTIVATIONS.items():
        for lr in LR_OPTIONS:
            fold_accs = []
            for train_idx, val_idx in skf.split(X_trainval, y_trainval):
                X_tr, X_val = X_trainval[train_idx], X_trainval[val_idx]
                y_tr, y_val = y_trainval[train_idx], y_trainval[val_idx]

                model = MLP(input_dim=X.shape[1], hidden1_neurons=neurons, hidden1_activation=act_fn, num_classes=len(class_names))
                optimizer = optim.Adam(model.parameters(), lr=lr)
                criterion = LInfLoss()

                val_acc = train_with_early_stopping(model, optimizer, criterion, X_tr, y_tr, X_val, y_val)
                fold_accs.append(val_acc)

            mean_acc = np.mean(fold_accs)
            print(f"neurons={neurons}, activation={act_name}, lr={lr} -> mean val acc={mean_acc:.4f}")

            # Select the hyperparameters with highest mean validation accuracy
            if mean_acc > best_val_mean_acc:
                best_val_mean_acc = mean_acc
                best_config = (neurons, act_name, lr)

print("\nBest hyperparameter combination found:")
print(f"Neurons: {best_config[0]}, Activation: {best_config[1]}, Learning rate: {best_config[2]}")
print(f"Mean validation accuracy: {best_val_mean_acc:.4f}")

# ============================
# FINAL TRAINING WITH OPTIMAL HYPERPARAMETERS
# ============================
best_neurons, best_act_name, best_lr = best_config
best_act_fn = ACTIVATIONS[best_act_name]

model = MLP(input_dim=X.shape[1], hidden1_neurons=best_neurons, hidden1_activation=best_act_fn, num_classes=len(class_names))
optimizer = optim.Adam(model.parameters(), lr=best_lr)
criterion = LInfLoss()

# We hold out a validation set from train+val for monitoring training progress visually
X_tr, X_val, y_tr, y_val = train_test_split(X_trainval, y_trainval, test_size=0.2, stratify=y_trainval, random_state=RANDOM_STATE)

train_losses, val_losses = [], []
train_accs, val_accs = [], []

best_val_acc = 0
patience_counter = 0
best_state = None

print("\nTraining final model with best hyperparameters...")
for epoch in range(EPOCHS):
    model.train()
    idx = np.random.permutation(len(X_tr))
    batch_losses = []
    batch_accs = []
    for i in range(0, len(X_tr), BATCH_SIZE):
        batch_idx = idx[i:i+BATCH_SIZE]
        xb = torch.tensor(X_tr[batch_idx], dtype=torch.float32)
        yb = torch.tensor(y_tr[batch_idx], dtype=torch.long)
        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()      # Backpropagation computes gradients for all parameters
        optimizer.step()     # Optimizer updates parameters to minimize loss
        batch_losses.append(loss.item())
        batch_accs.append((outputs.argmax(dim=1) == yb).float().mean().item())

    # Track average training loss and accuracy this epoch
    train_losses.append(np.mean(batch_losses))
    train_accs.append(np.mean(batch_accs))

    # Validate performance after epoch
    model.eval()
    with torch.no_grad():
        xb_val = torch.tensor(X_val, dtype=torch.float32)
        yb_val = torch.tensor(y_val, dtype=torch.long)
        val_outputs = model(xb_val)
        val_loss = criterion(val_outputs, yb_val).item()
        val_preds = val_outputs.argmax(dim=1).numpy()
        val_acc = accuracy_score(y_val, val_preds)

    val_losses.append(val_loss)
    val_accs.append(val_acc)

    # Early stopping to avoid overfitting and save best model state
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state = model.state_dict()
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break

# Restore best weights
model.load_state_dict(best_state)

# ============================
# EVALUATION ON TEST SET
# ============================
model.eval()
with torch.no_grad():
    xb_test = torch.tensor(X_test, dtype=torch.float32)
    yb_test = torch.tensor(y_test, dtype=torch.long)
    test_outputs = model(xb_test)
    test_preds = test_outputs.argmax(dim=1).numpy()

# The classification report details precision, recall, and F1 score per class
print("\nClassification Report (Test Set):")
print(classification_report(y_test, test_preds, target_names=class_names))

# Confusion matrices show how often each class is confused with others
cm_train = confusion_matrix(y_tr, model(torch.tensor(X_tr, dtype=torch.float32)).argmax(dim=1).numpy())
cm_test = confusion_matrix(y_test, test_preds)

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
sns.heatmap(cm_train, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
plt.title("Confusion Matrix - Training Set")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")

plt.subplot(1,2,2)
sns.heatmap(cm_test, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
plt.title("Confusion Matrix - Test Set")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.tight_layout()
plt.show()

# Plot loss and accuracy curves over epochs to visualize learning behavior
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.title("Loss vs Epochs")
plt.xlabel("Epoch")
plt.ylabel("L∞ Loss")
plt.legend()

plt.subplot(1,2,2)
plt.plot(train_accs, label="Training Accuracy")
plt.plot(val_accs, label="Validation Accuracy")
plt.title("Accuracy vs Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.show()

# Calculate weighted precision, recall, and F1-score on test data
prec, rec, f1, _ = precision_recall_fscore_support(y_test, test_preds, average='weighted', zero_division=0)
print(f"Test Accuracy: {accuracy_score(y_test, test_preds):.4f}")
print(f"Test Precision (weighted): {prec:.4f}")
print(f"Test Recall/Sensitivity (weighted): {rec:.4f}")
print(f"Test F1-score (weighted): {f1:.4f}")

print("\nBest hyperparameters found:")
print(f"Neurons in first hidden layer: {best_neurons}")
print(f"Activation function in first hidden layer: {best_act_name}")
print(f"Learning rate: {best_lr}")
