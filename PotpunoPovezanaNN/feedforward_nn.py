import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for matplotlib (required in some environments)
import matplotlib.pyplot as plt
import seaborn as sns

# --- DATA LOADING AND PREPROCESSING ---
# Neural networks require numerical input data, usually normalized/scaled for better training stability.
# Here, we load the dataset and prepare features and target labels.

df = pd.read_csv("Rice.csv")

features = ["AREA","PERIMETER","MAJORAXIS","MINORAXIS","ECCENTRICITY","CONVEX_AREA","EXTENT"]
X = df[features].values  # Numeric feature vectors

y_raw = df["CLASS"].values  # Raw class labels (strings)

# Encode string labels into integers - neural networks work with numeric labels for classification
le = LabelEncoder()
y = le.fit_transform(y_raw)
class_names = le.classes_  # Save class names for interpretation later

print("Classes found:", class_names)
print("Samples per class in full dataset:")
print(pd.Series(y).value_counts().sort_index())  # Check class distribution - important to understand if data is balanced

# --- Visual: Histogram of full dataset class distribution ---
# Understanding class distribution visually helps detect imbalance which may bias training and evaluation.
plt.figure(figsize=(6,4))
sns.countplot(x=y_raw, order=class_names)
plt.title("Distribution of Samples by Class (Full Dataset)")
plt.xlabel("Class")
plt.ylabel("Number of Samples")
plt.show()

# Standardize features to zero mean and unit variance.
# This is critical as neural networks converge faster and more reliably when features are scaled.
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data into training+validation and test sets.
# Stratified split ensures class proportions are preserved, important for unbiased evaluation.
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("Train+Val class distribution:")
print(pd.Series(y_trainval).value_counts().sort_index())
print("Test class distribution:")
print(pd.Series(y_test).value_counts().sort_index())

# --- Visual: Histogram for Train+Val and Test splits ---
# Confirm splits maintain class distribution; important for meaningful validation and testing.
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
sns.countplot(x=y_trainval, order=range(len(class_names)))
plt.xticks(ticks=range(len(class_names)), labels=class_names)
plt.title("Train+Validation Set Class Distribution")
plt.xlabel("Class")
plt.ylabel("Samples")

plt.subplot(1,2,2)
sns.countplot(x=y_test, order=range(len(class_names)))
plt.xticks(ticks=range(len(class_names)), labels=class_names)
plt.title("Test Set Class Distribution")
plt.xlabel("Class")
plt.ylabel("Samples")
plt.tight_layout()
plt.show()

# --- MODEL DEFINITION ---
# We define a multilayer fully connected feedforward neural network (MLP).
# Feedforward networks learn complex nonlinear mappings from input features to output classes.

class MLP(nn.Module):
    def __init__(self, input_dim, hidden1_neurons, hidden1_activation, num_classes):
        super().__init__()
        # First hidden layer: linear transform + chosen activation function.
        # The number of neurons controls model capacity: too few underfits, too many risks overfitting.
        self.fc1 = nn.Linear(input_dim, hidden1_neurons)
        self.act1 = hidden1_activation  # Activation adds non-linearity, enabling learning complex patterns.

        # Second hidden layer fixed at 64 neurons with ReLU activation for simplicity and good gradient flow.
        self.fc2 = nn.Linear(hidden1_neurons, 64)
        self.relu2 = nn.ReLU()

        # Output layer: linear layer with one neuron per class.
        # No activation here because CrossEntropyLoss expects raw logits and internally applies softmax.
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        # Forward pass: apply layers and activations in sequence.
        x = self.act1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

# Criterion function: CrossEntropyLoss.
# This is the standard loss for multi-class classification, combining LogSoftmax and Negative Log-Likelihood loss.
# It measures how well predicted class probabilities match true classes.
criterion = nn.CrossEntropyLoss()

# --- TRAINING FUNCTION WITH TRACKING OF ACCURACY FOR PLOTTING ---
# Implements mini-batch gradient descent with early stopping to prevent overfitting.

def train_with_early_stopping(model, optimizer, criterion, X_train, y_train, X_val, y_val, max_epochs=200, patience=10):
    """
    Trains model on X_train/y_train, validates on X_val/y_val.
    Tracks accuracy per epoch and stops training if validation accuracy does not improve for 'patience' epochs.
    Early stopping is critical to avoid overfitting and wasting computation after convergence.
    """
    best_val_acc = 0
    best_state = None
    patience_counter = 0

    train_acc_history = []
    val_acc_history = []

    for epoch in range(max_epochs):
        model.train()  # Enable training mode (affects dropout, batchnorm if present)

        # Shuffle training data indices for better SGD performance
        idx = np.random.permutation(len(X_train))
        batch_size = 32
        # Mini-batch gradient descent for stable, scalable training and noise in gradients that aids generalization
        for i in range(0, len(X_train), batch_size):
            batch_idx = idx[i:i+batch_size]
            xb = torch.tensor(X_train[batch_idx], dtype=torch.float32)
            yb = torch.tensor(y_train[batch_idx], dtype=torch.long)

            optimizer.zero_grad()  # Clear gradients before backward pass
            outputs = model(xb)    # Forward pass
            loss = criterion(outputs, yb)  # Compute loss
            loss.backward()        # Backpropagation computes gradients
            optimizer.step()       # Update weights with optimizer (Adam chosen for adaptive learning rates and fast convergence)

        model.eval()  # Evaluation mode disables dropout, batchnorm updates, etc.
        with torch.no_grad():
            # Compute training accuracy on entire training data for monitoring
            xb_train = torch.tensor(X_train, dtype=torch.float32)
            yb_train = torch.tensor(y_train, dtype=torch.long)
            train_outputs = model(xb_train)
            train_preds = train_outputs.argmax(dim=1).numpy()
            train_acc = accuracy_score(y_train, train_preds)
            train_acc_history.append(train_acc)

            # Compute validation accuracy for early stopping and generalization check
            xb_val = torch.tensor(X_val, dtype=torch.float32)
            yb_val = torch.tensor(y_val, dtype=torch.long)
            val_outputs = model(xb_val)
            val_preds = val_outputs.argmax(dim=1).numpy()
            val_acc = accuracy_score(y_val, val_preds)
            val_acc_history.append(val_acc)

        # Early stopping logic:
        # If validation accuracy improves, save the model and reset patience counter.
        # Else increment patience counter; stop if patience exceeded.
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Restore best model weights found during training
    model.load_state_dict(best_state)
    return best_val_acc, train_acc_history, val_acc_history

#=================================================================================
# --- HYPERPARAMETER SEARCH ---
# Search over multiple combinations of key hyperparameters:
# 1. Number of neurons in first hidden layer controls model capacity.
# 2. Activation function influences non-linearity and gradient flow.
# 3. Learning rate controls step size of optimizer updates.
# Hyperparameter tuning is essential to find the best performing network configuration.
#=================================================================================

NEURONS_OPTIONS = [32, 64, 128]
ACTIVATIONS = {
    "relu": nn.ReLU(),      # ReLU avoids vanishing gradient, promotes sparsity and computational efficiency
    "tanh": nn.Tanh(),      # Tanh centers data at zero, helps some networks but saturates for large inputs
    "sigmoid": nn.Sigmoid() # Sigmoid historically common, but prone to vanishing gradients and slower training
}
LR_OPTIONS = [0.001, 0.005, 0.01]

# Stratified K-Fold splits maintain class distribution in folds,
# important for unbiased validation in classification tasks.
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

best_val_acc = 0
best_config = None

print("Starting hyperparameter search over neurons, activation, and learning rate...")

for neurons in NEURONS_OPTIONS:
    for act_name, act_fn in ACTIVATIONS.items():
        for lr in LR_OPTIONS:
            fold_accs = []
            # 5-fold cross-validation to robustly evaluate each hyperparameter combination.
            for train_idx, val_idx in skf.split(X_trainval, y_trainval):
                X_tr, X_val = X_trainval[train_idx], X_trainval[val_idx]
                y_tr, y_val = y_trainval[train_idx], y_trainval[val_idx]

                model = MLP(input_dim=X.shape[1], hidden1_neurons=neurons, hidden1_activation=act_fn, num_classes=len(class_names))
                optimizer = optim.Adam(model.parameters(), lr=lr)  # Adam optimizer adapts learning rates per parameter

                val_acc, _, _ = train_with_early_stopping(model, optimizer, criterion, X_tr, y_tr, X_val, y_val)
                fold_accs.append(val_acc)

            mean_acc = np.mean(fold_accs)
            print(f"neurons={neurons}, activation={act_name}, lr={lr} -> mean val acc={mean_acc:.4f}")

            if mean_acc > best_val_acc:
                best_val_acc = mean_acc
                best_config = (neurons, act_fn, act_name, lr)

print("\nBest hyperparameter combination found:")
print(f"Neurons: {best_config[0]}, Activation: {best_config[2]}, Learning rate: {best_config[3]}")
print(f"Mean validation accuracy: {best_val_acc:.4f}")

# --- FINAL MODEL TRAINING WITH PLOTTING ---

neurons, best_act_fn, best_act_name, lr = best_config
model = MLP(input_dim=X.shape[1], hidden1_neurons=neurons, hidden1_activation=best_act_fn, num_classes=len(class_names))
optimizer = optim.Adam(model.parameters(), lr=lr)

# Split training+validation further into train and validation sets for final training
X_tr, X_val, y_tr, y_val = train_test_split(X_trainval, y_trainval, test_size=0.2, stratify=y_trainval, random_state=42)

# Train final model with early stopping, tracking accuracy for visualization
_, train_acc_hist, val_acc_hist = train_with_early_stopping(model, optimizer, criterion, X_tr, y_tr, X_val, y_val)

# --- Plot training and validation accuracy per epoch ---
# Visualizing training dynamics helps understand convergence behavior and potential overfitting.
plt.figure(figsize=(8,5))
plt.plot(train_acc_hist, label="Training Accuracy")
plt.plot(val_acc_hist, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy per Epoch")
plt.legend()
plt.grid(True)
plt.show()

# --- TEST SET EVALUATION ---
# Final model evaluation on unseen test data to measure generalization.

model.eval()
with torch.no_grad():
    xb_test = torch.tensor(X_test, dtype=torch.float32)
    yb_test = torch.tensor(y_test, dtype=torch.long)
    outputs_test = model(xb_test)
    preds_test = outputs_test.argmax(dim=1).numpy()

# Print detailed classification report:
# Precision, recall, f1-score are more informative than accuracy alone,
# especially in class imbalanced situations.
print("\nClassification Report on Test Set:")
print(classification_report(y_test, preds_test, target_names=class_names))

# Plot confusion matrix heatmap:
# Shows detailed patterns of correct and incorrect classifications between classes.
cm = confusion_matrix(y_test, preds_test)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.title("Confusion Matrix on Test Set")
plt.show()
