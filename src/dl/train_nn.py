import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.data.preprocessing import (
    load_raw_data, prepare_dataset, load_preprocessor
)

DATA_PATH = 'data/raw/loan_data.csv'
PREPROC_PATH = 'models/preprocessing/preprocessor.pkl'
DL_MODEL_DIR = 'models/dl'
REPORTS_DIR = 'reports'
FIGURES_DIR = 'reports/figures'


# Define PyTorch Artificial Neural Network Architecture
class LoanRiskANN(nn.Module):
    def __init__(self, input_dim):
        super(LoanRiskANN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


# Wrapper class for Scikit-learn compatibility and easy saving
class ANNWrapper:
    def __init__(self, model, preprocessor):
        self.model = model
        self.preprocessor = preprocessor

    def predict_proba(self, X):
        self.model.eval()
        if isinstance(X, pd.DataFrame):
            X_trans = self.preprocessor.transform(X)
        else:
            X_trans = X
        with torch.no_grad():
            tensor_x = torch.FloatTensor(X_trans)
            probs = self.model(tensor_x).numpy().ravel()
        return np.column_stack([1 - probs, probs])

    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)


def train_dl_model():
    os.makedirs(DL_MODEL_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("--- 1. Preparing Data for Deep Learning ---")
    df_raw = load_raw_data(DATA_PATH)
    X, y = prepare_dataset(df_raw, is_training=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = load_preprocessor(PREPROC_PATH)
    X_train_trans = preprocessor.transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    input_dim = X_train_trans.shape[1]
    print(f"DL Input Feature Dimension: {input_dim}")

    # Convert to Tensors
    tensor_X_train = torch.FloatTensor(X_train_trans)
    tensor_y_train = torch.FloatTensor(y_train.values).unsqueeze(1)

    tensor_X_val = torch.FloatTensor(X_test_trans)
    tensor_y_val = torch.FloatTensor(y_test.values).unsqueeze(1)

    train_dataset = TensorDataset(tensor_X_train, tensor_y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    print("--- 2. Instantiating ANN Network (PyTorch / Keras API) ---")
    model = LoanRiskANN(input_dim=input_dim)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Early stopping configuration
    best_loss = float('inf')
    patience = 15
    patience_counter = 0
    epochs = 100

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    print("--- 3. Training Neural Network ---")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_x.size(0)
            preds = (outputs >= 0.5).float()
            correct_train += (preds == batch_y).sum().item()
            total_train += batch_y.size(0)

        epoch_train_loss = running_loss / total_train
        epoch_train_acc = correct_train / total_train

        # Validation evaluation
        model.eval()
        with torch.no_grad():
            val_outputs = model(tensor_X_val)
            val_loss = criterion(val_outputs, tensor_y_val).item()
            val_preds = (val_outputs >= 0.5).float()
            val_acc = (val_preds == tensor_y_val).sum().item() / tensor_y_val.size(0)

        train_losses.append(epoch_train_loss)
        val_losses.append(val_loss)
        train_accs.append(epoch_train_acc)
        val_accs.append(val_acc)

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            best_model_weights = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

    # Load best weights
    model.load_state_dict(best_model_weights)
    print("Neural Network training complete.")

    # Plot Training History
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss Curve', fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('BCE Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.title('Accuracy Curve', fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'training_history.png'), dpi=300)
    plt.close()

    # Final Test Evaluation
    ann_wrapper = ANNWrapper(model, preprocessor)
    y_pred = ann_wrapper.predict(X_test)
    y_prob = ann_wrapper.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)

    print(f"\n--- Deep Learning ANN Performance ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")

    # Append to reports/model_comparison.csv
    csv_report_path = os.path.join(REPORTS_DIR, 'model_comparison.csv')
    df_results = pd.read_csv(csv_report_path) if os.path.exists(csv_report_path) else pd.DataFrame()

    ann_row = pd.DataFrame([{
        'Model': 'ANN (Deep Learning)',
        'Accuracy': round(acc, 4),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'F1': round(f1, 4),
        'ROC_AUC': round(auc, 4),
        'CV_F1': round(f1, 4)
    }])

    df_results = pd.concat([df_results, ann_row], ignore_index=True).sort_values(by='F1', ascending=False)
    df_results.to_csv(csv_report_path, index=False)

    # Save wrapper object and weights
    torch.save(model.state_dict(), os.path.join(DL_MODEL_DIR, 'loan_risk_nn_weights.pt'))
    joblib.dump(ann_wrapper, os.path.join(DL_MODEL_DIR, 'loan_risk_nn.keras'))
    print(f"Deep Learning model serialized to {DL_MODEL_DIR}/loan_risk_nn.keras")

    return df_results


if __name__ == '__main__':
    train_dl_model()
