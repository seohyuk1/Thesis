import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

# 프로젝트 루트 등록
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from utils.metrics_logger import save_summary


# =========================
# PATH
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

X_PATH = os.path.join(BASE_DIR, "data", "dataset", "transformer", "X.npy")
Y_PATH = os.path.join(BASE_DIR, "data", "dataset", "transformer", "y.npy")

# =========================
# RESULT PATH
# =========================
RESULT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULT_DIR, exist_ok=True)

X = np.load(X_PATH)
y = np.load(Y_PATH)

print("X shape:", X.shape)
print("y shape:", y.shape)


# =========================
# SPLIT (train / val / test)
# =========================
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp
)


# =========================
# TENSOR
# =========================
X_train = torch.tensor(X_train, dtype=torch.float32)
X_val   = torch.tensor(X_val, dtype=torch.float32)
X_test  = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.long)
y_val   = torch.tensor(y_val, dtype=torch.long)
y_test  = torch.tensor(y_test, dtype=torch.long)


# =========================
# POSITIONAL ENCODING
# =========================
class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=34):
        super().__init__()

        pe = torch.zeros(max_len, dim)

        position = torch.arange(0, max_len).unsqueeze(1).float()

        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-np.log(10000.0) / dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].to(x.device)


# =========================
# MODEL (UPGRADED TRANSFORMER)
# =========================
class TransformerModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.input_proj = nn.Linear(3, 64)

        self.pos_encoder = PositionalEncoding(64)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=8,
            dim_feedforward=128,
            dropout=0.2,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=3
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, 64))

        self.fc = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 5)
        )

    def forward(self, x):

        B = x.size(0)

        x = self.input_proj(x)  # (B, 33, 64)

        cls = self.cls_token.repeat(B, 1, 1)

        x = torch.cat([cls, x], dim=1)  # (B, 34, 64)

        x = self.pos_encoder(x)

        x = self.transformer(x)

        x = x[:, 0]  # CLS token

        return self.fc(x)


# =========================
# SETUP
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TransformerModel().to(device)

optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

# =========================
# TRAINING LOG
# =========================
training_logs = []

LOG_DIR = os.path.join(
    RESULT_DIR,
    "training_logs"
)

os.makedirs(
    LOG_DIR,
    exist_ok=True
)

LOG_PATH = os.path.join(
    LOG_DIR,
    "transformer_log.csv"
)

# =========================
# EVAL FUNCTION
# =========================
def evaluate(X_data, y_data):

    model.eval()

    with torch.no_grad():

        pred = model(X_data.to(device)).argmax(dim=1)

        acc = accuracy_score(y_data, pred.cpu())
        macro_f1 = f1_score(y_data, pred.cpu(), average="macro")
        weighted_f1 = f1_score(y_data, pred.cpu(), average="weighted")

    return acc, macro_f1, weighted_f1


# =========================
# TRAIN LOOP
# =========================
best_val = 0

for epoch in range(1, 51):

    model.train()

    optimizer.zero_grad()

    out = model(X_train.to(device))

    loss = criterion(out, y_train.to(device))

    loss.backward()
    optimizer.step()

    val_acc, val_f1, val_wf1 = evaluate(X_val, y_val)
    test_acc, test_f1, test_wf1 = evaluate(X_test, y_test)

    # best model 저장 (val 기준)
    if val_f1 > best_val:
        best_val = val_f1
        torch.save(
            model.state_dict(),
            os.path.join(BASE_DIR, "models", "saved", "best_transformer.pt")
        )

    print(
        f"Epoch {epoch:03d} | "
        f"Loss {loss.item():.4f} | "
        f"Val-F1 {val_f1:.4f} | "
        f"Test-F1 {test_f1:.4f}"
    )

    training_logs.append({
    "epoch": epoch,
    "loss": float(loss.item()),
    "val_accuracy": float(val_acc),
    "val_macro_f1": float(val_f1),
    "val_weighted_f1": float(val_wf1),
    "test_accuracy": float(test_acc),
    "test_macro_f1": float(test_f1),
    "test_weighted_f1": float(test_wf1)
})


# =========================
# FINAL TEST
# =========================
acc, macro_f1, weighted_f1 = evaluate(X_test, y_test)

print("\n===== FINAL RESULTS =====")
print("Accuracy:", acc)
print("Macro-F1:", macro_f1)
print("Weighted-F1:", weighted_f1)

# =========================
# CREATE RESULTS DIR
# =========================
RESULT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULT_DIR, exist_ok=True)

# =========================
# SAVE SUMMARY
# =========================
save_summary(
    BASE_DIR,
    "transformer",
    acc,
    macro_f1,
    weighted_f1
)

# =========================
# SAVE TRANSFORMER RESULT
# =========================
import pandas as pd

result_file = os.path.join(
    RESULT_DIR,
    "transformer_results.csv"
)

pd.DataFrame({
    "metric": [
        "accuracy",
        "macro_f1",
        "weighted_f1"
    ],
    "score": [
        acc,
        macro_f1,
        weighted_f1
    ]
}).to_csv(
    result_file,
    index=False
)

print(
    "\nTransformer results saved:",
    result_file
)

# =========================
# CONFUSION MATRIX
# =========================
from sklearn.metrics import confusion_matrix

model.eval()

with torch.no_grad():

    pred = model(
        X_test.to(device)
    ).argmax(dim=1)

cm = confusion_matrix(
    y_test.cpu().numpy(),
    pred.cpu().numpy()
)

cm_path = os.path.join(
    RESULT_DIR,
    "transformer_confusion_matrix.csv"
)

pd.DataFrame(cm).to_csv(
    cm_path,
    index=False
)

print(
    "Confusion Matrix saved:",
    cm_path
)

# =========================
# SAVE TRAINING LOG
# =========================
log_df = pd.DataFrame(
    training_logs
)

log_df.to_csv(
    LOG_PATH,
    index=False
)

print(
    "Training Log saved:",
    LOG_PATH
)

# =========================
# SAVE MODEL
# =========================
MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "saved",
    "transformer_model.pt"
)

torch.save(
    model.state_dict(),
    MODEL_PATH
)

print("\nTransformer saved:", MODEL_PATH)