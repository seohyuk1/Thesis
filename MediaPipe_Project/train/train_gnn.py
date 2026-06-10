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
import torch.nn.functional as F
import pandas as pd

from utils.metrics_logger import save_summary

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


# =========================
# SEED
# =========================
torch.manual_seed(42)
np.random.seed(42)

# =========================
# PATH
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

X_PATH = os.path.join(BASE_DIR, "data", "dataset", "gnn", "X.npy")
Y_PATH = os.path.join(BASE_DIR, "data", "dataset", "gnn", "y.npy")


# =========================
# LOAD DATA
# =========================
X = np.load(X_PATH)   # (N, 33, 3)
y = np.load(Y_PATH)

print("X shape:", X.shape)
print("y shape:", y.shape)

unique, counts = np.unique(y, return_counts=True)

print("Class Distribution")
for u, c in zip(unique, counts):
    print(f"Class {u}: {c}")


# =========================
# EDGE INDEX (MediaPipe Pose)
# =========================
EDGE_INDEX = torch.tensor([
    [11, 13], [13, 15],
    [12, 14], [14, 16],
    [11, 12],
    [11, 23], [12, 24],
    [23, 24],
    [23, 25], [25, 27],
    [24, 26], [26, 28],
], dtype=torch.long).t().contiguous()


# =========================
# BUILD GRAPH DATASET
# =========================
data_list = []

for i in range(len(X)):
    x = torch.tensor(X[i], dtype=torch.float)
    label = torch.tensor(y[i], dtype=torch.long)

    data_list.append(
        Data(x=x, edge_index=EDGE_INDEX, y=label)
    )


# =========================
# SPLIT (train / val / test)
# =========================
train_data, temp_data, y_train, y_temp = train_test_split(
    data_list, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

val_data, test_data, _, _ = train_test_split(
    temp_data, y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp
)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)


# =========================
# MODEL
# =========================
class GCN(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = GCNConv(3, 64)
        self.conv2 = GCNConv(64, 128)
        self.fc = torch.nn.Linear(128, 5)

    def forward(self, data):

        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        x = global_mean_pool(x, batch)
        x = self.fc(x)

        return x


# =========================
# SETUP
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GCN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

# =========================
# TRAINING LOG
# =========================
training_logs = []

# =========================
# TRAIN
# =========================
def train():

    model.train()
    total_loss = 0

    for data in train_loader:

        data = data.to(device)

        optimizer.zero_grad()
        out = model(data)

        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


# =========================
# EVALUATION
# =========================
def evaluate(loader):

    model.eval()

    y_true, y_pred = [], []

    with torch.no_grad():
        for data in loader:

            data = data.to(device)
            out = model(data)

            pred = out.argmax(dim=1)

            y_true.extend(data.y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")

    return acc, macro_f1, weighted_f1, y_true, y_pred


# =========================
# TRAIN LOOP
# =========================
best_val = 0

for epoch in range(1, 51):

    loss = train()

    val_acc, val_f1, val_wf1, _, _ = evaluate(val_loader)
    test_acc, test_f1, test_wf1, _, _ = evaluate(test_loader)

    # best model save (val 기준)
    if val_f1 > best_val:
        best_val = val_f1
        torch.save(
            model.state_dict(),
            os.path.join(BASE_DIR, "models", "saved", "best_gnn.pt")
        )

    print(
        f"Epoch {epoch:03d} | "
        f"Loss {loss:.4f} | "
        f"Val-F1 {val_f1:.4f} | "
        f"Test-F1 {test_f1:.4f}"
    )

    training_logs.append({
    "epoch": epoch,
    "loss": loss,
    "val_accuracy": val_acc,
    "val_macro_f1": val_f1,
    "val_weighted_f1": val_wf1,
    "test_accuracy": test_acc,
    "test_macro_f1": test_f1,
    "test_weighted_f1": test_wf1
})


# =========================
# FINAL TEST
# =========================
acc, macro_f1, weighted_f1, y_true, y_pred = evaluate(test_loader)

cm = confusion_matrix(y_true, y_pred)
cm_path = os.path.join(
    BASE_DIR,
    "results",
    "gnn_confusion_matrix.csv"
)

pd.DataFrame(cm).to_csv(
    cm_path,
    index=False
)

print(
    "Confusion Matrix saved:",
    cm_path
)

print("\n===== FINAL RESULTS =====")
print("Accuracy:", acc)
print("Macro-F1:", macro_f1)
print("Weighted-F1:", weighted_f1)
print("Confusion Matrix:\n", cm)

log_dir = os.path.join(
    BASE_DIR,
    "results",
    "training_logs"
)

os.makedirs(log_dir, exist_ok=True)

pd.DataFrame(training_logs).to_csv(
    os.path.join(
        log_dir,
        "gnn_log.csv"
    ),
    index=False
)

print(
    "\nTraining log saved:",
    os.path.join(log_dir, "gnn_log.csv")
)

log_dir = os.path.join(
    BASE_DIR,
    "results",
    "training_logs"
)

os.makedirs(log_dir, exist_ok=True)

pd.DataFrame(training_logs).to_csv(
    os.path.join(
        log_dir,
        "gnn_log.csv"
    ),
    index=False
)

print(
    "\nTraining log saved:",
    os.path.join(log_dir, "gnn_log.csv")
)


# =========================
# SAVE SUMMARY
# =========================
save_summary(
    "gnn",
    acc,
    macro_f1,
    weighted_f1
)
# =========================
# SAVE GNN RESULT
# =========================
result_file = os.path.join(
    BASE_DIR,
    "results",
    "gnn_results.csv"
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
    "\nGNN results saved:",
    result_file
)

# =========================
# SAVE REPORT CSV
# =========================
report_path = os.path.join(BASE_DIR, "results", "gnn_report.csv")
os.makedirs(os.path.dirname(report_path), exist_ok=True)

pd.DataFrame({
    "metric": ["accuracy", "macro_f1", "weighted_f1"],
    "score": [acc, macro_f1, weighted_f1]
}).to_csv(report_path, index=False)

print("\nReport saved:", report_path)