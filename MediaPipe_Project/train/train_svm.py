import os
import sys
import numpy as np
import pandas as pd
import joblib

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# =========================
# PROJECT ROOT
# =========================
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from utils.metrics_logger import save_summary


# =========================
# PATH
# =========================
X_PATH = os.path.join(
    BASE_DIR,
    "data",
    "dataset",
    "xgb",
    "X.npy"
)

Y_PATH = os.path.join(
    BASE_DIR,
    "data",
    "dataset",
    "xgb",
    "y.npy"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "saved",
    "svm_model.pkl"
)

RESULT_DIR = os.path.join(
    BASE_DIR,
    "results"
)

os.makedirs(RESULT_DIR, exist_ok=True)

LABELS = [
    "side_lunge",
    "burpee",
    "plank",
    "pushup",
    "crunch"
]


# =========================
# LOAD DATA
# =========================
X = np.load(X_PATH)
y = np.load(Y_PATH)

print("X shape:", X.shape)
print("y shape:", y.shape)


# =========================
# SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =========================
# MODEL
# =========================
model = SVC(
    kernel="rbf",
    C=10,
    gamma="scale",
    probability=True,
    class_weight="balanced"
)


# =========================
# TRAIN
# =========================
print("\nTraining SVM...")

model.fit(
    X_train,
    y_train
)

print("Training Complete")


# =========================
# EVALUATION
# =========================
y_pred = model.predict(X_test)

acc = accuracy_score(
    y_test,
    y_pred
)

macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro"
)

weighted_f1 = f1_score(
    y_test,
    y_pred,
    average="weighted"
)

print("\n===== FINAL RESULTS =====")

print("Accuracy:", acc)
print("Macro-F1:", macro_f1)
print("Weighted-F1:", weighted_f1)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=LABELS
    )
)

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)


# =========================
# SAVE SUMMARY
# =========================
save_summary(
    BASE_DIR,
    "svm",
    acc,
    macro_f1,
    weighted_f1
)


# =========================
# SAVE RESULT CSV
# =========================
result_file = os.path.join(
    RESULT_DIR,
    "svm_results.csv"
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
    "\nSVM results saved:",
    result_file
)


# =========================
# SAVE CONFUSION MATRIX
# =========================
cm_file = os.path.join(
    RESULT_DIR,
    "svm_confusion_matrix.csv"
)

pd.DataFrame(cm).to_csv(
    cm_file,
    index=False
)

print(
    "Confusion Matrix saved:",
    cm_file
)


# =========================
# SAVE TRAINING LOG
# =========================
LOG_DIR = os.path.join(
    RESULT_DIR,
    "training_logs"
)

os.makedirs(
    LOG_DIR,
    exist_ok=True
)

log_file = os.path.join(
    LOG_DIR,
    "svm_log.csv"
)

pd.DataFrame({
    "accuracy": [acc],
    "macro_f1": [macro_f1],
    "weighted_f1": [weighted_f1]
}).to_csv(
    log_file,
    index=False
)

print(
    "Training Log saved:",
    log_file
)


# =========================
# SAVE MODEL
# =========================
os.makedirs(
    os.path.dirname(MODEL_PATH),
    exist_ok=True
)

joblib.dump(
    model,
    MODEL_PATH
)

print(
    "\nModel saved:",
    MODEL_PATH
)