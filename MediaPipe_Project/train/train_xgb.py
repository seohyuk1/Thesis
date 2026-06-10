import os
import sys
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

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
DATA_DIR = os.path.join(
    BASE_DIR,
    "data",
    "dataset",
    "split"
)

X_train = np.load(
    os.path.join(DATA_DIR, "X_train.npy")
)
y_train = np.load(
    os.path.join(DATA_DIR, "y_train.npy")
)

X_val = np.load(
    os.path.join(DATA_DIR, "X_val.npy")
)
y_val = np.load(
    os.path.join(DATA_DIR, "y_val.npy")
)

X_test = np.load(
    os.path.join(DATA_DIR, "X_test.npy")
)
y_test = np.load(
    os.path.join(DATA_DIR, "y_test.npy")
)

LABELS = [
    "side_lunge",
    "burpee",
    "plank",
    "pushup",
    "crunch"
]

# =========================
# RESULT DIR
# =========================
RESULT_DIR = os.path.join(
    BASE_DIR,
    "results"
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)

# =========================
# MODEL
# =========================
model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softprob",
    num_class=len(LABELS),
    eval_metric="mlogloss",
    random_state=42
)

print("\nTraining XGBoost...")

model.fit(
    X_train,
    y_train,
    eval_set=[
        (X_val, y_val)
    ],
    verbose=False
)

print("Training Complete")

# =========================
# TEST
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

print("Accuracy:", round(acc, 4))
print("Macro-F1:", round(macro_f1, 4))
print("Weighted-F1:", round(weighted_f1, 4))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=LABELS
    )
)

# =========================
# CONFUSION MATRIX
# =========================
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
    "xgb",
    acc,
    macro_f1,
    weighted_f1
)

# =========================
# SAVE RESULT CSV
# =========================
result_file = os.path.join(
    RESULT_DIR,
    "xgb_results.csv"
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
    "\nXGB results saved:",
    result_file
)

# =========================
# SAVE CONFUSION MATRIX
# =========================
cm_file = os.path.join(
    RESULT_DIR,
    "xgb_confusion_matrix.csv"
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
    "xgb_log.csv"
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
MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "saved",
    "xgb_model.pkl"
)

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