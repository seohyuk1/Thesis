import numpy as np
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# =========================
# PATH
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

X_PATH = os.path.join(BASE_DIR, "data", "dataset", "xgb", "X.npy")
Y_PATH = os.path.join(BASE_DIR, "data", "dataset", "xgb", "y.npy")

MODEL_PATH = os.path.join(BASE_DIR, "models", "saved", "xgb_model.pkl")

LABELS = ["squat", "pushup", "pullup"]

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
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softmax",
    num_class=len(LABELS),
    eval_metric="mlogloss"
)

print("\nTraining...")

model.fit(X_train, y_train)

# =========================
# EVAL
# =========================
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))

# =========================
# SAVE
# =========================
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

joblib.dump(model, MODEL_PATH)

print("Model saved:", MODEL_PATH)