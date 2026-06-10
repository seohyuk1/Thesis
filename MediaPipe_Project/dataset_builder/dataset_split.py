import numpy as np
from sklearn.model_selection import train_test_split
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

X_PATH = os.path.join(BASE_DIR, "data", "dataset", "xgb", "X.npy")
Y_PATH = os.path.join(BASE_DIR, "data", "dataset", "xgb", "y.npy")

SAVE_DIR = os.path.join(BASE_DIR, "data", "dataset", "split")
os.makedirs(SAVE_DIR, exist_ok=True)

# =========================
# LOAD
# =========================
X = np.load(X_PATH)
y = np.load(Y_PATH)

print("전체 데이터:", X.shape)

# =========================
# SPLIT (70/15/15)
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
# SAVE
# =========================
np.save(os.path.join(SAVE_DIR, "X_train.npy"), X_train)
np.save(os.path.join(SAVE_DIR, "y_train.npy"), y_train)

np.save(os.path.join(SAVE_DIR, "X_val.npy"), X_val)
np.save(os.path.join(SAVE_DIR, "y_val.npy"), y_val)

np.save(os.path.join(SAVE_DIR, "X_test.npy"), X_test)
np.save(os.path.join(SAVE_DIR, "y_test.npy"), y_test)

print("\n===== SPLIT 완료 =====")
print("Train:", X_train.shape)
print("Val:", X_val.shape)
print("Test:", X_test.shape)