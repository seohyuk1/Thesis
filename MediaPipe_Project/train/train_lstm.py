import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =========================
# CONFIG
# =========================
X_PATH = "data/dataset/lstm/X.npy"
Y_PATH = "data/dataset/lstm/y.npy"

MODEL_SAVE_PATH = os.path.join("models", "saved", "lstm_model.keras")

LABEL_NAMES = ["squat", "pushup", "pullup"]


# =========================
# LOAD DATA
# =========================
def load_data():
    X = np.load(X_PATH)
    y = np.load(Y_PATH)

    print("X shape:", X.shape)   # (N, T, F)
    print("y shape:", y.shape)

    return X, y


# =========================
# MODEL
# =========================
def build_model(input_shape, num_classes):
    model = keras.Sequential()

    model.add(keras.layers.LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.LSTM(64))
    model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.Dense(64, activation="relu"))
    model.add(keras.layers.Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# =========================
# TRAIN
# =========================
def train_model(X, y):
    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    num_classes = len(LABEL_NAMES)

    model = build_model(X.shape[1:], num_classes)

    model.summary()

    # 학습
    model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=16,
        validation_data=(X_test, y_test)
    )

    # 예측
    y_pred = np.argmax(model.predict(X_test), axis=1)

    # 평가
    acc = accuracy_score(y_test, y_pred)

    print("\n======================")
    print("Accuracy:", acc)

    # ✔ 실제 존재하는 클래스만 출력
    all_labels = np.unique(y)
    target_names = [LABEL_NAMES[i] for i in all_labels]

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        labels=all_labels,
        target_names=target_names,
        zero_division=0
    ))

    print("\nConfusion Matrix:")
    print(confusion_matrix(
        y_test,
        y_pred,
        labels=all_labels
    ))

    return model


# =========================
# SAVE MODEL
# =========================
def save_model(model):
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    model.save(MODEL_SAVE_PATH)
    print(f"\nModel saved to {MODEL_SAVE_PATH}")


# =========================
# MAIN
# =========================
def main():
    X, y = load_data()

    model = train_model(X, y)

    save_model(model)


if __name__ == "__main__":
    main()