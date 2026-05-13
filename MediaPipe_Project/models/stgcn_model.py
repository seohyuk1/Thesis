import numpy as np
import os
from tensorflow import keras

# =========================
# CONFIG
# =========================
MODEL_PATH = os.path.join("models", "saved", "stgcn_model.keras")

LABEL_MAP = {
    0: "squat",
    1: "pushup",
    2: "pullup"
}


# =========================
# LOAD MODEL
# =========================
def load_model():
    try:
        model = keras.models.load_model(MODEL_PATH)
        print("ST-GCN 모델 로드 완료")
        return model
    except Exception as e:
        print("모델 로드 실패:", e)
        return None


# =========================
# PREPROCESS
# =========================
def preprocess_input(sequence):
    """
    sequence: (T, V, C)
    → (1, T, V, C)
    """
    sequence = np.array(sequence)

    if len(sequence.shape) != 3:
        print("입력 데이터 shape 오류:", sequence.shape)
        return None

    return np.expand_dims(sequence, axis=0)


# =========================
# PREDICT
# =========================
def predict(model, sequence):
    """
    sequence: (T, V, C)
    """
    input_data = preprocess_input(sequence)

    if input_data is None:
        return None

    preds = model.predict(input_data)
    class_idx = np.argmax(preds)

    return LABEL_MAP[class_idx]


# =========================
# TEST (옵션)
# =========================
if __name__ == "__main__":
    model = load_model()

    if model is not None:
        # 더미 테스트 데이터 (30프레임, 33관절, 3좌표)
        dummy = np.random.rand(30, 33, 3)

        result = predict(model, dummy)
        print("예측 결과:", result)