import os
import numpy as np
import joblib

# =========================
# CONFIG
# =========================
MODEL_PATH = os.path.join("models", "saved", "xgb_model.pkl")


# =========================
# XGB MODEL CLASS
# =========================
class XGBModel:
    def __init__(self, model_path=MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.load_model()

    # ---------------------
    # 모델 로드
    # ---------------------
    def load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"모델 파일 없음: {self.model_path}")

        self.model = joblib.load(self.model_path)
        print(f"[XGB] 모델 로드 완료: {self.model_path}")

    # ---------------------
    # 예측
    # ---------------------
    def predict(self, X):
        X = np.array(X)

        # 1D 입력이면 2D로 변환
        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        return self.model.predict(X)

    # ---------------------
    # 확률 예측
    # ---------------------
    def predict_proba(self, X):
        X = np.array(X)

        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        return self.model.predict_proba(X)

    # ---------------------
    # 클래스 정보
    # ---------------------
    def get_classes(self):
        return self.model.classes_