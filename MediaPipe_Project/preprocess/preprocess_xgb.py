import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocess.utils import *
from sklearn.model_selection import train_test_split


def preprocess_xgb(data_path):

    # CSV 로드
    df = load_csv_folder(data_path)

    # 결측치 제거
    df = df.dropna()

    # =========================
    # landmark feature만 사용    # =========================
    X = extract_landmark_features(df)

    # 라벨
    y = extract_labels(df)

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test