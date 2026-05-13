import numpy as np
from preprocess.utils import *
from sklearn.model_selection import train_test_split


def create_stgcn_sequences(data, labels, window_size=30):
    X, y = [], []

    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(labels[i+window_size])

    return np.array(X), np.array(y)


def preprocess_stgcn(data_path, window_size=30):
    df = load_csv_folder(data_path)

    # 결측치 제거
    df = df.dropna()

    X_all, y_all = [], []

    # 운동별 분리 (핵심)
    for mode in df['mode'].unique():
        df_mode = df[df['mode'] == mode]

        # landmark → (T, V, C)
        data = build_landmark_array(df_mode)

        # 라벨
        labels = extract_labels(df_mode)

        # sequence 생성
        X_seq, y_seq = create_stgcn_sequences(data, labels, window_size)

        X_all.append(X_seq)
        y_all.append(y_seq)

    # 전체 합치기
    X_final = np.concatenate(X_all)
    y_final = np.concatenate(y_all)

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_final,
        y_final,
        test_size=0.2,
        random_state=42,
        stratify=y_final
    )

    return X_train, X_test, y_train, y_test