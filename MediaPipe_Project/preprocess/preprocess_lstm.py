import numpy as np
from preprocess.utils import *
from sklearn.model_selection import train_test_split

def create_sequences(data, labels, window_size=30):
    X, y = [], []

    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(labels[i+window_size])

    return np.array(X), np.array(y)


def preprocess_lstm(data_path, window_size=30):
    df = load_csv_folder(data_path)

    df = df.dropna()

    X_all, y_all = [], []

    # 운동별 분리 (필수)
    for mode in df['mode'].unique():
        df_mode = df[df['mode'] == mode]

        # (T, V, C)
        data = build_landmark_array(df_mode)

        # 👉 LSTM용 reshape (T, V*C)
        T, V, C = data.shape
        data = data.reshape(T, V * C)

        labels = extract_labels(df_mode)

        X_seq, y_seq = create_sequences(data, labels, window_size)

        X_all.append(X_seq)
        y_all.append(y_seq)

    X_final = np.concatenate(X_all)
    y_final = np.concatenate(y_all)

    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y_final,
        test_size=0.2,
        random_state=42,
        stratify=y_final
    )

    return X_train, X_test, y_train, y_test