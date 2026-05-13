import numpy as np

# =========================
# LABEL
# =========================
LABEL_MAP = {
    0: "squat",
    1: "pushup",
    2: "pullup"
}

# =========================
# SOFT VOTING
# =========================
def soft_voting(
    xgb_prob,
    lstm_prob,
    stgcn_prob
):

    """
    각 모델의 확률 평균

    input:
        xgb_prob   -> (3,)
        lstm_prob  -> (3,)
        stgcn_prob -> (3,)

    output:
        final_class
        final_confidence
        final_probs
    """

    probs = np.array([
        xgb_prob,
        lstm_prob,
        stgcn_prob
    ])

    # 평균 확률
    final_probs = np.mean(
        probs,
        axis=0
    )

    final_class = np.argmax(
        final_probs
    )

    final_confidence = float(
        np.max(final_probs)
    )

    return (
        final_class,
        final_confidence,
        final_probs
    )

# =========================
# WEIGHTED VOTING
# =========================
def weighted_voting(
    xgb_prob,
    lstm_prob,
    stgcn_prob,
    weights=(0.2, 0.3, 0.5)
):

    """
    가중치 기반 앙상블

    기본:
        XGB    = 0.2
        LSTM   = 0.3
        ST-GCN = 0.5
    """

    probs = np.array([
        xgb_prob,
        lstm_prob,
        stgcn_prob
    ])

    weights = np.array(
        weights,
        dtype=np.float32
    )

    # 가중 평균
    final_probs = np.average(
        probs,
        axis=0,
        weights=weights
    )

    final_class = np.argmax(
        final_probs
    )

    final_confidence = float(
        np.max(final_probs)
    )

    return (
        final_class,
        final_confidence,
        final_probs
    )

# =========================
# HARD VOTING
# =========================
def hard_voting(
    xgb_class,
    lstm_class,
    stgcn_class
):

    """
    다수결 기반 앙상블
    """

    votes = [
        xgb_class,
        lstm_class,
        stgcn_class
    ]

    final_class = max(
        set(votes),
        key=votes.count
    )

    confidence = (
        votes.count(final_class) / 3.0
    )

    return (
        final_class,
        confidence
    )

# =========================
# MAIN ENSEMBLE
# =========================
def ensemble_predict(
    xgb_prob,
    lstm_prob,
    stgcn_prob,
    method="weighted"
):

    """
    method:
        soft
        weighted
        hard
    """

    # =========================
    # SOFT
    # =========================
    if method == "soft":

        final_class, confidence, probs = soft_voting(
            xgb_prob,
            lstm_prob,
            stgcn_prob
        )

    # =========================
    # WEIGHTED
    # =========================
    elif method == "weighted":

        final_class, confidence, probs = weighted_voting(
            xgb_prob,
            lstm_prob,
            stgcn_prob
        )

    # =========================
    # HARD
    # =========================
    elif method == "hard":

        xgb_class = np.argmax(xgb_prob)
        lstm_class = np.argmax(lstm_prob)
        stgcn_class = np.argmax(stgcn_prob)

        final_class, confidence = hard_voting(
            xgb_class,
            lstm_class,
            stgcn_class
        )

        probs = None

    else:
        raise ValueError(
            "지원하지 않는 ensemble method"
        )

    action_name = LABEL_MAP.get(
        final_class,
        "unknown"
    )

    return {
        "class_id": int(final_class),
        "action": action_name,
        "confidence": round(confidence, 4),
        "probabilities": probs
    }

# =========================
# TEST
# =========================
if __name__ == "__main__":

    # 예시 확률
    xgb_prob = np.array([
        0.7,
        0.2,
        0.1
    ])

    lstm_prob = np.array([
        0.6,
        0.3,
        0.1
    ])

    stgcn_prob = np.array([
        0.8,
        0.1,
        0.1
    ])

    result = ensemble_predict(
        xgb_prob,
        lstm_prob,
        stgcn_prob,
        method="weighted"
    )

    print("\n===== Ensemble Result =====")

    print(result)