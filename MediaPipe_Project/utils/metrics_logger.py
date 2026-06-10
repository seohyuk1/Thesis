import os
import pandas as pd

def save_summary(base_dir, model_name, acc, macro_f1, weighted_f1):

    result_dir = os.path.join(
        base_dir,
        "results"
    )

    os.makedirs(
        result_dir,
        exist_ok=True
    )

    result_path = os.path.join(
        result_dir,
        "summary.csv"
    )

    new_row = pd.DataFrame([{
        "model": model_name,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1
    }])

    if (
        os.path.exists(result_path)
        and os.path.getsize(result_path) > 0
    ):

        old = pd.read_csv(result_path)

        old = old[
            old["model"] != model_name
        ]

        df = pd.concat(
            [old, new_row],
            ignore_index=True
        )

    else:
        df = new_row

    df.to_csv(
        result_path,
        index=False
    )

    print(
        f"\nSummary saved: {result_path}"
    )