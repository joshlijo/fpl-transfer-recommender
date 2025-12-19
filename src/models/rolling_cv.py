"""
Phase 3A — Rolling time-based cross-validation for ranking models.
"""

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr

from src.pipeline.build_training_dataset import build_training_dataset
from src.config.feature_masks import RANK_FEATURE_MASKS

START_GW = 6
END_GW = 16

POSITIONS = ["Goalkeeper", "Defender", "Midfielder", "Forward"]

GBM_PARAMS = dict(
    max_depth=5,
    learning_rate=0.05,
    max_iter=300,
    random_state=42,
)

def evaluate(y_true, y_pred):
    return {
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "spearman": spearmanr(y_true, y_pred).correlation,
    }

def run_rolling_cv(df: pd.DataFrame, position: str):
    features = RANK_FEATURE_MASKS[position]
    pos_df = df[df["position"] == position].copy()

    fold_metrics = []

    for val_gw in range(START_GW + 5, END_GW + 1):
        train_df = pos_df[pos_df["target_gw"] < val_gw]
        val_df = pos_df[pos_df["target_gw"] == val_gw]

        if train_df.empty or val_df.empty:
            continue

        X_train = train_df[features]
        y_train = train_df["target_points"]

        X_val = val_df[features]
        y_val = val_df["target_points"]

        model = HistGradientBoostingRegressor(**GBM_PARAMS)
        model.fit(X_train, y_train)

        preds = model.predict(X_val)
        metrics = evaluate(y_val, preds)
        metrics["val_gw"] = val_gw

        fold_metrics.append(metrics)

    return pd.DataFrame(fold_metrics)

if __name__ == "__main__":
    print("\n=== PHASE 3A — ROLLING CV (RANKING) ===\n")

    df = build_training_dataset(start_gw=START_GW, end_gw=END_GW)

    for position in POSITIONS:
        print(f"\n--- {position.upper()} ---")

        cv_df = run_rolling_cv(df, position)

        if cv_df.empty:
            print("No folds evaluated.")
            continue

        print(cv_df.round(3).to_string(index=False))

        print("\nSUMMARY")
        print(
            cv_df[["rmse", "mae", "spearman"]]
            .agg(["mean", "std"])
            .round(3)
        )

    print("\n=== DONE ===\n")