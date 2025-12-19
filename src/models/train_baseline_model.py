"""
Baseline ML model for FPL point prediction.

Uses:
- Linear Regression
- Time-based train/validation split
- Explicit feature selection
- Proper evaluation (RMSE / MAE)

This establishes the first ML benchmark against heuristics.
"""

import sys
from pathlib import Path

# -------------------------------------------------
# Add project root to PYTHONPATH
# -------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

# -------------------------------------------------
# Imports
# -------------------------------------------------

from typing import List, Tuple
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

from src.pipeline.build_training_dataset import build_training_dataset


# -------------------------------------------------
# Feature configuration (LOCKED)
# -------------------------------------------------

NUMERIC_FEATURES: List[str] = [
    # Rolling form
    "ppg_last_1",
    "ppg_last_3",
    "ppg_last_5",
    "minutes_avg_last_5",
    "games_count_last_5",

    # Underlying stats
    "xg_avg_last_5",
    "xa_avg_last_5",
    "goals_avg_last_5",
    "assists_avg_last_5",
    "defcon_avg_last_5",
    "saves_avg_last_5",
    "goals_conceded_avg_last_5",

    # Fixture context
    "fixture_multiplier",
    "effective_elo_diff",
    "is_home",
]

TARGET_COL = "target_points"
TIME_COL = "target_gw"


# -------------------------------------------------
# Utilities
# -------------------------------------------------

def time_based_split(
    df: pd.DataFrame,
    train_gw_end: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train / validation by time.
    """
    train_df = df[df[TIME_COL] <= train_gw_end]
    val_df = df[df[TIME_COL] > train_gw_end]

    if train_df.empty or val_df.empty:
        raise RuntimeError("Invalid time split produced empty set")

    return train_df, val_df


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """
    Compute evaluation metrics.
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    return {
        "rmse": rmse,
        "mae": mean_absolute_error(y_true, y_pred),
    }

# -------------------------------------------------
# Main training routine
# -------------------------------------------------

def train_baseline_model(
    start_gw: int,
    end_gw: int,
    train_gw_end: int,
    season: str = "2025-2026",
):
    """
    Train and evaluate baseline Linear Regression model.
    """

    print("\n=== TRAINING BASELINE LINEAR REGRESSION ===\n")

    # -------------------------------------------------
    # 1. Build training dataset
    # -------------------------------------------------

    df = build_training_dataset(
        start_gw=start_gw,
        end_gw=end_gw,
        season=season,
    )

    print(f"Dataset shape: {df.shape}")
    print(f"Target mean: {df[TARGET_COL].mean():.3f}\n")

    # -------------------------------------------------
    # 2. Feature matrix
    # -------------------------------------------------

    X = df[NUMERIC_FEATURES].fillna(0.0)
    y = df[TARGET_COL].values

    # -------------------------------------------------
    # 3. Time-based split
    # -------------------------------------------------

    train_df, val_df = time_based_split(df, train_gw_end=train_gw_end)

    X_train = train_df[NUMERIC_FEATURES].fillna(0.0)
    y_train = train_df[TARGET_COL].values

    X_val = val_df[NUMERIC_FEATURES].fillna(0.0)
    y_val = val_df[TARGET_COL].values

    print(f"Train rows: {len(train_df)}")
    print(f"Val rows:   {len(val_df)}\n")

    # -------------------------------------------------
    # 4. Train model
    # -------------------------------------------------

    model = LinearRegression(n_jobs=-1)
    model.fit(X_train, y_train)

    # -------------------------------------------------
    # 5. Evaluate
    # -------------------------------------------------

    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)

    train_metrics = evaluate(y_train, train_preds)
    val_metrics = evaluate(y_val, val_preds)

    print("TRAIN METRICS")
    print(f"RMSE: {train_metrics['rmse']:.3f}")
    print(f"MAE:  {train_metrics['mae']:.3f}\n")

    print("VALIDATION METRICS")
    print(f"RMSE: {val_metrics['rmse']:.3f}")
    print(f"MAE:  {val_metrics['mae']:.3f}\n")

    # -------------------------------------------------
    # 6. Coefficient inspection (VERY IMPORTANT)
    # -------------------------------------------------

    coef_df = (
        pd.DataFrame(
            {
                "feature": NUMERIC_FEATURES,
                "coefficient": model.coef_,
            }
        )
        .sort_values("coefficient", ascending=False)
        .reset_index(drop=True)
    )

    print("TOP POSITIVE COEFFICIENTS")
    print(coef_df.head(10).to_string(index=False))

    print("\nTOP NEGATIVE COEFFICIENTS")
    print(coef_df.tail(10).to_string(index=False))

    print("\n=== DONE ===\n")

    return model, coef_df, val_metrics


# -------------------------------------------------
# CLI usage
# -------------------------------------------------

if __name__ == "__main__":
    # Example: GW6–16 dataset, validate on GW15–16
    train_baseline_model(
        start_gw=6,
        end_gw=16,
        train_gw_end=14,
    )
