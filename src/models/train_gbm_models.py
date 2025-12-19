"""
Train position-specific Gradient Boosting models (ranking-aware).
"""

import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr

from src.pipeline.build_training_dataset import build_training_dataset
from src.config.feature_masks import RANK_FEATURE_MASKS


# -------------------------------------------------
# Config
# -------------------------------------------------

POSITIONS = ["Goalkeeper", "Defender", "Midfielder", "Forward"]
TARGET = "target_points"

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


# -------------------------------------------------
# Evaluation
# -------------------------------------------------

def evaluate(y_true, y_pred):
    return {
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "spearman": spearmanr(y_true, y_pred).correlation,
    }


# -------------------------------------------------
# Training
# -------------------------------------------------

def train_position_model(df: pd.DataFrame, position: str):
    print(f"\n=== TRAINING {position.upper()} MODEL ===")

    pos_df = df[df["position"] == position].copy()
    features = RANK_FEATURE_MASKS[position]

    # Time-aware split
    train_df = pos_df[pos_df["target_gw"] <= 14]
    val_df = pos_df[pos_df["target_gw"] > 14]

    X_train, y_train = train_df[features], train_df[TARGET]
    X_val, y_val = val_df[features], val_df[TARGET]

    model = HistGradientBoostingRegressor(
        max_depth=5,
        learning_rate=0.05,
        max_iter=300,
        random_state=42,
    )

    model.fit(X_train, y_train)

    train_metrics = evaluate(y_train, model.predict(X_train))
    val_metrics = evaluate(y_val, model.predict(X_val))

    print("\nTRAIN METRICS")
    for k, v in train_metrics.items():
        print(f"{k.upper()}: {v:.3f}")

    print("\nVALIDATION METRICS")
    for k, v in val_metrics.items():
        print(f"{k.upper()}: {v:.3f}")

    model_path = MODELS_DIR / f"{position.lower()}_gbm.pkl"
    joblib.dump(model, model_path)

    print(f"\nSaved model → {model_path}")

    return val_metrics


# -------------------------------------------------
# Entrypoint
# -------------------------------------------------

if __name__ == "__main__":
    print("\n=== PHASE 3A — RANKING MODELS ===\n")

    df = build_training_dataset(start_gw=6, end_gw=16)

    for position in POSITIONS:
        train_position_model(df, position)

    print("\n=== ALL MODELS TRAINED ===\n")
