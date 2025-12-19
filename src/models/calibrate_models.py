import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr

from src.pipeline.build_training_dataset import build_training_dataset
from src.config.feature_masks import RANK_FEATURE_MASKS

MODELS_DIR = Path("models")
CALIBRATION_GWS = list(range(11, 17))
POSITIONS = ["Goalkeeper", "Defender", "Midfielder", "Forward"]


def evaluate(y_true, y_pred):
    return {
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "spearman": spearmanr(y_true, y_pred).correlation,
    }


def calibrate_position(df: pd.DataFrame, position: str):
    print(f"\n=== CALIBRATING {position.upper()} ===")

    calib_df = df[
        (df["position"] == position)
        & (df["target_gw"].isin(CALIBRATION_GWS))
    ]

    if calib_df.empty:
        print("No data — skipping")
        return

    X = calib_df[RANK_FEATURE_MASKS[position]]
    y = calib_df["target_points"]

    model = joblib.load(MODELS_DIR / f"{position.lower()}_gbm.pkl")
    raw_pred = model.predict(X)

    lr = LinearRegression()
    lr.fit(raw_pred.reshape(-1, 1), y)

    calibrated = lr.predict(raw_pred.reshape(-1, 1))

    before = evaluate(y, raw_pred)
    after = evaluate(y, calibrated)

    print("BEFORE:", before)
    print("AFTER :", after)

    joblib.dump(lr, MODELS_DIR / f"{position.lower()}_calibrator.pkl")


if __name__ == "__main__":
    df = build_training_dataset(6, 16)

    for pos in POSITIONS:
        calibrate_position(df, pos)