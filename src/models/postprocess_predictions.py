import pandas as pd
import numpy as np


def postprocess_predictions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "minutes_avg_last_5" in df.columns:
        minutes_factor = np.clip(
            df["minutes_avg_last_5"] / 90.0,
            0.3,
            1.0,
        )
        df["predicted_points"] *= minutes_factor

    if "low_confidence" in df.columns:
        df.loc[df["low_confidence"], "predicted_points"] *= 0.7

    return df
