"""
Trend features based on underlying performance, not fantasy points.
"""

import pandas as pd


def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if {"xg_avg_last_3", "xg_avg_last_5"}.issubset(df.columns):
        df["xg_trend"] = df["xg_avg_last_3"] - df["xg_avg_last_5"]

    if {"xa_avg_last_3", "xa_avg_last_5"}.issubset(df.columns):
        df["xa_trend"] = df["xa_avg_last_3"] - df["xa_avg_last_5"]

    if {"minutes_avg_last_3", "minutes_avg_last_5"}.issubset(df.columns):
        df["minutes_trend"] = (
            df["minutes_avg_last_3"] - df["minutes_avg_last_5"]
        )

    if {"defcon_avg_last_3", "defcon_avg_last_5"}.issubset(df.columns):
        df["defcon_trend"] = (
            df["defcon_avg_last_3"] - df["defcon_avg_last_5"]
        )

    return df