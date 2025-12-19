"""
Relative features: player vs positional peers.
"""

import pandas as pd


RELATIVE_COLS = [
    "xg_avg_last_5",
    "xa_avg_last_5",
    "minutes_avg_last_5",
    "goals_avg_last_5",
    "assists_avg_last_5",
    "defcon_avg_last_5",
]


def add_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    group_cols = ["position", "target_gw"]

    for col in RELATIVE_COLS:
        if col not in df.columns:
            continue

        grp = df.groupby(group_cols)[col]

        df[f"{col}_rel"] = grp.transform(lambda x: x - x.mean())
        df[f"{col}_z"] = grp.transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-6)
        )

    return df
