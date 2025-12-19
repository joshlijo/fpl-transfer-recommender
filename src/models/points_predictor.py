import pandas as pd

from src.config.constants import (
    PPG_WEIGHTS,
    ATTACK_DELTA_CAP,
    DEF_DELTA_CAP,
    GK_DELTA_MIN,
    GK_DELTA_MAX,
    MIN_MINUTES_FACTOR,
    MAX_MINUTES_FACTOR,
    POINT_CAPS,
)

# -------------------------------------------------
# Utility
# -------------------------------------------------

def clamp(x, low, high):
    return max(low, min(high, x))


# -------------------------------------------------
# Core prediction logic
# -------------------------------------------------

def compute_base_points(row: pd.Series) -> float:
    return (
        PPG_WEIGHTS["last_5"] * row.get("ppg_last_5", 0.0)
        + PPG_WEIGHTS["last_3"] * row.get("ppg_last_3", 0.0)
        + PPG_WEIGHTS["last_1"] * row.get("ppg_last_1", 0.0)
    )


def compute_attack_delta(row: pd.Series) -> float:
    delta = (
        0.6 * (row.get("xg_avg_last_5", 0.0) - row.get("goals_avg_last_5", 0.0))
        + 0.4 * (row.get("xa_avg_last_5", 0.0) - row.get("assists_avg_last_5", 0.0))
    )
    return clamp(delta, -ATTACK_DELTA_CAP, ATTACK_DELTA_CAP)


def compute_def_delta(row: pd.Series) -> float:
    delta = 0.4 * row.get("defcon_avg_last_5", 0.0)
    return clamp(delta, 0.0, DEF_DELTA_CAP)


def compute_gk_delta(row: pd.Series) -> float:
    delta = (
        0.5 * row.get("saves_avg_last_5", 0.0)
        - 0.3 * row.get("goals_conceded_avg_last_5", 0.0)
    )
    return clamp(delta, GK_DELTA_MIN, GK_DELTA_MAX)


def compute_minutes_factor(row: pd.Series) -> float:
    factor = row.get("minutes_avg_last_5", 0.0) / 90.0
    return clamp(factor, MIN_MINUTES_FACTOR, MAX_MINUTES_FACTOR)


def predict_points(row: pd.Series) -> float:
    position = row["position"]

    base_points = compute_base_points(row)

    if position == "Goalkeeper":
        raw = base_points + compute_gk_delta(row)
    elif position == "Defender":
        raw = base_points + compute_def_delta(row)
    elif position == "Midfielder":
        raw = base_points + compute_attack_delta(row) + compute_def_delta(row)
    else:
        raw = base_points + compute_attack_delta(row)

    reliable = raw * compute_minutes_factor(row)

    adjusted = (
        reliable * row.get("fixture_multiplier", 1.0)
        + row.get("cs_bonus", 0.0)
    )

    min_cap, max_cap = POINT_CAPS[position]
    return clamp(adjusted, min_cap, max_cap)


def run_point_predictions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["predicted_points"] = df.apply(predict_points, axis=1)
    return df