import pandas as pd
from src.config.constants import (
    ROLLING_WINDOWS,
    LOW_CONFIDENCE_GAMES_THRESHOLD,
    LOW_CONFIDENCE_MINUTES_THRESHOLD,
)


def _last_n_appearances(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Returns the last n appearances (minutes > 0) for each player.
    """
    return (
        df[df["minutes"] > 0]
        .sort_values(["player_id", "gameweek"])
        .groupby("player_id")
        .tail(n)
    )


def _aggregate_window(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Aggregates rolling stats for a given appearance window.
    """

    agg = (
        df.groupby("player_id")
        .agg(
            appearances=("gameweek", "count"),
            minutes_sum=("minutes", "sum"),
            minutes_avg=("minutes", "mean"),

            # Controlled outcome signal
            ppg=("event_points", "mean"),

            goals_avg=("goals_scored", "mean"),
            assists_avg=("assists", "mean"),
            xg_avg=("expected_goals", "mean"),
            xa_avg=("expected_assists", "mean"),

            defcon_avg=("defensive_contribution", "mean"),
            saves_avg=("saves", "mean"),
            goals_conceded_avg=("goals_conceded", "mean"),
        )
        .reset_index()
    )

    # Rename columns to window-specific names
    agg = agg.rename(
        columns={
            "appearances": f"appearances_last_{window}",
            "minutes_sum": f"minutes_sum_last_{window}",
            "minutes_avg": f"minutes_avg_last_{window}",
            "ppg": f"ppg_last_{window}",

            "goals_avg": f"goals_avg_last_{window}",
            "assists_avg": f"assists_avg_last_{window}",
            "xg_avg": f"xg_avg_last_{window}",
            "xa_avg": f"xa_avg_last_{window}",

            "defcon_avg": f"defcon_avg_last_{window}",
            "saves_avg": f"saves_avg_last_{window}",
            "goals_conceded_avg": f"goals_conceded_avg_last_{window}",
        }
    )

    return agg


def build_rolling_form_features(player_gw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds rolling form features using appearance-based windows.
    """

    df = player_gw_df.copy()

    # Hard safety check
    if df.columns.tolist().count("gameweek") > 1:
        raise ValueError("Duplicate 'gameweek' column detected")

    numeric_cols = [
        "minutes", "goals_scored", "assists",
        "expected_goals", "expected_assists",
        "defensive_contribution", "saves", "goals_conceded",
    ]

    for col in numeric_cols:
        df[col] = df.get(col, 0.0).fillna(0.0)

    features = None

    for w in ROLLING_WINDOWS:
        last_n = _last_n_appearances(df, w)
        agg = _aggregate_window(last_n, w)

        features = agg if features is None else features.merge(
            agg, on="player_id", how="outer"
        )

    features = features.fillna(0.0)

    # -------------------------------------------------
    # CONTROLLED PPG (CRITICAL)
    # -------------------------------------------------

    if (
        "ppg_last_5" in features.columns
        and "minutes_avg_last_5" in features.columns
        and "appearances_last_5" in features.columns
    ):
        # 1. Minutes dampening (penalize cameos)
        features["ppg_last_5"] = (
            features["ppg_last_5"]
            * (features["minutes_avg_last_5"] / 90.0).clip(0.4, 1.0)
        )

        # 2. Appearance dampening (small sample penalty)
        features["ppg_last_5"] *= (
            features["appearances_last_5"] / 5.0
        ).clip(0.4, 1.0)

        # 3. Hard clip to realistic FPL range
        features["ppg_last_5"] = features["ppg_last_5"].clip(0.0, 7.0)

    # -------------------------------------------------
    # Low-confidence flag
    # -------------------------------------------------

    features["low_confidence"] = (
        (features["appearances_last_5"] < LOW_CONFIDENCE_GAMES_THRESHOLD)
        | (features["minutes_avg_last_5"] < LOW_CONFIDENCE_MINUTES_THRESHOLD)
    )

    return features
