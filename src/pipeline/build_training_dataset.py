"""
Build supervised training dataset for FPL point prediction.

Each row represents:
- player state BEFORE GW t
- fixture difficulty FOR GW t
- label = actual event_points IN GW t
"""

from typing import List
import pandas as pd

from src.data.loaders import (
    load_player_gameweeks,
    load_players,
    load_fixtures,
)
from src.features.rolling_form import build_rolling_form_features
from src.features.fixture_difficulty import build_fixture_difficulty
from src.features.relative_features import add_relative_features
from src.features.trend_features import add_trend_features


def build_training_dataset(
    start_gw: int,
    end_gw: int,
    season: str = "2025-2026",
) -> pd.DataFrame:

    rows: List[pd.DataFrame] = []

    for target_gw in range(start_gw, end_gw + 1):

        form_gws = list(range(target_gw - 5, target_gw))
        player_gw_df = load_player_gameweeks(form_gws, season=season)

        if player_gw_df.empty:
            continue

        form_df = build_rolling_form_features(player_gw_df)

        fixtures_df = load_fixtures(target_gw, season=season)
        fixture_df = build_fixture_difficulty(fixtures_df)

        players_df = load_players(target_gw - 1, season=season)

        feature_df = (
            form_df
            .merge(
                players_df[["player_id", "position", "team_code"]],
                on="player_id",
                how="left",
            )
            .merge(
                fixture_df,
                left_on="team_code",
                right_on="team_id",
                how="inner",
            )
        )

        # Standardize fixture difficulty column name
        if "fixture_multiplier" in feature_df.columns:
            feature_df = feature_df.rename(
                columns={"fixture_multiplier": "fixture_difficulty"}
            )

        if feature_df.empty:
            continue

        label_df = (
            load_player_gameweeks([target_gw], season=season)
            [["player_id", "event_points"]]
            .rename(columns={"event_points": "target_points"})
        )

        full_df = feature_df.merge(label_df, on="player_id", how="inner")
        full_df["target_gw"] = target_gw

        rows.append(full_df)

    if not rows:
        raise RuntimeError("Training dataset is empty")

    dataset = pd.concat(rows, ignore_index=True)

    dataset = add_relative_features(dataset)
    dataset = add_trend_features(dataset)

    return dataset.sort_values(
        ["target_gw", "player_id"]
    ).reset_index(drop=True)
