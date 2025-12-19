"""
Prediction pipeline entrypoint.

This is the ONLY place where:
- data loading
- feature construction
- model inference context

are orchestrated together.
"""

import pandas as pd

from src.data.loaders import (
    load_player_gameweeks,
    load_players,
    load_fixtures,
)

from src.features.rolling_form import build_rolling_form_features
from src.features.fixture_difficulty import build_fixture_difficulty

# Phase 3A feature parity with training
from src.features.relative_features import add_relative_features
from src.features.trend_features import add_trend_features


def build_predictions(
    current_gw: int,
    horizon: int = 5,
    season: str = "2025-2026",
) -> pd.DataFrame:
    """
    Build ML-ready feature table for predicting NEXT gameweek points.
    """

    if current_gw is None:
        from src.config.settings import CURRENT_GW
        current_gw = CURRENT_GW

    next_gw = current_gw + 1

    # -------------------------------------------------
    # 1. Rolling form history (≤ current_gw)
    # -------------------------------------------------

    form_gws = list(range(current_gw - horizon, current_gw))
    player_gw_df = load_player_gameweeks(form_gws, season=season)

    if player_gw_df.empty:
        return pd.DataFrame()

    # -------------------------------------------------
    # 2. Snapshot data
    # -------------------------------------------------

    players_df = load_players(current_gw, season=season)
    fixtures_df = load_fixtures(next_gw, season=season)

    if players_df.empty or fixtures_df.empty:
        return pd.DataFrame()

    # -------------------------------------------------
    # 3. Core features
    # -------------------------------------------------

    form_df = build_rolling_form_features(player_gw_df)
    fixture_df = build_fixture_difficulty(fixtures_df)

    # -------------------------------------------------
    # 4. Assemble player × fixture table
    # -------------------------------------------------

    player_base = form_df.merge(
        players_df[["player_id", "web_name", "position", "team_code"]],
        on="player_id",
        how="left",
    )

    prediction_df = player_base.merge(
        fixture_df,
        left_on="team_code",
        right_on="team_id",
        how="inner",
    )

    if prediction_df.empty:
        return prediction_df

    # -------------------------------------------------
    # 4.1 STANDARDIZE FIXTURE DIFFICULTY NAME (CRITICAL)
    # -------------------------------------------------

    # Training expects `fixture_difficulty`
    if "fixture_multiplier" in prediction_df.columns:
        prediction_df = prediction_df.rename(
            columns={"fixture_multiplier": "fixture_difficulty"}
        )

    # -------------------------------------------------
    # 5. Phase 3A feature parity (CRITICAL)
    # -------------------------------------------------

    prediction_df["target_gw"] = next_gw

    prediction_df = add_relative_features(prediction_df)
    prediction_df = add_trend_features(prediction_df)

    # -------------------------------------------------
    # 6. Final ordering (NO predictions yet)
    # -------------------------------------------------

    prediction_df = prediction_df.sort_values(
        ["position", "player_id"]
    ).reset_index(drop=True)

    return prediction_df
