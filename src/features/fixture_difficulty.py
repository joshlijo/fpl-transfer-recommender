import pandas as pd
import numpy as np

from src.config.constants import (
    HOME_ELO_BONUS,
    FIXTURE_MULTIPLIER_MIN,
    FIXTURE_MULTIPLIER_MAX,
    CS_BONUS_ELO_THRESHOLD,
    CS_BONUS_POSITIVE,
    CS_BONUS_NEGATIVE,
)

def clamp(x, low, high):
    return max(low, min(high, x))


def elo_to_difficulty_bucket(effective_elo_diff: float) -> int:
    if effective_elo_diff >= 150:
        return 1
    if effective_elo_diff >= 75:
        return 2
    if effective_elo_diff >= -75:
        return 3
    if effective_elo_diff >= -150:
        return 4
    return 5


def elo_to_base_multiplier(effective_elo_diff: float) -> float:
    raw = 1 + (effective_elo_diff / 600)
    return clamp(raw, FIXTURE_MULTIPLIER_MIN, FIXTURE_MULTIPLIER_MAX)

def explode_fixtures(fixtures_df: pd.DataFrame) -> pd.DataFrame:
    home = pd.DataFrame({
        "team_id": fixtures_df["home_team"],
        "opponent_id": fixtures_df["away_team"],
        "team_elo": fixtures_df["home_team_elo"],
        "opponent_elo": fixtures_df["away_team_elo"],
        "is_home": True,
        "gameweek": fixtures_df["gameweek"],
        "match_id": fixtures_df["match_id"],
    })

    away = pd.DataFrame({
        "team_id": fixtures_df["away_team"],
        "opponent_id": fixtures_df["home_team"],
        "team_elo": fixtures_df["away_team_elo"],
        "opponent_elo": fixtures_df["home_team_elo"],
        "is_home": False,
        "gameweek": fixtures_df["gameweek"],
        "match_id": fixtures_df["match_id"],
    })

    return pd.concat([home, away], ignore_index=True)


def build_fixture_difficulty(fixtures_df: pd.DataFrame) -> pd.DataFrame:
    df = explode_fixtures(fixtures_df.copy())

    df["effective_elo_diff"] = np.where(
        df["is_home"],
        (df["team_elo"] + HOME_ELO_BONUS) - df["opponent_elo"],
        df["team_elo"] - df["opponent_elo"],
    )

    df["difficulty_bucket"] = df["effective_elo_diff"].apply(
        elo_to_difficulty_bucket
    )

    df["fixture_multiplier"] = df["effective_elo_diff"].apply(
        elo_to_base_multiplier
    )

    df["cs_bonus"] = 0.0
    df.loc[df["effective_elo_diff"] >= CS_BONUS_ELO_THRESHOLD, "cs_bonus"] = CS_BONUS_POSITIVE
    df.loc[df["effective_elo_diff"] <= -CS_BONUS_ELO_THRESHOLD, "cs_bonus"] = CS_BONUS_NEGATIVE

    return df[
        [
            "team_id",
            "opponent_id",
            "gameweek",
            "is_home",
            "team_elo",
            "opponent_elo",
            "effective_elo_diff",
            "difficulty_bucket",
            "fixture_multiplier",
            "cs_bonus",
            "match_id",
        ]
    ]