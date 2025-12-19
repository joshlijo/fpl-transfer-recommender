"""
Schema normalization utilities.

This module is the ONLY place where we:
- reconcile column name differences
- coerce dtypes
- guarantee required columns exist

IMPORTANT:
- Loaders define `gameweek`
- Normalizers NEVER invent or override `gameweek`
"""

import pandas as pd

def _require_columns(df: pd.DataFrame, required: list, context: str):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"[{context}] Missing required columns: {missing}"
        )


def _rename_if_present(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    rename_map = {k: v for k, v in mapping.items() if k in df.columns}
    return df.rename(columns=rename_map)

def normalize_player_gameweek_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes player_gameweek_stats.csv across seasons.

    Guarantees:
    - player_id
    - gameweek (already provided by loader)
    - minutes
    """

    df = df.copy()

    df = _rename_if_present(
        df,
        {
            "id": "player_id",
        },
    )

    _require_columns(
        df,
        required=["player_id", "gameweek", "minutes"],
        context="player_gameweek_stats",
    )

    df["player_id"] = df["player_id"].astype(int)
    df["gameweek"] = df["gameweek"].astype(int)
    df["minutes"] = df["minutes"].fillna(0).astype(float)

    return df

def normalize_players_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes players.csv.

    Guarantees:
    - player_id
    - team_code
    - position
    """

    df = df.copy()

    df = _rename_if_present(
        df,
        {
            "id": "player_id",
            "element_type": "position",
        },
    )

    _require_columns(
        df,
        required=["player_id", "team_code", "position"],
        context="players",
    )

    df["player_id"] = df["player_id"].astype(int)
    df["team_code"] = df["team_code"].astype(int)

    position_map = {
        1: "Goalkeeper",
        2: "Defender",
        3: "Midfielder",
        4: "Forward",
        "GKP": "Goalkeeper",
        "DEF": "Defender",
        "MID": "Midfielder",
        "FWD": "Forward",
    }

    df["position"] = df["position"].map(position_map).fillna(df["position"])

    return df

def normalize_fixtures_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes fixtures.csv.

    Guarantees:
    - home_team
    - away_team
    - home_team_elo
    - away_team_elo
    - gameweek
    """

    df = df.copy()

    df = _rename_if_present(
        df,
        {
            "event": "gameweek",
            "gw": "gameweek",
        },
    )

    _require_columns(
        df,
        required=[
            "home_team",
            "away_team",
            "home_team_elo",
            "away_team_elo",
            "gameweek",
        ],
        context="fixtures",
    )

    df["home_team"] = df["home_team"].astype(int)
    df["away_team"] = df["away_team"].astype(int)
    df["gameweek"] = df["gameweek"].astype(int)

    df["home_team_elo"] = df["home_team_elo"].astype(float)
    df["away_team_elo"] = df["away_team_elo"].astype(float)

    return df