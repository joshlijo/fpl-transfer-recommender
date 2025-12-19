"""
Centralized data loaders.

All CSV reading + schema normalization lives here.
Feature and model code should NEVER touch file paths.
"""

from pathlib import Path
import pandas as pd

from src.data.schema import (
    normalize_player_gameweek_df,
    normalize_players_df,
    normalize_fixtures_df,
)

# -------------------------------------------------
# Base paths
# -------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "fpl-elo-insights"
)

DEFAULT_SEASON = "2025-2026"
DEFAULT_TOURNAMENT = "Premier League"


# -------------------------------------------------
# Helpers
# -------------------------------------------------

def _season_path(season: str) -> Path:
    return DATA_ROOT / season / "By Tournament" / DEFAULT_TOURNAMENT


# -------------------------------------------------
# Loaders
# -------------------------------------------------

def load_player_gameweeks(
    gws: list[int],
    season: str = DEFAULT_SEASON,
) -> pd.DataFrame:
    """
    Load and normalize player_gameweek_stats for multiple GWs.

    NOTE:
    - This function is the SINGLE source of truth for `gameweek`.
    - Schema normalizers must NOT create or rename gameweek.
    """

    dfs = []
    base = _season_path(season)

    for gw in gws:
        path = base / f"GW{gw}" / "player_gameweek_stats.csv"
        if not path.exists():
            continue

        df = pd.read_csv(path)

        # ✅ authoritative gameweek assignment
        df["gameweek"] = gw

        df = normalize_player_gameweek_df(df)
        dfs.append(df)

    if not dfs:
        raise RuntimeError("No player_gameweek_stats loaded")

    return pd.concat(dfs, ignore_index=True)


def load_players(
    gw: int,
    season: str = DEFAULT_SEASON,
) -> pd.DataFrame:
    """
    Load and normalize players.csv for a specific GW snapshot.
    """

    path = _season_path(season) / f"GW{gw}" / "players.csv"

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    return normalize_players_df(df)


def load_fixtures(
    gw: int,
    season: str = DEFAULT_SEASON,
) -> pd.DataFrame:
    """
    Load and normalize fixtures.csv for a specific GW.
    """

    path = _season_path(season) / f"GW{gw}" / "fixtures.csv"

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    return normalize_fixtures_df(df)
