"""
Feature masks for ranking models.

These define the EXACT features each position model is allowed to see.
Must stay in sync with feature builders.
"""

COMMON_BASE = [
    # Minutes / availability
    "minutes_avg_last_5",
    "minutes_sum_last_5",
    "appearances_last_5",

    # Offensive underlying stats
    "xg_avg_last_5",
    "xa_avg_last_5",
    "goals_avg_last_5",
    "assists_avg_last_5",
    "ppg_last_5",

    # Defensive / GK stats
    "defcon_avg_last_5",
    "saves_avg_last_5",
    "goals_conceded_avg_last_5",

    # Momentum
    "xg_trend",
    "xa_trend",
    "minutes_trend",
    "defcon_trend",

    # Context
    "fixture_difficulty",
    "low_confidence",
]

RELATIVE_FEATURES = [
    "xg_avg_last_5_rel",
    "xa_avg_last_5_rel",
    "minutes_avg_last_5_rel",
    "xg_avg_last_5_z",
    "xa_avg_last_5_z",
    "minutes_avg_last_5_z",
]

RANK_FEATURE_MASKS = {
    "Goalkeeper": [
        "minutes_avg_last_5",
        "appearances_last_5",
        "saves_avg_last_5",
        "goals_conceded_avg_last_5",
        "defcon_avg_last_5",
        "minutes_trend",
        "fixture_difficulty",
        "low_confidence",
    ],

    "Defender": COMMON_BASE + RELATIVE_FEATURES,

    "Midfielder": COMMON_BASE + RELATIVE_FEATURES,

    "Forward": COMMON_BASE + RELATIVE_FEATURES,
}