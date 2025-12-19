"""
Centralized modeling constants.

All numerical assumptions used in:
- rolling form
- fixture difficulty
- point prediction

MUST live here.
"""

# -------------------------------------------------
# Rolling form
# -------------------------------------------------

ROLLING_WINDOWS = [1, 3, 5]

LOW_CONFIDENCE_GAMES_THRESHOLD = 3
LOW_CONFIDENCE_MINUTES_THRESHOLD = 60


# -------------------------------------------------
# Fixture difficulty
# -------------------------------------------------

HOME_ELO_BONUS = 50

FIXTURE_MULTIPLIER_MIN = 0.70
FIXTURE_MULTIPLIER_MAX = 1.30

DIFFICULTY_BUCKETS = {
    1: "Very Easy",
    2: "Easy",
    3: "Medium",
    4: "Hard",
    5: "Very Hard",
}

CS_BONUS_ELO_THRESHOLD = 75
CS_BONUS_POSITIVE = 0.6
CS_BONUS_NEGATIVE = -0.4


# -------------------------------------------------
# Point prediction — base weighting
# -------------------------------------------------

PPG_WEIGHTS = {
    "last_5": 0.55,
    "last_3": 0.30,
    "last_1": 0.15,
}


# -------------------------------------------------
# Point prediction — adjustments
# -------------------------------------------------

ATTACK_DELTA_CAP = 0.6
DEF_DELTA_CAP = 0.5

GK_DELTA_MIN = -0.5
GK_DELTA_MAX = 0.8


# -------------------------------------------------
# Minutes reliability
# -------------------------------------------------

MIN_MINUTES_FACTOR = 0.4
MAX_MINUTES_FACTOR = 1.0


# -------------------------------------------------
# Final sanity caps (per GW)
# -------------------------------------------------

POINT_CAPS = {
    "Goalkeeper": (1.5, 8.0),
    "Defender": (1.5, 8.5),
    "Midfielder": (2.0, 10.5),
    "Forward": (2.0, 9.5),
}
