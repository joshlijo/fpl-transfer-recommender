# Fantasy Premier League Points Prediction & Transfer Recommendation System

## Overview

The purpose of this project is to act as an end-to-end system for ranking players within the Fantasy Premier League that uses a leak-free machine learning process to predict their expected value for the upcoming matchup.
Rather than trying to model a prediction of fantasy points perfectly (which has a high degree of noise), the aim is to develop a robust decision support system that ranks players sensibly, like current FPL analytics systems, based on uncertainty.

## Key Design Principles

### 1. No Target Leakage

Every characteristic is built solely using information available prior to the predicted gameweek.
Raw season PPG and forward-looking statistics are avoided or tightly managed.

### 2. Causal, Rolling Features

Every training row corresponds to a player’s status before gameweek t, where the labels are picked from gameweek t itself.

Features include:

* Rolling appearance-based windows (last 5 matches)
* Minutes stability and availability
* Underlying attacking metrics (xG, xA)
* Defensive contribution metrics
* Fixture difficulty and short-term trends

### 3. Position-Specific Models

Gradient boosted models are trained for the following separately:

- Goalkeepers
- Defenders
- Midfield
- Forwards

Every role is assigned a corresponding feature mask, reflecting role-specific values.

### 4. Controlled Outcome Signal (PPG)

The pts/game rate is reinstated in a gentle fashion employing rolling windowing, minute basis dampening, sample size penalties, and hard clipping.
This maintains the scoring intuition while preventing single-game outbreaks from being dominant.

### 5. Expected Value vs Upside

All core models are based on Expected Value (EV) rather than potential maximum upside.
Players with high levels of explosive power but more uncertainty regarding playing time or position could be ranked lower than more settled players. This is deliberate and the result of a conservative approach to modeling.
A separate level of decision making (in addition to the ML model) can factor in the ceiling, risk appetite, and differences in strategy without compromising the estimates for EV.

## Project Structure

```text
src/
├── data/        # Data loaders
├── features/    # Feature engineering
├── pipeline/    # Training & inference builders
├── models/      # Training & calibration
├── inference/   # Prediction entrypoints
├── config/      # Constants & feature masks
models/
└── v1/          # Frozen model artifacts
```

## How to Run
```bash
python -m src.models.train_gbm_models
python -m src.models.calibrate_models
python -m src.inference.predict_ranks
```

## Model Versioning

The current stable version is labeled as v1.0. It has these properties: Leak-free, Feature stable, Calibrated, Frozen. Any further enhancements (such as ceiling modeling/transfer, for example) are established *on top of this existing baseline, not by modifying this existing baseline*.

## Author

### Joshua

Personal project in Applied ML, Feature Engineering, and Decision-Oriented Modeling.
