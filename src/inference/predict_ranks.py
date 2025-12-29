import joblib
import pandas as pd

from src.pipeline.build_predictions import build_predictions
from src.config.feature_masks import RANK_FEATURE_MASKS
from src.models.postprocess_predictions import postprocess_predictions

MODELS_DIR = "models"
POSITIONS = ["Goalkeeper", "Defender", "Midfielder", "Forward"]


def predict_ranks():
    # 🔑 DO NOT pass current_gw — let pipeline decide
    df = build_predictions()
    outputs = []

    for position in POSITIONS:
        pos_df = df[df["position"] == position].copy()
        if pos_df.empty:
            continue

        features = RANK_FEATURE_MASKS[position]

        model = joblib.load(f"{MODELS_DIR}/{position.lower()}_gbm.pkl")
        pos_df["raw_score"] = model.predict(pos_df[features])

        calibrator = joblib.load(
            f"{MODELS_DIR}/{position.lower()}_calibrator.pkl"
        )

        pos_df["predicted_points"] = calibrator.predict(
            pos_df["raw_score"].values.reshape(-1, 1)
        )

        pos_df = postprocess_predictions(pos_df)
        outputs.append(pos_df)

    final_df = (
        pd.concat(outputs)
        .sort_values(
            ["position", "predicted_points"],
            ascending=[True, False],
        )
    )

    return final_df


if __name__ == "__main__":
    df = predict_ranks()

    target_gw = int(df["target_gw"].iloc[0])
    print(f"\n=== RANKED PREDICTIONS FOR GW {target_gw} ===\n")

    for pos in df["position"].unique():
        print(f"\n--- TOP 10 {pos.upper()} ---")
        print(
            df[df["position"] == pos]
            .head(10)[["web_name", "predicted_points"]]
            .round(2)
            .to_string(index=False)
        )
