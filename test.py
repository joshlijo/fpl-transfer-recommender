from src.pipeline.build_training_dataset import build_training_dataset
df = build_training_dataset(6, 16)
print(df["ppg_last_5"].describe())