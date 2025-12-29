from pathlib import Path
import pandas as pd

# Correct root after re-clone
base = Path(
    "data/raw/fpl-elo-insights/data/2025-2026/By Tournament/Premier League"
)

if not base.exists():
    raise RuntimeError(f"Premier League path not found: {base}")

completed = []

for p in sorted(base.iterdir()):
    if not p.name.startswith("GW"):
        continue

    stats = p / "player_gameweek_stats.csv"
    if not stats.exists():
        continue

    df = pd.read_csv(stats)
    if not df.empty:
        completed.append(p.name)

print("Completed GWs with real data:")
print(sorted(completed, key=lambda x: int(x.replace("GW", ""))))
