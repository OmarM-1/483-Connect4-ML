from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

FILES = [
    "UCI-Midgame-d30.csv",
    "UCI-Midgame-d30.train.csv",
    "UCI-Midgame-d30.val.csv"
]

for name in FILES:
    path = DATA_DIR / name
    print(f"Checking {path}...")
    df = pd.read_csv(path, low_memory=False)
    print(f"\n{name}")
    print("-" * len(name))
    print("shape:", df.shape)
    print("columns:", df.columns.tolist())