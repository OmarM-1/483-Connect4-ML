"""
Train a RandomForest on the filtered UCI-Midgame-d30 dataset (score_gap > 0)
using the same 51-dim feature extractor as bridge/ml_policy.py so the saved
pkl drops directly into MLPolicyClient without any bridge changes.

Features (51-dim):
  42  board cells — side-to-move perspective (+1=self, -1=opp, 0=empty)
   7  column heights
   1  move count
   1  to_move (+1=P1, -1=P2)

Run from 483-Connect4-ML/:
  .venv/bin/python src/train/train_rf_filtered.py

Outputs:
  artifacts/models/connect4_rf_filtered_bridge_n300_s42.pkl
  artifacts/metrics/connect4_rf_filtered_bridge_n300_s42_metrics.json

To run in the bridge after training:
  bash launch_bridge.sh --our-username "You" --strategy ml \\
    --ml-model artifacts/models/connect4_rf_filtered_bridge_n300_s42.pkl
"""

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.evaluate.evaluate import evaluate_predictions, print_metrics

DATA_DIR      = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"
SEED          = 42
N_ESTIMATORS  = 300
MODEL_NAME    = f"connect4_rf_filtered_bridge_n{N_ESTIMATORS}_s{SEED}"

FILTERED_TRAIN = DATA_DIR / "UCI-Midgame-d30-filtered.train.csv"
FILTERED_VAL   = DATA_DIR / "UCI-Midgame-d30-filtered.val.csv"

_ROWS = 6
_COLS = 7


def featurize(sequence: str) -> np.ndarray:
    board   = np.zeros((_ROWS, _COLS), dtype=np.int8)
    heights = np.zeros(_COLS, dtype=np.int8)
    player  = 1
    for ch in str(sequence).strip():
        col = int(ch) - 1
        board[heights[col], col] = player
        heights[col] += 1
        player = 2 if player == 1 else 1
    perspective = 1 if player == 1 else -1
    return np.concatenate([
        (board * perspective).reshape(-1).astype(np.float32),
        heights.astype(np.float32),
        np.array([len(str(sequence).strip())], dtype=np.float32),
        np.array([float(perspective)], dtype=np.float32),
    ])


def legal_mask(sequence: str) -> np.ndarray:
    heights = np.zeros(_COLS, dtype=np.int8)
    for ch in str(sequence).strip():
        heights[int(ch) - 1] += 1
    return (heights < _ROWS).astype(np.int8)


def build_arrays(df: pd.DataFrame):
    seqs  = df["sequence"].astype(str).tolist()
    X     = np.stack([featurize(s) for s in seqs])
    y     = df["best_move"].astype(int).to_numpy() - 1
    masks = np.stack([legal_mask(s) for s in seqs])
    return X, y, masks


def main():
    if not FILTERED_TRAIN.exists() or not FILTERED_VAL.exists():
        print("ERROR: filtered CSVs not found.")
        print(f"  expected: {FILTERED_TRAIN}")
        print(f"  expected: {FILTERED_VAL}")
        print("Run the notebook §12 or filter manually:")
        print("  python -c \"import pandas as pd; df=pd.read_csv('data/UCI-Midgame-d30.train.csv'); df[df.score_gap>0].to_csv('data/UCI-Midgame-d30-filtered.train.csv', index=False)\"")
        sys.exit(1)

    print("Loading filtered datasets...")
    train_df = pd.read_csv(FILTERED_TRAIN, dtype={"sequence": str})
    val_df   = pd.read_csv(FILTERED_VAL,   dtype={"sequence": str})
    print(f"  train: {len(train_df):,} rows  |  val: {len(val_df):,} rows")

    print("\nBuilding 51-dim features...")
    t0 = time.time()
    X_train, y_train, train_masks = build_arrays(train_df)
    X_val,   y_val,   val_masks   = build_arrays(val_df)
    print(f"  done in {time.time()-t0:.1f}s  —  X_train {X_train.shape}, X_val {X_val.shape}")

    print(f"\nTraining RandomForest (n_estimators={N_ESTIMATORS}, max_depth=None, seed={SEED})...")
    t0 = time.time()
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=None,
        criterion="gini",
        random_state=SEED,
        n_jobs=-1,
        verbose=1,
    )
    model.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"  trained in {train_time:.1f}s")

    print("\nEvaluating on filtered val set...")
    y_pred  = model.predict(X_val)
    y_score = model.predict_proba(X_val)
    metrics = evaluate_predictions(
        y_true=y_val, y_pred=y_pred, y_score=y_score, legal_masks=val_masks
    )
    print_metrics(metrics)

    model_path = ARTIFACTS_DIR / "models" / f"{MODEL_NAME}.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as f:
        pickle.dump(model, f)
    print(f"\nModel saved : {model_path}  ({model_path.stat().st_size/1024/1024:.1f} MB)")

    metrics_out = {
        "model": MODEL_NAME,
        "train_csv": str(FILTERED_TRAIN.name),
        "val_csv": str(FILTERED_VAL.name),
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "feature_dim": int(X_train.shape[1]),
        "n_estimators": N_ESTIMATORS,
        "seed": SEED,
        "train_seconds": round(train_time, 2),
        "top1_accuracy": round(float(metrics["top1_accuracy"]), 4),
        "top2_accuracy": round(float(metrics["top2_accuracy"]), 4),
        "macro_f1": round(float(metrics["macro_f1"]), 4),
        "illegal_move_rate": round(float(metrics["illegal_move_rate"]), 4),
    }
    metrics_path = ARTIFACTS_DIR / "metrics" / f"{MODEL_NAME}_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"Metrics saved: {metrics_path}")

    print("\nTo run in bridge:")
    print(f"  bash launch_bridge.sh --our-username \"You\" --strategy ml --ml-model {model_path}")


if __name__ == "__main__":
    main()
