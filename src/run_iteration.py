"""Run one or more AlphaZero self-play iterations.

Each iteration:
  1. Generate self-play games → data/selfplay_iter{N}.csv
  2. Fine-tune network       → artifacts/models/selfplay_iter{N+1}.pth
  3. Log stats               → artifacts/metrics/selfplay_log.json

Models from every iteration are kept — roll back by pointing --start-model
at an earlier checkpoint.

Usage (from 483-Connect4-ML/):
    # Single iteration starting from supervised model
    python src/run_iteration.py \\
        --start-model artifacts/models/connect4_net.pth \\
        --iter 0 \\
        --games-per-iter 300 --simulations 100 \\
        --supervised-csv data/UCI-Midgame-d30.train.csv

    # Resume from iteration 3 for 5 more iterations
    python src/run_iteration.py \\
        --start-model artifacts/models/selfplay_iter3.pth \\
        --iter 3 --n-iters 5 \\
        --games-per-iter 300 --simulations 100
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start-model",    required=True, type=Path, help="Initial .pth checkpoint")
    p.add_argument("--iter",           type=int, default=0,   help="Starting iteration number")
    p.add_argument("--n-iters",        type=int, default=1,   help="Number of iterations to run")
    p.add_argument("--filters",        type=int, default=64)
    p.add_argument("--n-residuals",    type=int, default=6)
    p.add_argument("--games-per-iter", type=int, default=300)
    p.add_argument("--simulations",    type=int, default=100)
    p.add_argument("--epochs",         type=int, default=5)
    p.add_argument("--lr",             type=float, default=0.001)
    p.add_argument("--batch-size",     type=int, default=512)
    p.add_argument("--supervised-csv", type=Path, default=None)
    p.add_argument("--sup-ratio",      type=float, default=3.0)
    p.add_argument("--val-csv",        type=Path, default=Path("data/UCI-Midgame-d30.val.csv"))
    p.add_argument("--device",         default=None)
    p.add_argument("--seed",           type=int, default=42)
    p.add_argument("--no-mirror",      action="store_true")
    return p.parse_args()


def run(cmd: list[str]) -> None:
    print(f"\n>>> {' '.join(str(c) for c in cmd)}\n")
    result = subprocess.run(cmd, check=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")


def main() -> int:
    args = parse_args()

    src_dir      = Path(__file__).parent
    data_dir     = Path("data")
    models_dir   = Path("artifacts/models")
    metrics_dir  = Path("artifacts/metrics")
    log_path     = metrics_dir / "selfplay_log.json"

    data_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Load existing log
    log: list[dict] = []
    if log_path.exists():
        log = json.loads(log_path.read_text())

    current_model = args.start_model
    python = sys.executable

    for i in range(args.iter, args.iter + args.n_iters):
        print(f"\n{'='*60}")
        print(f"ITERATION {i}")
        print(f"{'='*60}")
        t_iter = time.time()

        sp_csv    = data_dir   / f"selfplay_iter{i}.csv"
        next_model = models_dir / f"selfplay_iter{i+1}.pth"
        metrics_out = metrics_dir / f"selfplay_iter{i+1}_metrics.json"

        # Step 1: generate self-play games
        gen_cmd = [
            python, str(src_dir / "self_play.py"),
            "--model",       str(current_model),
            "--filters",     str(args.filters),
            "--n-residuals", str(args.n_residuals),
            "--games",       str(args.games_per_iter),
            "--simulations", str(args.simulations),
            "--out",         str(sp_csv),
            "--seed",        str(args.seed + i),
        ]
        if args.device:
            gen_cmd += ["--device", args.device]
        if args.no_mirror:
            gen_cmd.append("--no-mirror")
        run(gen_cmd)

        # Step 2: fine-tune
        train_cmd = [
            python, str(src_dir / "train_selfplay.py"),
            "--data-csv",    str(sp_csv),
            "--init-model",  str(current_model),
            "--model-out",   str(next_model),
            "--metrics-out", str(metrics_out),
            "--val-csv",     str(args.val_csv),
            "--filters",     str(args.filters),
            "--n-residuals", str(args.n_residuals),
            "--epochs",      str(args.epochs),
            "--lr",          str(args.lr),
            "--batch-size",  str(args.batch_size),
            "--seed",        str(args.seed),
        ]
        if args.supervised_csv:
            train_cmd += ["--supervised-csv", str(args.supervised_csv),
                          "--sup-ratio", str(args.sup_ratio)]
        if args.device:
            train_cmd += ["--device", args.device]
        run(train_cmd)

        elapsed = time.time() - t_iter
        log.append({
            "iteration":     i,
            "model_in":      str(current_model),
            "model_out":     str(next_model),
            "games":         args.games_per_iter,
            "simulations":   args.simulations,
            "elapsed_sec":   round(elapsed, 1),
        })
        log_path.write_text(json.dumps(log, indent=2))
        print(f"\nIteration {i} done in {elapsed:.1f}s  ->  {next_model}")

        current_model = next_model

    print(f"\nAll iterations complete. Final model: {current_model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
