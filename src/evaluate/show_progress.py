"""Print a progress table across all self-play iterations.

Reads:
  artifacts/metrics/selfplay_iter{N}_metrics.json  — training stats (val_top1, loss)
  artifacts/metrics/eval_iter{N}.json               — eval results (W/L/D vs opponents)

Usage (from 483-Connect4-ML/):
    python src/show_progress.py
    python src/show_progress.py --csv          # machine-readable CSV
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_training(metrics_dir: Path) -> dict[int, dict]:
    out = {}
    for f in sorted(metrics_dir.glob("selfplay_iter*_metrics.json")):
        # filename: selfplay_iter{N}_metrics.json → iteration N-1 produced this model
        try:
            n = int(f.stem.split("iter")[1].split("_")[0])
        except (IndexError, ValueError):
            continue
        data = json.loads(f.read_text())
        history = data.get("history", [])
        if not history:
            continue
        best = max(history, key=lambda e: e.get("val", {}).get("top1", 0))
        out[n] = {
            "val_top1":    best.get("val", {}).get("top1", float("nan")),
            "val_top2":    best.get("val", {}).get("top2", float("nan")),
            "policy_loss": best.get("train", {}).get("policy_loss", float("nan")),
            "value_loss":  best.get("train", {}).get("value_loss", float("nan")),
            "best_epoch":  best.get("epoch", "?"),
        }
    return out


def load_evals(metrics_dir: Path) -> dict[int, dict]:
    out = {}
    for f in sorted(metrics_dir.glob("eval_iter*.json")):
        try:
            n = int(f.stem.split("iter")[1])
        except (IndexError, ValueError):
            continue
        data = json.loads(f.read_text())
        results = data.get("results", {})
        out[n] = {opp: r.get("score", float("nan")) for opp, r in results.items()}
        out[n]["_wld"] = {
            opp: (r.get("wins", 0), r.get("losses", 0), r.get("draws", 0))
            for opp, r in results.items()
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", action="store_true", help="Print machine-readable CSV instead of table")
    args = p.parse_args()

    metrics_dir = Path("artifacts/metrics")
    training = load_training(metrics_dir)
    evals    = load_evals(metrics_dir)

    all_iters = sorted(set(training) | set(evals))
    if not all_iters:
        print("No data found in artifacts/metrics/")
        return

    if args.csv:
        print("iter,val_top1,policy_loss,value_loss,vs_random,vs_prev,vs_solver")
        for n in all_iters:
            t = training.get(n, {})
            e = evals.get(n, {})
            print(f"{n},"
                  f"{t.get('val_top1', ''):.4f},"
                  f"{t.get('policy_loss', ''):.4f},"
                  f"{t.get('value_loss', ''):.4f},"
                  f"{e.get('random', ''):.3f},"
                  f"{e.get('prev', ''):.3f},"
                  f"{e.get('solver', ''):.3f}")
        return

    # Pretty table
    header = f"{'Iter':>4}  {'val_top1':>8}  {'p_loss':>7}  {'v_loss':>7}  {'vs_rand':>9}  {'vs_prev':>9}  {'vs_solv':>9}"
    print(header)
    print("-" * len(header))

    for n in all_iters:
        t = training.get(n, {})
        e = evals.get(n, {})

        val_top1 = f"{t['val_top1']:.4f}" if "val_top1" in t else "  -   "
        p_loss   = f"{t['policy_loss']:.4f}" if "policy_loss" in t else "  -   "
        v_loss   = f"{t['value_loss']:.4f}" if "value_loss" in t else "  -   "

        def score_cell(opp: str) -> str:
            if opp not in e:
                return "    -    "
            score = e[opp]
            w, l, d = e["_wld"].get(opp, (0, 0, 0))
            return f"{score:.3f}({w}W{l}L{d}D)"

        print(f"{n:>4}  {val_top1:>8}  {p_loss:>7}  {v_loss:>7}  "
              f"{score_cell('random'):>14}  {score_cell('prev'):>14}  {score_cell('solver'):>14}")

    print()
    print("Iter = model saved as selfplay_iter{N}.pth")
    print("score = (wins + 0.5*draws) / games")


if __name__ == "__main__":
    main()
