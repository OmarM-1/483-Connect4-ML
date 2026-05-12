"""Train AlphaZero-style Connect4Net from solver-labeled CSV data.

Policy target: best_move (1-based → one-hot over 7 cols)
Value target:  solver_score normalized to [0, 1]  (column required in CSV;
               if absent, value head is skipped and only policy is trained)

Usage (from 483-Connect4-ML/):
    python src/train_network.py \
        --train-csv data/UCI-Midgame-d30.train.csv \
        --val-csv   data/UCI-Midgame-d30.val.csv \
        --model-out artifacts/models/connect4_net.pth
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from network import (
    Connect4Net,
    NetConfig,
    TrainConfig,
    sequence_to_tensor,
)

COLS = 7
SCORE_CLIP = 22  # solver scores are bounded; clip then normalize to [0,1]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class Connect4Dataset(Dataset):
    def __init__(self, df: pd.DataFrame, has_score: bool) -> None:
        self.sequences  = df["sequence"].astype(str).tolist()
        self.policies   = (df["best_move"].astype(int).values - 1)  # 0-based
        self.has_score  = has_score
        if has_score:
            scores = df["solver_score"].astype(float).values
            scores = np.clip(scores, -SCORE_CLIP, SCORE_CLIP)
            # Normalize to [0, 1]: current player's win probability
            # positive score = good for current player
            self.values = ((scores / SCORE_CLIP) + 1.0) / 2.0
        else:
            self.values = np.zeros(len(df), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        x = sequence_to_tensor(self.sequences[idx])
        y_policy = int(self.policies[idx])
        y_value  = float(self.values[idx])
        return x, y_policy, y_value


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_epoch(
    net: Connect4Net,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    has_score: bool,
) -> dict:
    net.train()
    policy_criterion = nn.CrossEntropyLoss()
    value_criterion  = nn.MSELoss()

    total_policy_loss = 0.0
    total_value_loss  = 0.0
    correct = 0
    n = 0

    for x, y_policy, y_value in loader:
        x        = x.to(device)
        y_policy = y_policy.to(device)
        y_value  = y_value.float().to(device)

        optimizer.zero_grad()
        pred_policy, pred_value = net(x)

        policy_loss = policy_criterion(pred_policy, y_policy)
        loss = policy_loss
        if has_score:
            value_loss = value_criterion(pred_value, y_value)
            loss = policy_loss + value_loss
            total_value_loss += value_loss.item() * len(x)

        loss.backward()
        optimizer.step()

        total_policy_loss += policy_loss.item() * len(x)
        correct += (pred_policy.argmax(dim=1) == y_policy).sum().item()
        n += len(x)

    return {
        "policy_loss": total_policy_loss / n,
        "value_loss":  total_value_loss / n if has_score else None,
        "top1":        correct / n,
    }


@torch.no_grad()
def evaluate(
    net: Connect4Net,
    loader: DataLoader,
    device: torch.device,
    has_score: bool,
) -> dict:
    net.eval()
    policy_criterion = nn.CrossEntropyLoss()
    value_criterion  = nn.MSELoss()

    total_policy_loss = 0.0
    total_value_loss  = 0.0
    top1 = top2 = 0
    n = 0

    for x, y_policy, y_value in loader:
        x        = x.to(device)
        y_policy = y_policy.to(device)
        y_value  = y_value.float().to(device)

        pred_policy, pred_value = net(x)

        total_policy_loss += policy_criterion(pred_policy, y_policy).item() * len(x)
        if has_score:
            total_value_loss += value_criterion(pred_value, y_value).item() * len(x)

        top2_preds = pred_policy.topk(2, dim=1).indices
        top1 += (top2_preds[:, 0] == y_policy).sum().item()
        top2 += (top2_preds == y_policy.unsqueeze(1)).any(dim=1).sum().item()
        n += len(x)

    return {
        "policy_loss": total_policy_loss / n,
        "value_loss":  total_value_loss / n if has_score else None,
        "top1":        top1 / n,
        "top2":        top2 / n,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-csv", type=Path, default=Path("data/UCI-Midgame-d30.train.csv"))
    p.add_argument("--val-csv",   type=Path, default=Path("data/UCI-Midgame-d30.val.csv"))
    p.add_argument("--model-out", type=Path, default=Path("artifacts/models/connect4_net.pth"))
    p.add_argument("--metrics-out", type=Path, default=Path("artifacts/metrics/connect4_net_metrics.json"))
    p.add_argument("--filters",      type=int, default=32)
    p.add_argument("--n-residuals",  type=int, default=3)
    p.add_argument("--n-fc-layers",  type=int, default=2)
    p.add_argument("--epochs",       type=int, default=10)
    p.add_argument("--batch-size",   type=int, default=512)
    p.add_argument("--lr",           type=float, default=0.01)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--device",       default=None, help="cuda or cpu (auto-detect if omitted)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    train_df = pd.read_csv(args.train_csv, dtype={"sequence": "string"}, low_memory=False)
    val_df   = pd.read_csv(args.val_csv,   dtype={"sequence": "string"}, low_memory=False)
    has_score = "solver_score" in train_df.columns
    print(f"Train rows: {len(train_df)}  Val rows: {len(val_df)}  Value head: {'on' if has_score else 'off (no solver_score column)'}")

    train_ds = Connect4Dataset(train_df, has_score)
    val_ds   = Connect4Dataset(val_df,   has_score)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    cfg = NetConfig(filters=args.filters, n_residuals=args.n_residuals, n_fc_layers=args.n_fc_layers)
    net = Connect4Net(cfg).to(device)
    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}  Config: filters={args.filters} residuals={args.n_residuals}")

    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(args.epochs * 0.6), int(args.epochs * 0.85)],
        gamma=0.1,
    )

    best_top1 = 0.0
    history = []
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        train_stats = train_epoch(net, train_loader, optimizer, device, has_score)
        val_stats   = evaluate(net, val_loader, device, has_score)
        scheduler.step()

        elapsed = time.time() - t0
        vl = f"{val_stats['value_loss']:.4f}" if val_stats['value_loss'] is not None else "n/a"
        print(
            f"Epoch {epoch:>3}/{args.epochs}  "
            f"train_top1={train_stats['top1']:.4f}  "
            f"val_top1={val_stats['top1']:.4f}  val_top2={val_stats['top2']:.4f}  "
            f"val_policy_loss={val_stats['policy_loss']:.4f}  val_value_loss={vl}  "
            f"elapsed={elapsed:.1f}s"
        )

        history.append({"epoch": epoch, "train": train_stats, "val": val_stats})

        if val_stats["top1"] > best_top1:
            best_top1 = val_stats["top1"]
            args.model_out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"net_state_dict": net.state_dict(), "net_config": cfg.__dict__, "seed": args.seed}, args.model_out)
            print(f"  -> saved best model (val_top1={best_top1:.4f})")

    metrics = {
        "train_csv": str(args.train_csv),
        "val_csv": str(args.val_csv),
        "net_config": cfg.__dict__,
        "seed": args.seed,
        "epochs": args.epochs,
        "best_val_top1": best_top1,
        "best_val_top2": history[max(range(len(history)), key=lambda i: history[i]["val"]["top1"])]["val"]["top2"],
        "history": history,
    }
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.write_text(json.dumps(metrics, indent=2))

    print(f"\nBest val top-1: {best_top1:.4f}")
    print(f"Model: {args.model_out}")
    print(f"Metrics: {args.metrics_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
