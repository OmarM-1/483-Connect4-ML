"""Fine-tune Connect4Net on self-play data with soft policy targets.

Policy target: MCTS visit-count distribution (7-dim float, sums to 1)
Value target:  game outcome normalized to [0, 1]

Loss:
    L_policy = -sum(pi_mcts * log(pi_net + eps))   # cross-entropy, soft targets
    L_value  = MSE(v_net, outcome)
    L        = L_policy + L_value

Optionally mixes in supervised solver data (--supervised-csv) to prevent
catastrophic forgetting of midgame knowledge.

Checkpoint saved in same format as train_network.py so MCTSClient / Connect4NetClient
can load it without changes: {"net_state_dict", "net_config", "seed"}

Usage (from 483-Connect4-ML/):
    python src/train_selfplay.py \\
        --data-csv     data/selfplay_iter0.csv \\
        --init-model   artifacts/models/connect4_net.pth \\
        --model-out    artifacts/models/selfplay_iter1.pth \\
        --supervised-csv data/UCI-Midgame-d30.train.csv \\
        --filters 64 --n-residuals 6 \\
        --epochs 5 --lr 0.001
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

COLS      = 7
LOG_EPS   = 1e-8
SCORE_CLIP = 22


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class SelfPlayDataset(Dataset):
    """Rows from self_play.py CSV: (sequence, mcts_policy, outcome)."""

    def __init__(self, df: pd.DataFrame) -> None:
        from src.train.network import sequence_to_tensor
        self._seq_to_tensor = sequence_to_tensor
        self.sequences = df["sequence"].fillna("").astype(str).tolist()
        self.policies  = [
            np.fromstring(s, sep=" ", dtype=np.float32)
            for s in df["mcts_policy"].astype(str)
        ]
        self.outcomes  = df["outcome"].astype(float).values.astype(np.float32)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        x  = self._seq_to_tensor(self.sequences[idx])
        pi = torch.from_numpy(self.policies[idx])   # (7,) float32
        z  = float(self.outcomes[idx])
        return x, pi, z


class SupervisedDataset(Dataset):
    """Solver-labeled CSV. Converts best_move → one-hot policy, solver_score → outcome."""

    def __init__(self, df: pd.DataFrame) -> None:
        from src.train.network import sequence_to_tensor
        self._seq_to_tensor = sequence_to_tensor
        self.sequences = df["sequence"].fillna("").astype(str).tolist()
        moves = df["best_move"].astype(int).values - 1  # 0-based
        self.policies = np.eye(COLS, dtype=np.float32)[moves]  # one-hot (N, 7)

        if "solver_score" in df.columns:
            scores = df["solver_score"].astype(float).values
            scores = np.clip(scores, -SCORE_CLIP, SCORE_CLIP)
            self.outcomes = ((scores / SCORE_CLIP) + 1.0) / 2.0
        else:
            self.outcomes = np.full(len(df), 0.5, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        x  = self._seq_to_tensor(self.sequences[idx])
        pi = torch.from_numpy(self.policies[idx])
        z  = float(self.outcomes[idx])
        return x, pi, z


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def soft_policy_loss(pred_policy: torch.Tensor, target_pi: torch.Tensor) -> torch.Tensor:
    """Cross-entropy loss with soft policy targets.

    pred_policy: (N, 7) softmax probabilities from network
    target_pi:   (N, 7) MCTS visit distribution (sums to 1)
    """
    return -(target_pi * torch.log(pred_policy + LOG_EPS)).sum(dim=1).mean()


def train_epoch(
    net,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train_value: bool = True,
) -> dict:
    net.train()
    total_policy_loss = 0.0
    total_value_loss  = 0.0
    n = 0

    for x, pi, z in loader:
        x  = x.to(device)
        pi = pi.to(device)
        z  = z.float().to(device)

        optimizer.zero_grad()
        pred_policy, pred_value = net(x)

        p_loss = soft_policy_loss(pred_policy, pi)
        v_loss = F.mse_loss(pred_value, z)

        if train_value:
            (p_loss + v_loss).backward()
        else:
            p_loss.backward()

        optimizer.step()

        bs = x.size(0)
        total_policy_loss += p_loss.item() * bs
        total_value_loss  += v_loss.item() * bs
        n += bs

    return {"policy_loss": total_policy_loss / n, "value_loss": total_value_loss / n}


@torch.no_grad()
def evaluate_supervised(net, val_loader: DataLoader, device: torch.device) -> dict:
    """Evaluate top-1 accuracy on hard-label supervised val set."""
    net.eval()
    top1 = top2 = n = 0
    for x, y_policy, _ in val_loader:
        x        = x.to(device)
        y_policy = y_policy.to(device)
        pred_policy, _ = net(x)
        top2_preds = pred_policy.topk(2, dim=1).indices
        top1 += (top2_preds[:, 0] == y_policy).sum().item()
        top2 += (top2_preds == y_policy.unsqueeze(1)).any(dim=1).sum().item()
        n += len(x)
    return {"top1": top1 / n, "top2": top2 / n}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-csv",       required=True,  type=Path)
    p.add_argument("--init-model",     type=Path, default=None, help="Checkpoint to fine-tune from (omit for random init)")
    p.add_argument("--random-init",    action="store_true", help="Start from random weights instead of loading a checkpoint")
    p.add_argument("--model-out",      required=True,  type=Path)
    p.add_argument("--metrics-out",    type=Path, default=None)
    p.add_argument("--supervised-csv", type=Path, default=None, help="Solver CSV for mixed training")
    p.add_argument("--val-csv",        type=Path, default=Path("data/UCI-Midgame-d30.val.csv"))
    p.add_argument("--filters",        type=int,  default=64)
    p.add_argument("--n-residuals",    type=int,  default=6)
    p.add_argument("--epochs",         type=int,  default=5)
    p.add_argument("--batch-size",     type=int,  default=512)
    p.add_argument("--lr",             type=float, default=0.001)
    p.add_argument("--weight-decay",   type=float, default=1e-4)
    p.add_argument("--freeze-value-head", action="store_true",
                   help="Freeze value head weights — only policy head + body update from self-play")
    p.add_argument("--sup-ratio",      type=float, default=3.0,
                   help="Supervised samples per self-play sample (default 3.0 = 75%% supervised)")
    p.add_argument("--seed",           type=int,  default=42)
    p.add_argument("--device",         default=None)
    return p.parse_args()


def main() -> int:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from src.train.network import Connect4Net, NetConfig
    from src.train.train_network import Connect4Dataset  # reuse for val evaluation

    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    # Load self-play data
    sp_df = pd.read_csv(args.data_csv, dtype={"sequence": "string"})
    print(f"Self-play rows: {len(sp_df)}")
    sp_ds = SelfPlayDataset(sp_df)

    # Build combined dataset with weighted sampling: sup_ratio supervised per self-play sample.
    # This ensures supervised data is used throughout ALL batches every epoch, not just the first ~39.
    if args.supervised_csv and args.supervised_csv.exists():
        sup_df = pd.read_csv(args.supervised_csv, dtype={"sequence": "string"}, low_memory=False)
        sup_ds = SupervisedDataset(sup_df)
        print(f"Supervised mix rows: {len(sup_df)}  sup_ratio={args.sup_ratio:.1f}x")

        from torch.utils.data import ConcatDataset, WeightedRandomSampler
        combined = ConcatDataset([sp_ds, sup_ds])
        # Normalize by dataset size so total weight of each split equals its
        # desired proportion regardless of how many rows each split has.
        # P(sp sample)  = 1 / (1 + sup_ratio)
        # P(sup sample) = sup_ratio / (1 + sup_ratio)
        sp_weight_each  = 1.0 / len(sp_ds)
        sup_weight_each = float(args.sup_ratio) / len(sup_ds)
        weights = ([sp_weight_each] * len(sp_ds)) + ([sup_weight_each] * len(sup_ds))
        # Total samples per epoch = len(sp_ds) * (1 + sup_ratio)
        n_samples = int(len(sp_ds) * (1 + args.sup_ratio))
        sampler = WeightedRandomSampler(weights, num_samples=n_samples, replacement=True)
        train_loader = DataLoader(combined, batch_size=args.batch_size, sampler=sampler,
                                  num_workers=2, pin_memory=True)
        sp_pct = 100.0 / (1 + args.sup_ratio)
        print(f"Combined samples/epoch: {n_samples}  ({sp_pct:.0f}% self-play / {100-sp_pct:.0f}% supervised)")
    else:
        train_loader = DataLoader(sp_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=2, pin_memory=True)

    # Optional val set for top-1 tracking
    val_loader = None
    if args.val_csv and args.val_csv.exists():
        val_df   = pd.read_csv(args.val_csv, dtype={"sequence": "string"}, low_memory=False)
        has_score = "solver_score" in val_df.columns
        val_ds   = Connect4Dataset(val_df, has_score)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
        print(f"Val rows: {len(val_df)}")

    # Load model
    cfg = NetConfig(filters=args.filters, n_residuals=args.n_residuals)
    net = Connect4Net(cfg).to(device)
    if args.random_init or args.init_model is None:
        print("Random weight initialization")
    else:
        ckpt = torch.load(args.init_model, map_location=device, weights_only=False)
        net.load_state_dict(ckpt.get("net_state_dict", ckpt))
        print(f"Loaded weights from {args.init_model}")

    if args.freeze_value_head:
        for p in net.value_head.parameters():
            p.requires_grad = False
        print("Value head frozen — only policy head + body will update")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, net.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )

    history = []
    best_top1 = 0.0
    best_state: dict | None = None
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        train_stats = train_epoch(net, train_loader, optimizer, device,
                                  train_value=not args.freeze_value_head)

        val_str = ""
        val_stats = {}
        if val_loader:
            val_stats = evaluate_supervised(net, val_loader, device)
            val_str = f"  val_top1={val_stats['top1']:.4f}  val_top2={val_stats['top2']:.4f}"
            if val_stats["top1"] > best_top1:
                best_top1 = val_stats["top1"]
                best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
                val_str += "  *best*"

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:>2}/{args.epochs}  "
            f"policy_loss={train_stats['policy_loss']:.4f}  "
            f"value_loss={train_stats['value_loss']:.4f}"
            f"{val_str}  elapsed={elapsed:.1f}s"
        )
        history.append({"epoch": epoch, "train": train_stats, "val": val_stats})

    # Save best checkpoint (fall back to final if no val set)
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    save_state = best_state if best_state is not None else net.state_dict()
    torch.save({"net_state_dict": save_state, "net_config": cfg.__dict__, "seed": args.seed}, args.model_out)
    tag = f"best val_top1={best_top1:.4f}" if best_state is not None else "final epoch"
    print(f"Saved: {args.model_out}  ({tag})")

    if args.metrics_out:
        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_out.write_text(json.dumps({
            "data_csv": str(args.data_csv),
            "init_model": str(args.init_model),
            "net_config": cfg.__dict__,
            "history": history,
        }, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
