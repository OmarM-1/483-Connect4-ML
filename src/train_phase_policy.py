#!/usr/bin/env python3
"""Train a phase-aware Connect4 policy model.

Supports:
- Supervised phase policy training from labeled CSV data.
- Lightweight self-play RL training (REINFORCE) using only NumPy.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_torch_spec = importlib.util.find_spec("torch")
if _torch_spec is not None:
    torch = importlib.import_module("torch")
else:
    torch = None

from preprocess import (
    board_from_current_player_perspective,
    get_legal_moves,
    sequence_to_board,
)

ROWS = 6
COLS = 7

PHASE_WINDOWS = {
    "early": (4, 11),
    "mid": (12, 28),
    "late": (29, 41),
    "all": (4, 41),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=Path("data/UCI-Midgame-d30.train.csv"),
        help="Training CSV path",
    )
    parser.add_argument(
        "--val-csv",
        type=Path,
        default=Path("data/UCI-Midgame-d30.val.csv"),
        help="Validation CSV path",
    )
    parser.add_argument(
        "--training-mode",
        choices=["supervised", "rl"],
        default="supervised",
        help="Training mode: supervised from labels or self-play RL",
    )
    parser.add_argument(
        "--phase",
        choices=["early", "mid", "late", "all", "custom"],
        default="mid",
        help="Phase window to train on",
    )
    parser.add_argument(
        "--move-min",
        type=int,
        default=None,
        help="Custom minimum move count (used with --phase custom)",
    )
    parser.add_argument(
        "--move-max",
        type=int,
        default=None,
        help="Custom maximum move count (used with --phase custom)",
    )
    parser.add_argument(
        "--canonicalize-mirror",
        action="store_true",
        help="Mirror boards to a canonical orientation before feature extraction",
    )
    parser.add_argument(
        "--max-rows-train",
        type=int,
        default=None,
        help="Optional cap on train rows after phase filtering",
    )
    parser.add_argument(
        "--max-rows-val",
        type=int,
        default=None,
        help="Optional cap on validation rows after phase filtering",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for sampling and model training",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1e-4,
        help="Regularization strength for the SGD policy model",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=2000,
        help="Maximum optimizer iterations for the policy model",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-3,
        help="Optimization tolerance for the policy model",
    )
    parser.add_argument(
        "--rl-episodes",
        type=int,
        default=2000,
        help="Number of self-play episodes for RL mode",
    )
    parser.add_argument(
        "--rl-batch-episodes",
        type=int,
        default=64,
        help="Episodes per policy update batch in RL mode",
    )
    parser.add_argument(
        "--rl-lr",
        type=float,
        default=0.03,
        help="Policy gradient learning rate in RL mode",
    )
    parser.add_argument(
        "--rl-entropy-coef",
        type=float,
        default=0.01,
        help="Entropy regularization strength in RL mode",
    )
    parser.add_argument(
        "--rl-entropy-coef-final",
        type=float,
        default=None,
        help="Optional final entropy coefficient for linear decay schedule",
    )
    parser.add_argument(
        "--rl-epsilon",
        type=float,
        default=0.05,
        help="Epsilon-greedy exploration probability in RL mode",
    )
    parser.add_argument(
        "--rl-epsilon-final",
        type=float,
        default=None,
        help="Optional final epsilon for linear decay schedule",
    )
    parser.add_argument(
        "--rl-algo",
        choices=["reinforce", "actor-critic"],
        default="reinforce",
        help="RL algorithm for torch backend",
    )
    parser.add_argument(
        "--rl-gamma",
        type=float,
        default=0.99,
        help="Discount factor for actor-critic returns",
    )
    parser.add_argument(
        "--rl-value-loss-coef",
        type=float,
        default=0.5,
        help="Value loss weight for actor-critic",
    )
    parser.add_argument(
        "--rl-seed-weights",
        action="store_true",
        help="Initialize RL policy weights from supervised labels (same phase window)",
    )
    parser.add_argument(
        "--rl-seed-steps",
        type=int,
        default=120,
        help="Number of supervised warm-start update steps when --rl-seed-weights is enabled",
    )
    parser.add_argument(
        "--rl-seed-batch-size",
        type=int,
        default=512,
        help="Batch size used for supervised warm-start updates",
    )
    parser.add_argument(
        "--rl-seed-lr",
        type=float,
        default=None,
        help="Optional learning rate override for supervised warm-start (defaults to RL lr)",
    )
    parser.add_argument(
        "--rl-eval-interval-episodes",
        type=int,
        default=100,
        help="Evaluate RL checkpoint on validation set every N episodes (0 disables periodic eval)",
    )
    parser.add_argument(
        "--rl-hard-replay-fraction",
        type=float,
        default=0.0,
        help="Fraction of validation misses to replay each eval interval (0 disables)",
    )
    parser.add_argument(
        "--rl-hard-replay-steps",
        type=int,
        default=1,
        help="Number of replay updates to run per eval interval when hard replay is enabled",
    )
    parser.add_argument(
        "--rl-backend",
        choices=["auto", "torch", "numpy"],
        default="auto",
        help="RL implementation backend (auto picks torch when available, else numpy)",
    )
    parser.add_argument(
        "--rl-device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device preference for torch backend (auto chooses cuda when available)",
    )
    parser.add_argument(
        "--rl-policy-arch",
        choices=["linear", "mlp"],
        default="linear",
        help="Torch RL policy architecture",
    )
    parser.add_argument(
        "--rl-hidden-dim",
        type=int,
        default=128,
        help="Hidden size for --rl-policy-arch mlp",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=Path("artifacts/models/connect4_phase_policy.pkl"),
        help="Where to save the trained policy model",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=Path("artifacts/metrics/connect4_phase_policy_metrics.json"),
        help="Where to save phase policy metrics JSON",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=Path("artifacts/reports/connect4_phase_policy_report.txt"),
        help="Where to save the human-readable report",
    )
    parser.add_argument(
        "--experiment-log-csv",
        type=Path,
        default=Path("artifacts/metrics/connect4_phase_policy_experiment_log.csv"),
        help="CSV file where a summary row is appended after each run",
    )
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def mirror_sequence(sequence: str) -> str:
    return "".join(str(8 - int(ch)) for ch in sequence)


def mirror_best_move(best_move: int) -> int:
    return 8 - best_move


def canonicalize_sequence_and_label(sequence: str, best_move: int) -> tuple[str, int]:
    mirrored = mirror_sequence(sequence)
    if mirrored < sequence:
        return mirrored, mirror_best_move(best_move)
    return sequence, best_move


def resolve_window(args: argparse.Namespace) -> tuple[int, int]:
    if args.phase == "custom":
        if args.move_min is None or args.move_max is None:
            raise ValueError("--phase custom requires --move-min and --move-max")
        if args.move_min < 1 or args.move_max > 41 or args.move_min > args.move_max:
            raise ValueError("invalid custom phase window; expected 1 <= min <= max <= 41")
        return args.move_min, args.move_max

    move_min, move_max = PHASE_WINDOWS[args.phase]
    if args.move_min is not None:
        move_min = args.move_min
    if args.move_max is not None:
        move_max = args.move_max

    if move_min < 1 or move_max > 41 or move_min > move_max:
        raise ValueError("invalid phase window; expected 1 <= min <= max <= 41")
    return move_min, move_max


def filter_phase_rows(df: pd.DataFrame, move_min: int, move_max: int) -> pd.DataFrame:
    frame = df.copy()
    if "move_count" in frame.columns:
        move_counts = frame["move_count"].astype(int)
    else:
        move_counts = frame["sequence"].astype(str).str.len().astype(int)
    mask = (move_counts >= move_min) & (move_counts <= move_max)
    return frame.loc[mask].reset_index(drop=True)


def maybe_sample_rows(df: pd.DataFrame, max_rows: int | None, random_state: int) -> pd.DataFrame:
    if max_rows is None or max_rows <= 0 or len(df) <= max_rows:
        return df.reset_index(drop=True)
    return df.sample(n=max_rows, random_state=random_state).reset_index(drop=True)


def build_features(
    df: pd.DataFrame,
    canonicalize_mirror: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    if "sequence" not in df.columns or "best_move" not in df.columns:
        raise ValueError("CSV must include sequence and best_move columns")

    features: list[np.ndarray] = []
    labels: list[int] = []
    legal_masks: list[np.ndarray] = []
    sequences: list[str] = []

    for sequence_value, best_move_value in zip(df["sequence"].astype(str), df["best_move"].astype(int)):
        sequence = sequence_value.strip()
        best_move = int(best_move_value)
        if canonicalize_mirror:
            sequence, best_move = canonicalize_sequence_and_label(sequence, best_move)

        board = sequence_to_board(sequence)
        current = board_from_current_player_perspective(sequence)
        legal = get_legal_moves(board).astype(bool)

        features.append(current.reshape(-1).astype(np.float32))
        labels.append(best_move - 1)
        legal_masks.append(legal)
        sequences.append(sequence)

    X = np.stack(features)
    y = np.asarray(labels, dtype=np.int64)
    legal_mask = np.stack(legal_masks)
    return X, y, legal_mask, sequences


def load_phase_frame(
    csv_path: Path,
    move_min: int,
    move_max: int,
    max_rows: int | None,
    random_state: int,
) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    frame = pd.read_csv(csv_path, dtype={"sequence": "string"}, low_memory=False)
    required = {"sequence", "best_move"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")

    frame = filter_phase_rows(frame, move_min, move_max)
    frame = maybe_sample_rows(frame, max_rows=max_rows, random_state=random_state)
    return frame


def top_k_accuracy(scores: np.ndarray, y_true: np.ndarray, classes: np.ndarray, k: int) -> float:
    if scores.ndim != 2:
        raise ValueError("scores must be a 2D array")
    if scores.shape[1] == 0:
        raise ValueError("scores must have at least one class")
    k = max(1, min(k, scores.shape[1]))
    top_idx = np.argpartition(scores, -k, axis=1)[:, -k:]
    top_labels = classes[top_idx]
    hits = np.any(top_labels == y_true[:, None], axis=1)
    return float(np.mean(hits))


def masked_scores(scores: np.ndarray, legal_mask: np.ndarray) -> np.ndarray:
    masked = np.array(scores, copy=True)
    masked[~legal_mask] = -np.inf
    return masked


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    denom = np.sum(exp)
    if denom <= 0:
        return np.full_like(logits, 1.0 / len(logits), dtype=np.float64)
    return exp / denom


def linear_schedule(start_value: float, end_value: float, step_idx: int, total_steps: int) -> float:
    if total_steps <= 1:
        return float(end_value)
    progress = min(1.0, max(0.0, float(step_idx) / float(total_steps - 1)))
    return float(start_value + (end_value - start_value) * progress)


def legal_moves_from_heights(heights: np.ndarray) -> np.ndarray:
    return heights < ROWS


def drop_piece(board: np.ndarray, heights: np.ndarray, col: int, player: int) -> int:
    row = int(heights[col])
    board[row, col] = player
    heights[col] += 1
    return row


def check_four(board: np.ndarray, row: int, col: int, player: int) -> bool:
    for dr, dc in ((1, 0), (0, 1), (1, 1), (1, -1)):
        count = 1
        rr, cc = row + dr, col + dc
        while 0 <= rr < ROWS and 0 <= cc < COLS and board[rr, cc] == player:
            count += 1
            rr += dr
            cc += dc
        rr, cc = row - dr, col - dc
        while 0 <= rr < ROWS and 0 <= cc < COLS and board[rr, cc] == player:
            count += 1
            rr -= dr
            cc -= dc
        if count >= 4:
            return True
    return False


def sequence_to_state(sequence: str) -> tuple[np.ndarray, np.ndarray, int]:
    board = sequence_to_board(sequence).astype(np.int8)
    heights = np.count_nonzero(board, axis=0).astype(np.int8)
    to_move = 1 if len(sequence) % 2 == 0 else -1
    return board, heights, to_move


def resolve_rl_backend(backend: str) -> str:
    if backend == "auto":
        return "torch" if torch is not None else "numpy"
    if backend == "torch" and torch is None:
        raise ValueError("PyTorch is not installed; use --rl-backend numpy or install torch")
    return backend


def resolve_torch_device(device_pref: str) -> str:
    if torch is None:
        return "cpu"
    if device_pref == "cpu":
        return "cpu"
    if device_pref == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("--rl-device cuda requested but CUDA is not available")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_torch_policy_network(policy_arch: str, hidden_dim: int):
    if torch is None:
        raise ValueError("PyTorch is not installed")
    if policy_arch == "linear":
        return torch.nn.Linear(ROWS * COLS, COLS, bias=True)
    if policy_arch == "mlp":
        if hidden_dim <= 0:
            raise ValueError("--rl-hidden-dim must be > 0 for mlp policy")
        return torch.nn.Sequential(
            torch.nn.Linear(ROWS * COLS, hidden_dim, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, COLS, bias=True),
        )
    raise ValueError(f"Unsupported torch policy architecture: {policy_arch}")


def build_torch_value_network(policy_arch: str, hidden_dim: int):
    if torch is None:
        raise ValueError("PyTorch is not installed")
    if policy_arch == "linear":
        return torch.nn.Linear(ROWS * COLS, 1, bias=True)
    if policy_arch == "mlp":
        if hidden_dim <= 0:
            raise ValueError("--rl-hidden-dim must be > 0 for mlp value network")
        return torch.nn.Sequential(
            torch.nn.Linear(ROWS * COLS, hidden_dim, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1, bias=True),
        )
    raise ValueError(f"Unsupported torch value architecture: {policy_arch}")


def policy_logits(policy: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    return x @ policy["W"] + policy["b"]


def masked_policy_probs(policy: dict[str, np.ndarray], x: np.ndarray, legal_mask: np.ndarray) -> np.ndarray:
    logits = policy_logits(policy, x)
    masked = np.where(legal_mask, logits, -1e9)
    return softmax(masked)


def sample_policy_move(
    policy: dict[str, np.ndarray],
    x: np.ndarray,
    legal_mask: np.ndarray,
    rng: np.random.Generator,
    epsilon: float,
) -> tuple[int, np.ndarray]:
    legal_idx = np.flatnonzero(legal_mask)
    if len(legal_idx) == 0:
        raise ValueError("No legal moves available")

    probs = masked_policy_probs(policy, x, legal_mask)
    if rng.random() < epsilon:
        action = int(rng.choice(legal_idx))
    else:
        action = int(rng.choice(np.arange(COLS), p=probs))
    return action, probs


def run_self_play_episode(
    policy: dict[str, np.ndarray],
    rng: np.random.Generator,
    start_sequence: str,
    epsilon: float,
) -> tuple[list[tuple[np.ndarray, int, np.ndarray, int]], int]:
    board, heights, player = sequence_to_state(start_sequence)
    trajectory: list[tuple[np.ndarray, int, np.ndarray, int]] = []

    while True:
        legal = legal_moves_from_heights(heights)
        if not np.any(legal):
            return trajectory, 0

        x = (board * player).reshape(-1).astype(np.float32)
        action, probs = sample_policy_move(policy, x, legal, rng, epsilon=epsilon)
        trajectory.append((x, action, probs, player))

        row = drop_piece(board, heights, action, player)
        if check_four(board, row, action, player):
            return trajectory, player
        if np.all(heights >= ROWS):
            return trajectory, 0
        player *= -1


def apply_reinforce_update(
    policy: dict[str, np.ndarray],
    batch_episodes: list[tuple[list[tuple[np.ndarray, int, np.ndarray, int]], int]],
    lr: float,
    entropy_coef: float,
) -> None:
    grad_w = np.zeros_like(policy["W"], dtype=np.float64)
    grad_b = np.zeros_like(policy["b"], dtype=np.float64)
    steps = 0

    for trajectory, winner in batch_episodes:
        for x, action, probs, acting_player in trajectory:
            reward = 0.0
            if winner != 0:
                reward = 1.0 if winner == acting_player else -1.0

            one_hot = np.zeros(COLS, dtype=np.float64)
            one_hot[action] = 1.0
            reinforce_dir = one_hot - probs
            entropy_dir = -(np.log(np.maximum(probs, 1e-9)) + 1.0)
            direction = (reward * reinforce_dir) + (entropy_coef * entropy_dir)

            grad_w += np.outer(x.astype(np.float64), direction)
            grad_b += direction
            steps += 1

    if steps == 0:
        return

    scale = lr / float(steps)
    policy["W"] += scale * grad_w.astype(np.float32)
    policy["b"] += scale * grad_b.astype(np.float32)


def seed_policy_from_labels(
    policy: dict[str, np.ndarray],
    X_train: np.ndarray,
    y_train: np.ndarray,
    legal_mask_train: np.ndarray,
    rng: np.random.Generator,
    steps: int = 200,
    lr: float = 0.05,
    batch_size: int = 512,
) -> None:
    if len(X_train) == 0:
        return

    n = len(X_train)
    for _ in range(max(1, steps)):
        take = min(batch_size, n)
        idx = rng.integers(0, n, size=take)
        grad_w = np.zeros_like(policy["W"], dtype=np.float64)
        grad_b = np.zeros_like(policy["b"], dtype=np.float64)
        used = 0

        for i in idx:
            x = X_train[i]
            y = int(y_train[i])
            legal = legal_mask_train[i].astype(bool)
            if not legal[y]:
                continue
            probs = masked_policy_probs(policy, x, legal)
            one_hot = np.zeros(COLS, dtype=np.float64)
            one_hot[y] = 1.0
            direction = one_hot - probs
            grad_w += np.outer(x.astype(np.float64), direction)
            grad_b += direction
            used += 1

        if used > 0:
            scale = lr / float(used)
            policy["W"] += scale * grad_w.astype(np.float32)
            policy["b"] += scale * grad_b.astype(np.float32)


def train_rl_policy(
    train_df: pd.DataFrame,
    random_state: int,
    episodes: int,
    batch_episodes: int,
    lr: float,
    entropy_coef: float,
    entropy_coef_final: float,
    epsilon: float,
    epsilon_final: float,
    seed_weights: bool,
    canonicalize_mirror: bool,
    backend: str,
    device_pref: str,
    policy_arch: str,
    hidden_dim: int,
    algo: str,
    gamma: float,
    value_loss_coef: float,
    seed_steps: int,
    seed_batch_size: int,
    seed_lr: float | None,
    X_val: np.ndarray,
    y_val: np.ndarray,
    legal_mask_val: np.ndarray,
    eval_interval_episodes: int,
    hard_replay_fraction: float,
    hard_replay_steps: int,
) -> tuple[dict[str, Any], dict[str, float]]:
    resolved_backend = resolve_rl_backend(backend)
    if resolved_backend == "torch":
        return train_rl_policy_torch(
            train_df=train_df,
            random_state=random_state,
            episodes=episodes,
            batch_episodes=batch_episodes,
            lr=lr,
            entropy_coef=entropy_coef,
            entropy_coef_final=entropy_coef_final,
            epsilon=epsilon,
            epsilon_final=epsilon_final,
            seed_weights=seed_weights,
            canonicalize_mirror=canonicalize_mirror,
            device_pref=device_pref,
            policy_arch=policy_arch,
            hidden_dim=hidden_dim,
            algo=algo,
            gamma=gamma,
            value_loss_coef=value_loss_coef,
            seed_steps=seed_steps,
            seed_batch_size=seed_batch_size,
            seed_lr=seed_lr,
            X_val=X_val,
            y_val=y_val,
            legal_mask_val=legal_mask_val,
            eval_interval_episodes=eval_interval_episodes,
            hard_replay_fraction=hard_replay_fraction,
            hard_replay_steps=hard_replay_steps,
        )

    return train_rl_policy_numpy(
        train_df=train_df,
        random_state=random_state,
        episodes=episodes,
        batch_episodes=batch_episodes,
        lr=lr,
        entropy_coef=entropy_coef,
        entropy_coef_final=entropy_coef_final,
        epsilon=epsilon,
        epsilon_final=epsilon_final,
        seed_weights=seed_weights,
        canonicalize_mirror=canonicalize_mirror,
        algo=algo,
        seed_steps=seed_steps,
        seed_batch_size=seed_batch_size,
        seed_lr=seed_lr,
        X_val=X_val,
        y_val=y_val,
        legal_mask_val=legal_mask_val,
        eval_interval_episodes=eval_interval_episodes,
        hard_replay_fraction=hard_replay_fraction,
        hard_replay_steps=hard_replay_steps,
    )


def supervised_update_numpy(
    policy: dict[str, np.ndarray],
    X_batch: np.ndarray,
    y_batch: np.ndarray,
    legal_batch: np.ndarray,
    lr: float,
) -> None:
    grad_w = np.zeros_like(policy["W"], dtype=np.float64)
    grad_b = np.zeros_like(policy["b"], dtype=np.float64)
    used = 0
    for x, y, legal in zip(X_batch, y_batch, legal_batch):
        yy = int(y)
        if not legal[yy]:
            continue
        probs = masked_policy_probs(policy, x, legal.astype(bool))
        one_hot = np.zeros(COLS, dtype=np.float64)
        one_hot[yy] = 1.0
        direction = one_hot - probs
        grad_w += np.outer(x.astype(np.float64), direction)
        grad_b += direction
        used += 1
    if used == 0:
        return
    scale = lr / float(used)
    policy["W"] += scale * grad_w.astype(np.float32)
    policy["b"] += scale * grad_b.astype(np.float32)


def train_rl_policy_numpy(
    train_df: pd.DataFrame,
    random_state: int,
    episodes: int,
    batch_episodes: int,
    lr: float,
    entropy_coef: float,
    entropy_coef_final: float,
    epsilon: float,
    epsilon_final: float,
    seed_weights: bool,
    canonicalize_mirror: bool,
    algo: str,
    seed_steps: int,
    seed_batch_size: int,
    seed_lr: float | None,
    X_val: np.ndarray,
    y_val: np.ndarray,
    legal_mask_val: np.ndarray,
    eval_interval_episodes: int,
    hard_replay_fraction: float,
    hard_replay_steps: int,
) -> tuple[dict[str, Any], dict[str, float]]:
    rng = np.random.default_rng(random_state)
    policy: dict[str, np.ndarray] = {
        "backend": "numpy",
        "W": rng.normal(0.0, 0.01, size=(ROWS * COLS, COLS)).astype(np.float32),
        "b": np.zeros(COLS, dtype=np.float32),
    }

    if seed_weights and len(train_df) > 0:
        X_seed, y_seed, legal_seed, _ = build_features(train_df, canonicalize_mirror=canonicalize_mirror)
        seed_policy_from_labels(
            policy,
            X_seed,
            y_seed,
            legal_seed,
            rng=rng,
            steps=max(1, seed_steps),
            lr=seed_lr if seed_lr is not None else min(0.1, max(0.005, lr)),
            batch_size=max(1, seed_batch_size),
        )

    start_sequences = train_df["sequence"].astype(str).str.strip().tolist()
    if not start_sequences:
        start_sequences = [""]

    total_steps = 0
    wins = 0
    losses = 0
    draws = 0
    best_val_top1 = float("-inf")
    best_episode = 0
    best_w = np.array(policy["W"], copy=True)
    best_b = np.array(policy["b"], copy=True)
    hard_replay_updates = 0

    pending: list[tuple[list[tuple[np.ndarray, int, np.ndarray, int]], int]] = []
    total_episodes = max(1, episodes)
    for episode_idx in range(1, total_episodes + 1):
        curr_epsilon = linear_schedule(epsilon, epsilon_final, episode_idx - 1, total_episodes)
        curr_entropy = linear_schedule(entropy_coef, entropy_coef_final, episode_idx - 1, total_episodes)
        start_sequence = str(rng.choice(start_sequences))
        trajectory, winner = run_self_play_episode(policy, rng, start_sequence, epsilon=curr_epsilon)
        total_steps += len(trajectory)

        if winner == 1:
            wins += 1
        elif winner == -1:
            losses += 1
        else:
            draws += 1

        pending.append((trajectory, winner))
        if len(pending) >= max(1, batch_episodes):
            apply_reinforce_update(policy, pending, lr=lr, entropy_coef=curr_entropy)
            pending.clear()

        should_eval = (
            eval_interval_episodes > 0
            and (episode_idx % eval_interval_episodes == 0 or episode_idx == total_episodes)
        )
        if should_eval:
            if pending:
                apply_reinforce_update(policy, pending, lr=lr, entropy_coef=curr_entropy)
                pending.clear()

            y_pred, _ = rl_policy_scores(policy, X_val, legal_mask_val)
            val_top1 = float(accuracy_score(y_val, y_pred))
            if val_top1 > best_val_top1:
                best_val_top1 = val_top1
                best_episode = episode_idx
                best_w = np.array(policy["W"], copy=True)
                best_b = np.array(policy["b"], copy=True)

            if hard_replay_fraction > 0.0 and hard_replay_steps > 0:
                miss_idx = np.flatnonzero(y_pred != y_val)
                if len(miss_idx) > 0:
                    take = max(1, int(len(miss_idx) * hard_replay_fraction))
                    take = min(take, len(miss_idx))
                    for _ in range(hard_replay_steps):
                        replay_idx = rng.choice(miss_idx, size=take, replace=False)
                        supervised_update_numpy(
                            policy,
                            X_val[replay_idx],
                            y_val[replay_idx],
                            legal_mask_val[replay_idx],
                            lr=lr,
                        )
                        hard_replay_updates += 1

    if pending:
        apply_reinforce_update(policy, pending, lr=lr, entropy_coef=entropy_coef_final)

    if best_episode > 0:
        policy["W"] = best_w
        policy["b"] = best_b

    training_stats = {
        "backend": "numpy",
        "device": "cpu",
        "algo": algo,
        "epsilon_start": epsilon,
        "epsilon_final": epsilon_final,
        "entropy_coef_start": entropy_coef,
        "entropy_coef_final": entropy_coef_final,
        "eval_interval_episodes": eval_interval_episodes,
        "hard_replay_fraction": hard_replay_fraction,
        "hard_replay_steps": hard_replay_steps,
        "hard_replay_updates": hard_replay_updates,
        "best_val_top1": best_val_top1 if best_episode > 0 else None,
        "best_checkpoint_episode": best_episode if best_episode > 0 else None,
        "episodes": float(max(1, episodes)),
        "avg_episode_steps": float(total_steps / max(1, episodes)),
        "win_rate_player1": float(wins / max(1, episodes)),
        "loss_rate_player1": float(losses / max(1, episodes)),
        "draw_rate": float(draws / max(1, episodes)),
    }
    return policy, training_stats


def masked_policy_probs_torch(policy: dict[str, Any], x: np.ndarray, legal_mask: np.ndarray) -> np.ndarray:
    if torch is None:
        raise ValueError("PyTorch is not installed")
    device = policy["device"]
    with torch.no_grad():
        x_t = torch.from_numpy(x).to(device=device, dtype=torch.float32)
        logits = policy["network"](x_t)
        legal_t = torch.from_numpy(legal_mask.astype(np.bool_)).to(device=device)
        masked_logits = logits.masked_fill(~legal_t, -1e9)
        probs = torch.softmax(masked_logits, dim=-1)
    return probs.detach().cpu().numpy().astype(np.float32)


def run_self_play_episode_torch(
    policy: dict[str, Any],
    rng: np.random.Generator,
    start_sequence: str,
    epsilon: float,
) -> tuple[list[tuple[np.ndarray, int, np.ndarray, int]], int]:
    board, heights, player = sequence_to_state(start_sequence)
    trajectory: list[tuple[np.ndarray, int, np.ndarray, int]] = []

    while True:
        legal = legal_moves_from_heights(heights)
        if not np.any(legal):
            return trajectory, 0

        x = (board * player).reshape(-1).astype(np.float32)
        legal_idx = np.flatnonzero(legal)
        probs = masked_policy_probs_torch(policy, x, legal)
        if rng.random() < epsilon:
            action = int(rng.choice(legal_idx))
        else:
            action = int(rng.choice(np.arange(COLS), p=probs))
        trajectory.append((x, action, legal.astype(bool), player))

        row = drop_piece(board, heights, action, player)
        if check_four(board, row, action, player):
            return trajectory, player
        if np.all(heights >= ROWS):
            return trajectory, 0
        player *= -1


def apply_reinforce_update_torch(
    policy: dict[str, Any],
    batch_episodes: list[tuple[list[tuple[np.ndarray, int, np.ndarray, int]], int]],
    entropy_coef: float,
) -> None:
    if torch is None:
        raise ValueError("PyTorch is not installed")

    network = policy["network"]
    optimizer = policy["optimizer"]
    device = policy["device"]

    losses: list[Any] = []
    optimizer.zero_grad(set_to_none=True)

    for trajectory, winner in batch_episodes:
        for x, action, legal, acting_player in trajectory:
            reward = 0.0
            if winner != 0:
                reward = 1.0 if winner == acting_player else -1.0

            x_t = torch.from_numpy(x).to(device=device, dtype=torch.float32)
            logits = network(x_t)
            legal_t = torch.from_numpy(legal.astype(np.bool_)).to(device=device)
            masked_logits = logits.masked_fill(~legal_t, -1e9)
            log_probs = torch.log_softmax(masked_logits, dim=-1)
            probs = torch.softmax(masked_logits, dim=-1)
            logp = log_probs[action]
            entropy = -(probs * log_probs).sum()
            loss = (-reward * logp) - (entropy_coef * entropy)
            losses.append(loss)

    if not losses:
        return

    total_loss = torch.stack(losses).mean()
    total_loss.backward()
    optimizer.step()


def apply_actor_critic_update_torch(
    policy: dict[str, Any],
    batch_episodes: list[tuple[list[tuple[np.ndarray, int, np.ndarray, int]], int]],
    entropy_coef: float,
    gamma: float,
    value_loss_coef: float,
) -> None:
    if torch is None:
        raise ValueError("PyTorch is not installed")

    network = policy["network"]
    value_network = policy["value_network"]
    optimizer = policy["optimizer"]
    device = policy["device"]

    losses: list[Any] = []
    optimizer.zero_grad(set_to_none=True)

    for trajectory, winner in batch_episodes:
        if not trajectory:
            continue

        horizon = len(trajectory)
        for step_idx, (x, action, legal, acting_player) in enumerate(trajectory):
            if winner == 0:
                terminal_reward = 0.0
            else:
                terminal_reward = 1.0 if winner == acting_player else -1.0
            discounted_return = terminal_reward * (gamma ** (horizon - 1 - step_idx))

            x_t = torch.from_numpy(x).to(device=device, dtype=torch.float32)

            logits = network(x_t)
            legal_t = torch.from_numpy(legal.astype(np.bool_)).to(device=device)
            masked_logits = logits.masked_fill(~legal_t, -1e9)

            log_probs = torch.log_softmax(masked_logits, dim=-1)
            probs = torch.softmax(masked_logits, dim=-1)
            logp = log_probs[action]
            entropy = -(probs * log_probs).sum()

            value = value_network(x_t).squeeze(-1)
            target = torch.tensor(discounted_return, dtype=torch.float32, device=device)
            advantage = (target - value).detach()

            policy_loss = -advantage * logp
            value_loss = torch.nn.functional.mse_loss(value, target)
            loss = policy_loss + (value_loss_coef * value_loss) - (entropy_coef * entropy)
            losses.append(loss)

    if not losses:
        return

    total_loss = torch.stack(losses).mean()
    total_loss.backward()
    optimizer.step()


def supervised_update_torch(
    policy: dict[str, Any],
    optimizer,
    X_batch: np.ndarray,
    y_batch: np.ndarray,
    legal_batch: np.ndarray,
) -> None:
    if torch is None or len(X_batch) == 0:
        return
    network = policy["network"]
    device = policy["device"]
    x_t = torch.from_numpy(X_batch).to(device=device, dtype=torch.float32)
    y_t = torch.from_numpy(y_batch).to(device=device, dtype=torch.long)
    legal_t = torch.from_numpy(legal_batch.astype(np.bool_)).to(device=device)
    logits = network(x_t)
    masked_logits = logits.masked_fill(~legal_t, -1e9)
    log_probs = torch.log_softmax(masked_logits, dim=-1)
    nll = -log_probs[torch.arange(len(X_batch), device=device), y_t]
    valid = legal_t[torch.arange(len(X_batch), device=device), y_t]
    if valid.any():
        loss = nll[valid].mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


def seed_policy_from_labels_torch(
    policy: dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    legal_mask_train: np.ndarray,
    rng: np.random.Generator,
    optimizer,
    steps: int = 200,
    batch_size: int = 512,
    lr_override: float | None = None,
) -> None:
    if torch is None or len(X_train) == 0:
        return

    network = policy["network"]
    device = policy["device"]
    n = len(X_train)

    old_lrs = [group["lr"] for group in optimizer.param_groups]
    if lr_override is not None:
        for group in optimizer.param_groups:
            group["lr"] = lr_override

    for _ in range(max(1, steps)):
        take = min(batch_size, n)
        idx = rng.integers(0, n, size=take)

        x_batch = torch.from_numpy(X_train[idx]).to(device=device, dtype=torch.float32)
        y_batch = torch.from_numpy(y_train[idx]).to(device=device, dtype=torch.long)
        legal_batch = torch.from_numpy(legal_mask_train[idx].astype(np.bool_)).to(device=device)

        logits = network(x_batch)
        masked_logits = logits.masked_fill(~legal_batch, -1e9)
        log_probs = torch.log_softmax(masked_logits, dim=-1)
        nll = -log_probs[torch.arange(take, device=device), y_batch]
        valid = legal_batch[torch.arange(take, device=device), y_batch]
        if valid.any():
            loss = nll[valid].mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    for group, old_lr in zip(optimizer.param_groups, old_lrs):
        group["lr"] = old_lr


def train_rl_policy_torch(
    train_df: pd.DataFrame,
    random_state: int,
    episodes: int,
    batch_episodes: int,
    lr: float,
    entropy_coef: float,
    entropy_coef_final: float,
    epsilon: float,
    epsilon_final: float,
    seed_weights: bool,
    canonicalize_mirror: bool,
    device_pref: str,
    policy_arch: str,
    hidden_dim: int,
    algo: str,
    gamma: float,
    value_loss_coef: float,
    seed_steps: int,
    seed_batch_size: int,
    seed_lr: float | None,
    X_val: np.ndarray,
    y_val: np.ndarray,
    legal_mask_val: np.ndarray,
    eval_interval_episodes: int,
    hard_replay_fraction: float,
    hard_replay_steps: int,
) -> tuple[dict[str, Any], dict[str, float]]:
    if torch is None:
        raise ValueError("PyTorch is not installed; cannot use torch RL backend")

    rng = np.random.default_rng(random_state)
    torch.manual_seed(random_state)
    device_name = resolve_torch_device(device_pref)
    device = torch.device(device_name)

    network = build_torch_policy_network(policy_arch, hidden_dim).to(device)
    value_network = build_torch_value_network(policy_arch, hidden_dim).to(device)
    with torch.no_grad():
        for module in list(network.modules()) + list(value_network.modules()):
            if isinstance(module, torch.nn.Linear):
                module.weight.normal_(mean=0.0, std=0.01)
                module.bias.zero_()
    optimizer = torch.optim.Adam(
        list(network.parameters()) + list(value_network.parameters()),
        lr=lr,
    )

    policy: dict[str, Any] = {
        "backend": "torch",
        "device": device_name,
        "policy_arch": policy_arch,
        "hidden_dim": int(hidden_dim),
        "algo": algo,
        "network": network,
        "value_network": value_network,
        "optimizer": optimizer,
    }

    if seed_weights and len(train_df) > 0:
        X_seed, y_seed, legal_seed, _ = build_features(train_df, canonicalize_mirror=canonicalize_mirror)
        seed_policy_from_labels_torch(
            policy,
            X_seed,
            y_seed,
            legal_seed,
            rng=rng,
            optimizer=optimizer,
            steps=max(1, seed_steps),
            batch_size=max(1, seed_batch_size),
            lr_override=seed_lr,
        )

    start_sequences = train_df["sequence"].astype(str).str.strip().tolist()
    if not start_sequences:
        start_sequences = [""]

    total_steps = 0
    wins = 0
    losses = 0
    draws = 0
    best_val_top1 = float("-inf")
    best_episode = 0
    best_policy_state = {k: v.detach().cpu().clone() for k, v in network.state_dict().items()}
    best_value_state = {k: v.detach().cpu().clone() for k, v in value_network.state_dict().items()}
    hard_replay_updates = 0

    pending: list[tuple[list[tuple[np.ndarray, int, np.ndarray, int]], int]] = []
    total_episodes = max(1, episodes)
    for episode_idx in range(1, total_episodes + 1):
        curr_epsilon = linear_schedule(epsilon, epsilon_final, episode_idx - 1, total_episodes)
        curr_entropy = linear_schedule(entropy_coef, entropy_coef_final, episode_idx - 1, total_episodes)
        start_sequence = str(rng.choice(start_sequences))
        trajectory, winner = run_self_play_episode_torch(policy, rng, start_sequence, epsilon=curr_epsilon)
        total_steps += len(trajectory)

        if winner == 1:
            wins += 1
        elif winner == -1:
            losses += 1
        else:
            draws += 1

        pending.append((trajectory, winner))
        if len(pending) >= max(1, batch_episodes):
            if algo == "actor-critic":
                apply_actor_critic_update_torch(
                    policy,
                    pending,
                    entropy_coef=curr_entropy,
                    gamma=gamma,
                    value_loss_coef=value_loss_coef,
                )
            else:
                apply_reinforce_update_torch(policy, pending, entropy_coef=curr_entropy)
            pending.clear()

        should_eval = (
            eval_interval_episodes > 0
            and (episode_idx % eval_interval_episodes == 0 or episode_idx == total_episodes)
        )
        if should_eval:
            if pending:
                if algo == "actor-critic":
                    apply_actor_critic_update_torch(
                        policy,
                        pending,
                        entropy_coef=curr_entropy,
                        gamma=gamma,
                        value_loss_coef=value_loss_coef,
                    )
                else:
                    apply_reinforce_update_torch(policy, pending, entropy_coef=curr_entropy)
                pending.clear()

            y_pred, _ = rl_policy_scores(policy, X_val, legal_mask_val)
            val_top1 = float(accuracy_score(y_val, y_pred))
            if val_top1 > best_val_top1:
                best_val_top1 = val_top1
                best_episode = episode_idx
                best_policy_state = {k: v.detach().cpu().clone() for k, v in network.state_dict().items()}
                best_value_state = {k: v.detach().cpu().clone() for k, v in value_network.state_dict().items()}

            if hard_replay_fraction > 0.0 and hard_replay_steps > 0:
                miss_idx = np.flatnonzero(y_pred != y_val)
                if len(miss_idx) > 0:
                    take = max(1, int(len(miss_idx) * hard_replay_fraction))
                    take = min(take, len(miss_idx))
                    for _ in range(hard_replay_steps):
                        replay_idx = rng.choice(miss_idx, size=take, replace=False)
                        supervised_update_torch(
                            policy,
                            optimizer,
                            X_val[replay_idx],
                            y_val[replay_idx],
                            legal_mask_val[replay_idx],
                        )
                        hard_replay_updates += 1

    if pending:
        if algo == "actor-critic":
            apply_actor_critic_update_torch(
                policy,
                pending,
                entropy_coef=entropy_coef_final,
                gamma=gamma,
                value_loss_coef=value_loss_coef,
            )
        else:
            apply_reinforce_update_torch(policy, pending, entropy_coef=entropy_coef_final)

    if best_episode > 0:
        network.load_state_dict(best_policy_state)
        value_network.load_state_dict(best_value_state)

    policy["state_dict"] = {k: v.detach().cpu() for k, v in network.state_dict().items()}
    policy["value_state_dict"] = {k: v.detach().cpu() for k, v in value_network.state_dict().items()}
    policy["optimizer_state_dict"] = optimizer.state_dict()

    training_stats = {
        "backend": "torch",
        "device": device_name,
        "policy_arch": policy_arch,
        "hidden_dim": int(hidden_dim),
        "algo": algo,
        "gamma": gamma,
        "value_loss_coef": value_loss_coef,
        "epsilon_start": epsilon,
        "epsilon_final": epsilon_final,
        "entropy_coef_start": entropy_coef,
        "entropy_coef_final": entropy_coef_final,
        "eval_interval_episodes": eval_interval_episodes,
        "hard_replay_fraction": hard_replay_fraction,
        "hard_replay_steps": hard_replay_steps,
        "hard_replay_updates": hard_replay_updates,
        "best_val_top1": best_val_top1 if best_episode > 0 else None,
        "best_checkpoint_episode": best_episode if best_episode > 0 else None,
        "seed_steps": int(seed_steps),
        "seed_batch_size": int(seed_batch_size),
        "seed_lr": seed_lr,
        "episodes": float(max(1, episodes)),
        "avg_episode_steps": float(total_steps / max(1, episodes)),
        "win_rate_player1": float(wins / max(1, episodes)),
        "loss_rate_player1": float(losses / max(1, episodes)),
        "draw_rate": float(draws / max(1, episodes)),
    }
    return policy, training_stats


def rl_policy_scores(
    policy: dict[str, Any],
    X: np.ndarray,
    legal_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    backend = policy.get("backend", "numpy")
    scores = np.zeros((len(X), COLS), dtype=np.float32)
    y_pred = np.zeros(len(X), dtype=np.int64)

    if backend == "torch":
        for i in range(len(X)):
            probs = masked_policy_probs_torch(policy, X[i], legal_mask[i].astype(bool))
            scores[i] = probs
            y_pred[i] = int(np.argmax(probs))
        return y_pred, scores

    for i in range(len(X)):
        probs = masked_policy_probs(policy, X[i], legal_mask[i].astype(bool))
        scores[i] = probs.astype(np.float32)
        y_pred[i] = int(np.argmax(probs))
    return y_pred, scores


def infer_scores(
    model_kind: str,
    model: Any,
    X: np.ndarray,
    legal_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if model_kind == "supervised":
        y_pred = model.predict(X)
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(X)
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(X)
        else:
            raise ValueError("Model must expose predict_proba or decision_function")
        if scores.ndim == 1:
            scores = np.column_stack([-scores, scores])
        classes = np.asarray(model.classes_)
        return y_pred, scores, classes

    if model_kind == "rl":
        classes = np.arange(COLS, dtype=np.int64)
        y_pred, scores = rl_policy_scores(model, X, legal_mask)
        return y_pred, scores, classes

    raise ValueError(f"Unsupported model kind: {model_kind}")


def evaluate_policy(
    model,
    X_val: np.ndarray,
    y_val: np.ndarray,
    legal_mask: np.ndarray,
    model_kind: str,
) -> tuple[dict, str]:
    y_pred, scores, classes = infer_scores(model_kind, model, X_val, legal_mask)
    raw_top1 = float(accuracy_score(y_val, y_pred))
    raw_macro_f1 = float(f1_score(y_val, y_pred, average="macro", zero_division=0))
    raw_top2 = top_k_accuracy(scores, y_val, classes, k=2)

    masked = masked_scores(scores, legal_mask)
    masked_pred_idx = np.argmax(masked, axis=1)
    masked_pred = classes[masked_pred_idx]
    masked_top1 = float(accuracy_score(y_val, masked_pred))
    masked_macro_f1 = float(f1_score(y_val, masked_pred, average="macro", zero_division=0))
    masked_top2 = top_k_accuracy(masked, y_val, classes, k=2)

    illegal_rate = float(np.mean(~legal_mask[np.arange(len(y_pred)), y_pred]))
    masked_illegal_rate = float(np.mean(~legal_mask[np.arange(len(masked_pred)), masked_pred]))

    cm = confusion_matrix(y_val, y_pred, labels=classes)
    report = classification_report(
        y_val,
        y_pred,
        labels=classes,
        digits=4,
        zero_division=0,
    )

    metrics = {
        "raw": {
            "top1_accuracy": raw_top1,
            "top2_accuracy": raw_top2,
            "macro_f1": raw_macro_f1,
            "illegal_move_rate": illegal_rate,
        },
        "masked": {
            "top1_accuracy": masked_top1,
            "top2_accuracy": masked_top2,
            "macro_f1": masked_macro_f1,
            "illegal_move_rate": masked_illegal_rate,
        },
        "confusion_matrix": cm.tolist(),
    }
    return metrics, report


def append_experiment_log_row(
    log_csv: Path,
    args: argparse.Namespace,
    move_min: int,
    move_max: int,
    summary: dict[str, Any],
) -> None:
    model_metrics = next(iter(summary["models"].values()))
    rl_stats = summary.get("rl", {}).get("training_stats", {})

    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "train_csv": summary.get("train_csv"),
        "val_csv": summary.get("val_csv"),
        "training_mode": summary.get("training_mode"),
        "phase": summary.get("phase"),
        "move_min": move_min,
        "move_max": move_max,
        "train_rows": summary.get("train_rows"),
        "val_rows": summary.get("val_rows"),
        "random_state": summary.get("random_state"),
        "train_seconds": summary.get("train_seconds"),
        "eval_seconds": summary.get("eval_seconds"),
        "avg_inference_ms_per_row": summary.get("avg_inference_ms_per_row"),
        "raw_top1": model_metrics["raw"]["top1_accuracy"],
        "raw_top2": model_metrics["raw"]["top2_accuracy"],
        "raw_macro_f1": model_metrics["raw"]["macro_f1"],
        "masked_top1": model_metrics["masked"]["top1_accuracy"],
        "masked_top2": model_metrics["masked"]["top2_accuracy"],
        "masked_macro_f1": model_metrics["masked"]["macro_f1"],
        "raw_illegal_rate": model_metrics["raw"]["illegal_move_rate"],
        "masked_illegal_rate": model_metrics["masked"]["illegal_move_rate"],
        "rl_backend_requested": summary.get("rl", {}).get("backend"),
        "rl_device_requested": summary.get("rl", {}).get("device_pref"),
        "rl_policy_arch_requested": summary.get("rl", {}).get("policy_arch"),
        "rl_hidden_dim_requested": summary.get("rl", {}).get("hidden_dim"),
        "rl_backend_resolved": rl_stats.get("backend"),
        "rl_device_resolved": rl_stats.get("device"),
        "rl_policy_arch_resolved": rl_stats.get("policy_arch"),
        "rl_hidden_dim_resolved": rl_stats.get("hidden_dim"),
        "rl_episodes": summary.get("rl", {}).get("episodes"),
        "rl_batch_episodes": summary.get("rl", {}).get("batch_episodes"),
        "rl_lr": summary.get("rl", {}).get("lr"),
        "rl_entropy_coef": summary.get("rl", {}).get("entropy_coef"),
        "rl_epsilon": summary.get("rl", {}).get("epsilon"),
    }

    ensure_parent(log_csv)
    write_header = not log_csv.exists() or log_csv.stat().st_size == 0
    with log_csv.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> int:
    args = parse_args()
    move_min, move_max = resolve_window(args)

    train_df = load_phase_frame(
        args.train_csv,
        move_min=move_min,
        move_max=move_max,
        max_rows=args.max_rows_train,
        random_state=args.random_state,
    )
    val_df = load_phase_frame(
        args.val_csv,
        move_min=move_min,
        move_max=move_max,
        max_rows=args.max_rows_val,
        random_state=args.random_state,
    )

    X_train, y_train, legal_train_mask, _ = build_features(
        train_df,
        canonicalize_mirror=args.canonicalize_mirror,
    )
    X_val, y_val, legal_mask, _ = build_features(val_df, canonicalize_mirror=args.canonicalize_mirror)

    model_kind = args.training_mode
    rl_training_stats: dict[str, float] = {}

    start = time.perf_counter()
    if model_kind == "supervised":
        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    SGDClassifier(
                        loss="log_loss",
                        alpha=args.alpha,
                        max_iter=args.max_iter,
                        tol=args.tol,
                        random_state=args.random_state,
                        class_weight="balanced",
                        average=True,
                    ),
                ),
            ]
        )
        model.fit(X_train, y_train)
    else:
        model, rl_training_stats = train_rl_policy(
            train_df=train_df,
            random_state=args.random_state,
            episodes=args.rl_episodes,
            batch_episodes=args.rl_batch_episodes,
            lr=args.rl_lr,
            entropy_coef=args.rl_entropy_coef,
            entropy_coef_final=args.rl_entropy_coef_final
            if args.rl_entropy_coef_final is not None
            else args.rl_entropy_coef,
            epsilon=args.rl_epsilon,
            epsilon_final=args.rl_epsilon_final if args.rl_epsilon_final is not None else args.rl_epsilon,
            seed_weights=args.rl_seed_weights,
            canonicalize_mirror=args.canonicalize_mirror,
            backend=args.rl_backend,
            device_pref=args.rl_device,
            policy_arch=args.rl_policy_arch,
            hidden_dim=args.rl_hidden_dim,
            algo=args.rl_algo,
            gamma=args.rl_gamma,
            value_loss_coef=args.rl_value_loss_coef,
            seed_steps=args.rl_seed_steps,
            seed_batch_size=args.rl_seed_batch_size,
            seed_lr=args.rl_seed_lr,
            X_val=X_val,
            y_val=y_val,
            legal_mask_val=legal_mask,
            eval_interval_episodes=args.rl_eval_interval_episodes,
            hard_replay_fraction=args.rl_hard_replay_fraction,
            hard_replay_steps=args.rl_hard_replay_steps,
        )
    train_seconds = time.perf_counter() - start

    eval_start = time.perf_counter()
    metrics, report = evaluate_policy(
        model,
        X_val,
        y_val,
        legal_mask,
        model_kind=model_kind,
    )
    eval_seconds = time.perf_counter() - eval_start
    avg_inference_ms = (eval_seconds / max(1, len(X_val))) * 1000.0

    summary = {
        "train_csv": str(args.train_csv),
        "val_csv": str(args.val_csv),
        "phase": args.phase,
        "training_mode": model_kind,
        "move_window": [move_min, move_max],
        "canonicalize_mirror": bool(args.canonicalize_mirror),
        "random_state": args.random_state,
        "alpha": args.alpha,
        "max_iter": args.max_iter,
        "tol": args.tol,
        "rl": {
            "episodes": args.rl_episodes,
            "batch_episodes": args.rl_batch_episodes,
            "lr": args.rl_lr,
            "entropy_coef": args.rl_entropy_coef,
            "entropy_coef_final": args.rl_entropy_coef_final,
            "epsilon": args.rl_epsilon,
            "epsilon_final": args.rl_epsilon_final,
            "algo": args.rl_algo,
            "gamma": args.rl_gamma,
            "value_loss_coef": args.rl_value_loss_coef,
            "seed_weights": bool(args.rl_seed_weights),
            "seed_steps": args.rl_seed_steps,
            "seed_batch_size": args.rl_seed_batch_size,
            "seed_lr": args.rl_seed_lr,
            "eval_interval_episodes": args.rl_eval_interval_episodes,
            "hard_replay_fraction": args.rl_hard_replay_fraction,
            "hard_replay_steps": args.rl_hard_replay_steps,
            "backend": args.rl_backend,
            "device_pref": args.rl_device,
            "policy_arch": args.rl_policy_arch,
            "hidden_dim": args.rl_hidden_dim,
            "training_stats": rl_training_stats,
        },
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "features": int(X_train.shape[1]),
        "train_seconds": train_seconds,
        "eval_seconds": eval_seconds,
        "avg_inference_ms_per_row": avg_inference_ms,
        "models": {f"{model_kind}_phase_policy": metrics},
    }

    ensure_parent(args.model_out)
    ensure_parent(args.metrics_out)
    ensure_parent(args.report_out)

    with args.model_out.open("wb") as f:
        pickle.dump(model, f)

    with args.metrics_out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with args.report_out.open("w", encoding="utf-8") as f:
        f.write(f"Phase policy model ({args.phase})\n")
        f.write(f"Training mode: {model_kind}\n")
        f.write(f"Window: {move_min}-{move_max}\n")
        f.write(f"Canonicalize mirror: {args.canonicalize_mirror}\n")
        f.write(f"Train rows: {len(train_df)}\n")
        f.write(f"Val rows: {len(val_df)}\n")
        f.write(f"Features: {X_train.shape[1]}\n")
        f.write(f"Train seconds: {train_seconds:.3f}\n")
        f.write(f"Eval seconds: {eval_seconds:.3f}\n")
        f.write(f"Avg inference ms/row: {avg_inference_ms:.6f}\n\n")
        if model_kind == "rl":
            f.write("=== rl training ===\n")
            f.write(f"Episodes: {args.rl_episodes}\n")
            f.write(f"Batch episodes: {args.rl_batch_episodes}\n")
            f.write(f"LR: {args.rl_lr}\n")
            f.write(f"Entropy coef: {args.rl_entropy_coef}\n")
            f.write(f"Entropy coef final: {args.rl_entropy_coef_final}\n")
            f.write(f"Epsilon: {args.rl_epsilon}\n")
            f.write(f"Epsilon final: {args.rl_epsilon_final}\n")
            f.write(f"Algorithm: {args.rl_algo}\n")
            f.write(f"Gamma: {args.rl_gamma}\n")
            f.write(f"Value loss coef: {args.rl_value_loss_coef}\n")
            f.write(f"Seed weights: {args.rl_seed_weights}\n")
            f.write(f"Seed steps: {args.rl_seed_steps}\n")
            f.write(f"Seed batch size: {args.rl_seed_batch_size}\n")
            f.write(f"Seed lr override: {args.rl_seed_lr}\n")
            f.write(f"Eval interval episodes: {args.rl_eval_interval_episodes}\n")
            f.write(f"Hard replay fraction: {args.rl_hard_replay_fraction}\n")
            f.write(f"Hard replay steps: {args.rl_hard_replay_steps}\n")
            f.write(f"Backend (requested): {args.rl_backend}\n")
            f.write(f"Device (requested): {args.rl_device}\n")
            f.write(f"Policy arch (requested): {args.rl_policy_arch}\n")
            f.write(f"Hidden dim (requested): {args.rl_hidden_dim}\n")
            if rl_training_stats:
                f.write(f"Backend (resolved): {rl_training_stats.get('backend', 'unknown')}\n")
                f.write(f"Device (resolved): {rl_training_stats.get('device', 'unknown')}\n")
                f.write(f"Policy arch (resolved): {rl_training_stats.get('policy_arch', 'unknown')}\n")
                if 'hidden_dim' in rl_training_stats:
                    f.write(f"Hidden dim (resolved): {rl_training_stats.get('hidden_dim', 'unknown')}\n")
                f.write(f"Algorithm (resolved): {rl_training_stats.get('algo', 'unknown')}\n")
                if 'gamma' in rl_training_stats:
                    f.write(f"Gamma (resolved): {rl_training_stats.get('gamma', 'unknown')}\n")
                if 'value_loss_coef' in rl_training_stats:
                    f.write(f"Value loss coef (resolved): {rl_training_stats.get('value_loss_coef', 'unknown')}\n")
                if 'best_val_top1' in rl_training_stats:
                    f.write(f"Best val top1: {rl_training_stats.get('best_val_top1', 'unknown')}\n")
                if 'best_checkpoint_episode' in rl_training_stats:
                    f.write(f"Best checkpoint episode: {rl_training_stats.get('best_checkpoint_episode', 'unknown')}\n")
                if 'hard_replay_updates' in rl_training_stats:
                    f.write(f"Hard replay updates: {rl_training_stats.get('hard_replay_updates', 'unknown')}\n")
            if rl_training_stats:
                f.write(f"Avg episode steps: {rl_training_stats['avg_episode_steps']:.4f}\n")
                f.write(f"P1 win/loss/draw: {rl_training_stats['win_rate_player1']:.4f}/")
                f.write(f"{rl_training_stats['loss_rate_player1']:.4f}/")
                f.write(f"{rl_training_stats['draw_rate']:.4f}\n")
            f.write("\n")
        f.write("=== raw ===\n")
        f.write(f"Top-1: {metrics['raw']['top1_accuracy']:.4f}\n")
        f.write(f"Top-2: {metrics['raw']['top2_accuracy']:.4f}\n")
        f.write(f"Macro-F1: {metrics['raw']['macro_f1']:.4f}\n")
        f.write(f"Illegal move rate: {metrics['raw']['illegal_move_rate']:.4f}\n\n")
        f.write("=== masked ===\n")
        f.write(f"Top-1: {metrics['masked']['top1_accuracy']:.4f}\n")
        f.write(f"Top-2: {metrics['masked']['top2_accuracy']:.4f}\n")
        f.write(f"Macro-F1: {metrics['masked']['macro_f1']:.4f}\n")
        f.write(f"Illegal move rate: {metrics['masked']['illegal_move_rate']:.4f}\n\n")
        f.write("=== classification report (raw predictions) ===\n")
        f.write(report)

    append_experiment_log_row(
        log_csv=args.experiment_log_csv,
        args=args,
        move_min=move_min,
        move_max=move_max,
        summary=summary,
    )

    print("Phase policy training complete")
    print(f"Phase: {args.phase} | window: {move_min}-{move_max}")
    print(f"Training mode: {model_kind}")
    print(f"Train/Val rows: {len(train_df)}/{len(val_df)}")
    if model_kind == "rl" and rl_training_stats:
        print(
            "RL avg episode steps / P1 win-lost-draw: "
            f"{rl_training_stats['avg_episode_steps']:.3f} / "
            f"{rl_training_stats['win_rate_player1']:.3f}-"
            f"{rl_training_stats['loss_rate_player1']:.3f}-"
            f"{rl_training_stats['draw_rate']:.3f}"
        )
        print(
            "RL backend/device (resolved): "
            f"{rl_training_stats.get('backend', 'unknown')}/{rl_training_stats.get('device', 'unknown')}"
        )
        print(
            "RL policy architecture (resolved): "
            f"{rl_training_stats.get('policy_arch', 'unknown')}"
        )
        print(
            "RL algorithm (resolved): "
            f"{rl_training_stats.get('algo', args.rl_algo)}"
        )
    print(
        "Raw Top-1/Top-2: "
        f"{metrics['raw']['top1_accuracy']:.4f}/{metrics['raw']['top2_accuracy']:.4f}"
    )
    print(
        "Masked Top-1/Top-2: "
        f"{metrics['masked']['top1_accuracy']:.4f}/{metrics['masked']['top2_accuracy']:.4f}"
    )
    print(f"Illegal move rate (raw/masked): {metrics['raw']['illegal_move_rate']:.4f}/{metrics['masked']['illegal_move_rate']:.4f}")
    print(f"Model saved: {args.model_out}")
    print(f"Metrics saved: {args.metrics_out}")
    print(f"Report saved: {args.report_out}")
    print(f"Experiment row appended: {args.experiment_log_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
