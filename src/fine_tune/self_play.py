"""Self-play game generator for AlphaZero-style training.

Plays Connect4Net+MCTS against itself from move 0, recording every position
as a (sequence, mcts_policy, outcome) training example.

CSV schema:
    sequence    — move history (1-indexed digit string)
    mcts_policy — 7 space-separated floats (visit-count distribution, sums to 1)
    outcome     — float in [0,1] from current player's perspective
                  1.0=win, 0.5=draw, 0.0=loss

Mirror augmentation doubles data: each position is also saved with the board
mirrored left-right (column c → 8-c, policy flipped).

Usage (from 483-Connect4-ML/):
    python src/self_play.py \\
        --model artifacts/models/connect4_net.pth \\
        --filters 64 --n-residuals 6 \\
        --games 300 --simulations 100 \\
        --out data/selfplay_iter0.csv
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

ROWS = 6
COLS = 7
TEMP_THRESHOLD = 20  # moves below this index use temperature sampling
TEMPERATURE    = 1.0


def _mirror_sequence(seq: str) -> str:
    return "".join(str(8 - int(c)) for c in seq)


def _mirror_policy(policy: np.ndarray) -> np.ndarray:
    return policy[::-1].copy()


def generate_game(mcts, rng: np.random.Generator) -> list[tuple[str, np.ndarray, int]]:
    """Play one self-play game. Returns trajectory before outcome is known.

    Each entry: (sequence_at_that_point, mcts_policy_7, player_who_moved)
    player_who_moved: 1 or 2 (used to assign outcome retrospectively)
    """
    from mcts import Board

    board = Board()
    sequence = ""
    trajectory: list[tuple[str, np.ndarray, int]] = []

    while not board.terminal:
        policy, _ = mcts.search(sequence)

        move_idx = len(sequence)
        if move_idx < TEMP_THRESHOLD:
            # Temperature sampling — explore openings
            powered = policy ** (1.0 / TEMPERATURE)
            s = powered.sum()
            probs = powered / s if s > 0 else policy
        else:
            # Greedy — best move
            probs = np.zeros(COLS, dtype=np.float32)
            probs[int(np.argmax(policy))] = 1.0

        col = int(rng.choice(COLS, p=probs))
        player = 1 if board.move_count % 2 == 0 else 2
        trajectory.append((sequence, policy.copy(), player))

        board.play(col)
        sequence += str(col + 1)  # 1-indexed

    winner = board.winner  # 0=draw, 1=P1, 2=P2
    return trajectory, winner


def trajectory_to_rows(
    trajectory: list[tuple[str, np.ndarray, int]],
    winner: int,
    mirror: bool = True,
) -> list[dict]:
    rows = []
    for seq, policy, player in trajectory:
        if winner == 0:
            outcome = 0.5
        elif winner == player:
            outcome = 1.0
        else:
            outcome = 0.0

        policy_str = " ".join(f"{v:.6f}" for v in policy)
        rows.append({"sequence": seq, "mcts_policy": policy_str, "outcome": outcome})

        if mirror:
            m_seq    = _mirror_sequence(seq)
            m_policy = _mirror_policy(policy)
            m_str    = " ".join(f"{v:.6f}" for v in m_policy)
            rows.append({"sequence": m_seq, "mcts_policy": m_str, "outcome": outcome})

    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model",       default=None,   help="Path to .pth checkpoint (omit for random init)")
    p.add_argument("--random-model", action="store_true", help="Use randomly initialized network")
    p.add_argument("--filters",     type=int, default=64)
    p.add_argument("--n-residuals", type=int, default=6)
    p.add_argument("--games",       type=int, default=300, help="Number of games to generate")
    p.add_argument("--simulations", type=int, default=100, help="MCTS simulations per move")
    p.add_argument("--out",         required=True,  help="Output CSV path")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--device",      default=None)
    p.add_argument("--no-mirror",   action="store_true", help="Disable mirror augmentation")
    p.add_argument("--append",      action="store_true", help="Append to existing CSV")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from src.train.network import Connect4Net, NetConfig
    from mcts import MCTS

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    cfg = NetConfig(filters=args.filters, n_residuals=args.n_residuals)
    net = Connect4Net(cfg)
    if not args.random_model and args.model and args.model != "RANDOM":
        ckpt = torch.load(args.model, map_location=device, weights_only=False)
        net.load_state_dict(ckpt.get("net_state_dict", ckpt))
        model_label = args.model
    else:
        model_label = "random-init"

    mcts = MCTS(net, simulations=args.simulations, device=device_str)
    print(f"Model:  {model_label}  ({args.filters}f/{args.n_residuals}r)")
    print(f"Games:  {args.games}  Sims/move: {args.simulations}  Device: {device_str}")
    print(f"Output: {args.out}  Mirror: {not args.no_mirror}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append and out_path.exists() else "w"

    fieldnames = ["sequence", "mcts_policy", "outcome"]
    total_rows = 0
    t0 = time.time()

    with out_path.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if mode == "w":
            writer.writeheader()

        for g in range(1, args.games + 1):
            traj, winner = generate_game(mcts, rng)
            rows = trajectory_to_rows(traj, winner, mirror=not args.no_mirror)
            writer.writerows(rows)
            f.flush()
            total_rows += len(rows)

            outcome_str = f"P{winner} wins" if winner else "draw"
            elapsed = time.time() - t0
            print(f"Game {g:>4}/{args.games}  moves={len(traj):>2}  {outcome_str:<10}  "
                  f"rows={total_rows}  elapsed={elapsed:.1f}s")

    print(f"\nDone. {total_rows} rows written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
