"""Evaluate model win rate against three opponents:

  1. solver  — C++ alpha-beta solver (perfect play)
  2. prev    — previous iteration model + MCTS
  3. random  — uniform random legal moves

Plays N games per opponent, alternating who goes first each game.
Reports W/L/D from the evaluated model's perspective.

Usage (from 483-Connect4-ML/):
    python src/eval_winrate.py \\
        --model artifacts/models/selfplay_iter3.pth \\
        --solver /home/fenari/CPSC481/Project/Connect4-Bot/solver \\
        --prev-model artifacts/models/selfplay_iter2.pth \\
        --filters 64 --n-residuals 6 \\
        --games 50 --simulations 100
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROWS = 6
COLS = 7


# ---------------------------------------------------------------------------
# Board
# ---------------------------------------------------------------------------

class Board:
    def __init__(self) -> None:
        self.grid    = [[0] * COLS for _ in range(ROWS)]
        self.heights = [0] * COLS
        self.move_count = 0
        self.terminal   = False
        self.winner     = 0

    def can_play(self, col: int) -> bool:
        return 0 <= col < COLS and self.heights[col] < ROWS

    def legal_cols(self) -> list[int]:
        return [c for c in range(COLS) if self.can_play(c)]

    def play(self, col: int) -> None:
        row    = self.heights[col]
        player = 1 if self.move_count % 2 == 0 else 2
        self.grid[row][col] = player
        self.heights[col]  += 1
        self.move_count    += 1
        if self._is_win(row, col, player):
            self.terminal = True
            self.winner   = player
        elif self.move_count == ROWS * COLS:
            self.terminal = True

    def _is_win(self, row: int, col: int, player: int) -> bool:
        for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
            count = 1
            for sign in (-1, 1):
                cx, cy = col + sign * dx, row + sign * dy
                while 0 <= cx < COLS and 0 <= cy < ROWS and self.grid[cy][cx] == player:
                    count += 1
                    cx += sign * dx
                    cy += sign * dy
            if count >= 4:
                return True
        return False


# ---------------------------------------------------------------------------
# Opponents
# ---------------------------------------------------------------------------

class MCTSAgent:
    def __init__(self, model_path: str, filters: int, n_residuals: int,
                 simulations: int, device: str) -> None:
        sys.path.insert(0, str(Path(__file__).parent))
        from src.train.network import Connect4Net, NetConfig
        from src.fine_tune.mcts import MCTS

        cfg = NetConfig(filters=filters, n_residuals=n_residuals)
        net = Connect4Net(cfg)
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        net.load_state_dict(ckpt.get("net_state_dict", ckpt))
        self.mcts = MCTS(net, simulations=simulations, device=device,
                         dirichlet_epsilon=0.0)  # no exploration noise during eval

    def best_move(self, sequence: str) -> int:
        return self.mcts.best_move(sequence)

    def close(self) -> None:
        pass


class SolverAgent:
    def __init__(self, binary: str, weak: bool = True) -> None:
        binary = str(Path(binary).resolve())
        cwd    = str(Path(binary).parent.parent)  # run from repo root so opening book loads
        cmd = [binary]
        if weak:
            cmd.append("-w")
        self.proc = subprocess.Popen(
            cmd, cwd=cwd,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1,
        )

    def best_move(self, sequence: str) -> int:
        self.proc.stdin.write(f"{sequence}?\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline().strip()
        return int(line) - 1  # 0-based

    def close(self) -> None:
        try:
            self.proc.stdin.close()
            self.proc.terminate()
            self.proc.wait(timeout=2)
        except Exception:
            self.proc.kill()


class RandomAgent:
    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)

    def best_move(self, sequence: str) -> int:
        board = Board()
        for ch in sequence:
            board.play(int(ch) - 1)
        legal = board.legal_cols()
        return int(self.rng.choice(legal))

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Match
# ---------------------------------------------------------------------------

def play_match(model_agent, opponent_agent, n_games: int, rng) -> dict:
    """Play n_games, alternating who goes first. Returns W/L/D for model."""
    wins = losses = draws = 0

    for g in range(n_games):
        board    = Board()
        sequence = ""
        model_is_p1 = (g % 2 == 0)

        while not board.terminal:
            is_model_turn = (board.move_count % 2 == 0) == model_is_p1
            agent = model_agent if is_model_turn else opponent_agent
            col   = agent.best_move(sequence)

            if not board.can_play(col):
                legal = board.legal_cols()
                col   = int(rng.choice(legal))

            board.play(col)
            sequence += str(col + 1)

        model_player = 1 if model_is_p1 else 2
        if board.winner == 0:
            draws += 1
        elif board.winner == model_player:
            wins += 1
        else:
            losses += 1

        total = wins + losses + draws
        print(f"  Game {total:>3}/{n_games}  {'model' if board.winner == model_player else ('draw' if board.winner == 0 else 'opp  ')} wins  "
              f"W={wins} L={losses} D={draws}")

    total = wins + losses + draws
    return {
        "wins": wins, "losses": losses, "draws": draws,
        "win_rate": wins / total,
        "draw_rate": draws / total,
        "score": (wins + 0.5 * draws) / total,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model",        required=True,  help="Model to evaluate (.pth)")
    p.add_argument("--filters",      type=int, default=64)
    p.add_argument("--n-residuals",  type=int, default=6)
    p.add_argument("--simulations",  type=int, default=100)
    p.add_argument("--games",        type=int, default=50, help="Games per opponent")
    p.add_argument("--solver",       default=None, help="Path to C++ solver binary")
    p.add_argument("--prev-model",   default=None, help="Previous iteration .pth for head-to-head")
    p.add_argument("--prev-filters",     type=int, default=None)
    p.add_argument("--prev-residuals",   type=int, default=None)
    p.add_argument("--prev-simulations", type=int, default=None)
    p.add_argument("--device",       default=None)
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--out",          type=Path, default=None, help="Save JSON results here")
    return p.parse_args()


def main() -> int:
    args   = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    rng    = np.random.default_rng(args.seed)

    print(f"Evaluating: {args.model}")
    print(f"Device: {device}  Simulations: {args.simulations}  Games/opponent: {args.games}\n")

    model = MCTSAgent(args.model, args.filters, args.n_residuals, args.simulations, device)
    results = {}

    # --- vs random ---
    print("=== vs Random ===")
    rand_agent = RandomAgent(seed=args.seed)
    results["random"] = play_match(model, rand_agent, args.games, rng)
    rand_agent.close()
    r = results["random"]
    print(f"  -> W={r['wins']} L={r['losses']} D={r['draws']}  "
          f"win_rate={r['win_rate']:.3f}  score={r['score']:.3f}\n")

    # --- vs prev model ---
    if args.prev_model:
        print("=== vs Prev Model ===")
        pf = args.prev_filters   or args.filters
        pr = args.prev_residuals or args.n_residuals
        ps = args.prev_simulations or args.simulations
        prev = MCTSAgent(args.prev_model, pf, pr, ps, device)
        results["prev"] = play_match(model, prev, args.games, rng)
        prev.close()
        r = results["prev"]
        print(f"  -> W={r['wins']} L={r['losses']} D={r['draws']}  "
              f"win_rate={r['win_rate']:.3f}  score={r['score']:.3f}\n")

    # --- vs solver ---
    if args.solver:
        print("=== vs Solver ===")
        solver = SolverAgent(args.solver)
        results["solver"] = play_match(model, solver, args.games, rng)
        solver.close()
        r = results["solver"]
        print(f"  -> W={r['wins']} L={r['losses']} D={r['draws']}  "
              f"win_rate={r['win_rate']:.3f}  score={r['score']:.3f}\n")

    # --- summary ---
    print("=== Summary ===")
    for opp, r in results.items():
        print(f"  vs {opp:<10}  W={r['wins']:>3} L={r['losses']:>3} D={r['draws']:>3}  "
              f"score={r['score']:.3f}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": str(args.model),
            "prev_model": str(args.prev_model) if args.prev_model else None,
            "simulations": args.simulations,
            "games_per_opp": args.games,
            "seed": args.seed,
            "results": results,
        }
        args.out.write_text(json.dumps(payload, indent=2))
        print(f"\nSaved: {args.out}")

    model.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
