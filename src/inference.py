"""Model inference for Connect4 move selection.

Implements MLPolicyClient with the same best_move(sequence) -> int interface
as SolverClient in browser_bridge.py, so it can be dropped in as an alternative.

Feature contract (51-dim, must match train_connect4_baseline.py):
  42  board cells (side-to-move perspective, +1=self, -1=opp, 0=empty)
   7  column heights
   1  move_count
   1  to_move (+1 = P1, -1 = P2)

best_move() always returns a legal move (legal mask applied before selection).
"""

from __future__ import annotations

import math
import pickle
import random
from pathlib import Path

import numpy as np

ROWS = 6
COLS = 7
_EASY_TEMPERATURE = 3.0
_LOG_EPS = 1e-8


def _featurize(sequence: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (x_51, legal_mask_7). Sequence is 1-indexed column digits."""
    board = np.zeros((ROWS, COLS), dtype=np.int8)
    heights = np.zeros(COLS, dtype=np.int8)
    player = 1
    for ch in sequence:
        col = int(ch) - 1
        board[heights[col], col] = player
        heights[col] += 1
        player = 2 if player == 1 else 1

    perspective = 1 if player == 1 else -1
    signed = (board * perspective).astype(np.int8)

    x = np.concatenate([
        signed.reshape(-1).astype(np.float32),
        heights.astype(np.float32),
        np.array([len(sequence)], dtype=np.float32),
        np.array([float(perspective)], dtype=np.float32),
    ])
    legal = (heights < ROWS)
    return x, legal


def _softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max())
    return e / e.sum()


class MLPolicyClient:
    """Sklearn-model-based move selector compatible with the bot's SolverClient interface.

    Args:
        model_path: Path to a .pkl sklearn model trained with train_connect4_baseline.py.
        difficulty:  "hard"   — argmax (deterministic best)
                     "medium" — sample from model probability distribution
                     "easy"   — temperature-softened sampling (near-random)
    """

    DIFFICULTIES = {"hard", "medium", "easy"}

    def __init__(self, model_path: str | Path, difficulty: str = "hard") -> None:
        if difficulty not in self.DIFFICULTIES:
            raise ValueError(f"difficulty must be one of {self.DIFFICULTIES}, got {difficulty!r}")
        self.difficulty = difficulty

        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        try:
            import joblib
            self.model = joblib.load(path)
        except Exception:
            with open(path, "rb") as f:
                self.model = pickle.load(f)

        self._classes = np.asarray(self.model.classes_)  # 1-based [1..7]
        self._has_proba = hasattr(self.model, "predict_proba")
        self._has_df = hasattr(self.model, "decision_function")
        if not self._has_proba and not self._has_df:
            raise ValueError("Model must expose predict_proba or decision_function")

    def _scores(self, x: np.ndarray) -> np.ndarray:
        """Raw score vector aligned to self._classes."""
        if self._has_proba:
            return self.model.predict_proba(x)[0]
        scores = self.model.decision_function(x)
        if scores.ndim == 2:
            return scores[0]
        return scores

    def best_move(self, sequence: str) -> int:
        """Return 0-based column index for best legal move given move sequence."""
        x, legal = _featurize(sequence)
        scores = self._scores(x.reshape(1, -1))

        # Build full 7-column array aligned to column index (0-based).
        # model.classes_ are 1-based column labels.
        col_scores = np.full(COLS, -np.inf)
        for i, cls in enumerate(self._classes):
            col = int(cls) - 1
            if 0 <= col < COLS:
                col_scores[col] = scores[i]

        # Zero out illegal columns.
        col_scores[~legal] = -np.inf

        legal_cols = np.flatnonzero(legal)
        if len(legal_cols) == 0:
            raise RuntimeError(f"No legal moves for sequence {sequence!r}")

        if self.difficulty == "hard":
            return int(np.argmax(col_scores))

        # Build probability distribution over legal columns only.
        legal_scores = col_scores[legal_cols]

        if self.difficulty == "medium":
            probs = _softmax(legal_scores)
        else:  # easy
            log_p = np.log(np.maximum(_softmax(legal_scores), _LOG_EPS))
            probs = _softmax(log_p / _EASY_TEMPERATURE)

        chosen = random.choices(legal_cols.tolist(), weights=probs.tolist(), k=1)[0]
        return int(chosen)

    def close(self) -> None:
        pass


if __name__ == "__main__":
    import sys

    model_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else str(Path(__file__).parent.parent / "artifacts" / "models" / "connect4_random_forest.pkl")
    )
    difficulty = sys.argv[2] if len(sys.argv) > 2 else "hard"

    client = MLPolicyClient(model_path, difficulty=difficulty)
    print(f"Model loaded: {model_path}")
    print(f"Difficulty:   {difficulty}")

    test_sequences = ["", "4", "44", "4416721", "1234567"]
    for seq in test_sequences:
        move = client.best_move(seq)
        print(f"  sequence={seq!r:12s}  best_move (0-based col) = {move}  (1-based: {move+1})")
