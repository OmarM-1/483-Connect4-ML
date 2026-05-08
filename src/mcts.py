"""Monte Carlo Tree Search for Connect4 using Connect4Net.

PUCT formula (AlphaZero style):
    U(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))

Usage:
    from network import Connect4Net, NetConfig, sequence_to_tensor
    from mcts import MCTS

    net = Connect4Net(...)
    # load weights ...
    mcts = MCTS(net, simulations=200)
    policy, value = mcts.search("4416")   # policy = (7,) visit-count distribution
    best_col = int(policy.argmax())       # 0-based column
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch

ROWS = 6
COLS = 7


# ---------------------------------------------------------------------------
# Board (self-contained, no external deps)
# ---------------------------------------------------------------------------

class Board:
    def __init__(self) -> None:
        self.grid    = [[0] * COLS for _ in range(ROWS)]
        self.heights = [0] * COLS
        self.move_count = 0
        self.terminal   = False
        self.winner     = 0  # 0=none/draw, 1=P1, 2=P2

    def clone(self) -> "Board":
        b = Board()
        b.grid       = [row[:] for row in self.grid]
        b.heights    = self.heights[:]
        b.move_count = self.move_count
        b.terminal   = self.terminal
        b.winner     = self.winner
        return b

    def can_play(self, col: int) -> bool:
        return 0 <= col < COLS and self.heights[col] < ROWS

    def legal_cols(self) -> list[int]:
        return [c for c in range(COLS) if self.can_play(c)]

    def play(self, col: int) -> None:
        """Play 0-based column."""
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

    def sequence_to_board(self, sequence: str) -> None:
        for ch in sequence:
            self.play(int(ch) - 1)


def board_from_sequence(sequence: str) -> Board:
    b = Board()
    b.sequence_to_board(sequence)
    return b


def board_to_tensor(board: Board) -> torch.Tensor:
    """Convert Board to (3, 6, 7) float32 tensor matching network.sequence_to_tensor."""
    p1 = np.zeros((ROWS, COLS), dtype=np.float32)
    p2 = np.zeros((ROWS, COLS), dtype=np.float32)
    for r in range(ROWS):
        for c in range(COLS):
            if board.grid[r][c] == 1:
                p1[r, c] = 1.0
            elif board.grid[r][c] == 2:
                p2[r, c] = 1.0
    to_move = float(board.move_count % 2 == 0)  # 1.0 = P1's turn
    tensor = np.stack([
        np.full((ROWS, COLS), to_move, dtype=np.float32),
        p1,
        p2,
    ])
    return torch.from_numpy(tensor)


# ---------------------------------------------------------------------------
# MCTS Node
# ---------------------------------------------------------------------------

class Node:
    __slots__ = ("board", "parent", "action", "children",
                 "prior", "visit_count", "value_sum", "expanded")

    def __init__(
        self,
        board: Board,
        parent: Optional["Node"] = None,
        action: Optional[int] = None,
        prior: float = 0.0,
    ) -> None:
        self.board       = board
        self.parent      = parent
        self.action      = action       # 0-based col that led here
        self.children: dict[int, Node] = {}
        self.prior       = prior
        self.visit_count = 0
        self.value_sum   = 0.0
        self.expanded    = False

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def ucb_score(self, parent_visits: int, c_puct: float) -> float:
        u = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        return self.q_value + u

    def best_child(self, c_puct: float) -> "Node":
        n = self.visit_count
        return max(self.children.values(), key=lambda ch: ch.ucb_score(n, c_puct))


# ---------------------------------------------------------------------------
# MCTS
# ---------------------------------------------------------------------------

class MCTS:
    def __init__(
        self,
        net,
        simulations: int = 200,
        c_puct: float = 1.25,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        device: Optional[str] = None,
    ) -> None:
        self.net         = net
        self.simulations = simulations
        self.c_puct      = c_puct
        self.dirichlet_alpha   = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.net.to(self.device)
        self.net.eval()

    @torch.no_grad()
    def _evaluate(self, board: Board) -> tuple[np.ndarray, float]:
        """Run network on board. Returns (policy_7, value_scalar)."""
        x = board_to_tensor(board).unsqueeze(0).to(self.device)
        policy, value = self.net(x)
        policy = policy.cpu().numpy()[0]   # (7,)
        value  = float(value.cpu().numpy()[0])

        # Mask illegal moves and renormalize.
        legal = np.array([board.can_play(c) for c in range(COLS)], dtype=np.float32)
        policy *= legal
        s = policy.sum()
        if s > 0:
            policy /= s
        else:
            policy = legal / legal.sum()

        return policy, value

    def _expand(self, node: Node) -> float:
        """Expand leaf node. Returns value from network (or terminal outcome)."""
        if node.board.terminal:
            # Value from the perspective of the node's *current player to move*.
            # If winner == previous player (who just moved), current player lost.
            if node.board.winner == 0:
                return 0.5  # draw
            # The player who just moved (opposite of current) won.
            return 0.0  # current player lost

        policy, value = self._evaluate(node.board)
        for col in node.board.legal_cols():
            child_board = node.board.clone()
            child_board.play(col)
            node.children[col] = Node(
                board=child_board,
                parent=node,
                action=col,
                prior=float(policy[col]),
            )
        node.expanded = True
        return value

    def _backpropagate(self, node: Node, value: float) -> None:
        """Propagate value up, flipping perspective at each level."""
        current = node
        while current is not None:
            current.visit_count += 1
            current.value_sum   += value
            value = 1.0 - value  # flip for opponent
            current = current.parent

    def _select(self, node: Node) -> Node:
        """Traverse tree selecting best UCB child until leaf."""
        while node.expanded and not node.board.terminal:
            node = node.best_child(self.c_puct)
        return node

    def search(self, sequence: str) -> tuple[np.ndarray, float]:
        """Run MCTS from position. Returns (policy_7, root_value).

        policy_7: visit-count distribution over 7 columns (sums to 1, illegal=0)
        root_value: estimated win probability for current player
        """
        root_board = board_from_sequence(sequence)
        root = Node(board=root_board)

        # Initial expansion + Dirichlet noise on root priors for exploration.
        value = self._expand(root)
        if root.children and self.dirichlet_epsilon > 0:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(root.children))
            for (col, child), n in zip(root.children.items(), noise):
                child.prior = (1 - self.dirichlet_epsilon) * child.prior + self.dirichlet_epsilon * n

        self._backpropagate(root, value)

        for _ in range(self.simulations - 1):
            leaf = self._select(root)
            v    = self._expand(leaf)
            self._backpropagate(leaf, v)

        # Policy = normalized visit counts.
        visits = np.zeros(COLS, dtype=np.float32)
        for col, child in root.children.items():
            visits[col] = child.visit_count
        total = visits.sum()
        policy = visits / total if total > 0 else visits

        return policy, root.q_value

    def best_move(self, sequence: str) -> int:
        """Return 0-based column. Compatible with MLPolicyClient interface."""
        policy, _ = self.search(sequence)
        return int(np.argmax(policy))


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from network import Connect4Net, NetConfig

    net = Connect4Net(NetConfig(filters=32, n_residuals=3))
    mcts = MCTS(net, simulations=50)

    for seq in ["", "4", "4416721"]:
        policy, value = mcts.search(seq)
        best = int(np.argmax(policy))
        print(f"seq={seq!r:10s}  best_col(0-based)={best}  value={value:.3f}  "
              f"policy={np.round(policy, 3)}")
