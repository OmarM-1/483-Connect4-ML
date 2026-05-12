"""AlphaZero-style neural network for Connect4.

Architecture adapted from willis-richard/connect4 (MIT License).
Input:  3 × 6 × 7 tensor  (to_move channel, P1 pieces, P2 pieces)
Output: policy (7-dim softmax), value (scalar tanh → [0,1])
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

ROWS = 6
COLS = 7
AREA = ROWS * COLS  # 42


# ---------------------------------------------------------------------------
# Board featurization  (sequence str → 3×6×7 float tensor)
# ---------------------------------------------------------------------------

def sequence_to_tensor(sequence: str) -> torch.Tensor:
    """Convert move sequence to (3, 6, 7) float32 tensor.

    Channels:
        0 — to_move: 1.0 everywhere if it's P1's turn, 0.0 if P2's turn
        1 — P1 pieces
        2 — P2 pieces
    """
    board = np.zeros((ROWS, COLS), dtype=np.float32)
    heights = np.zeros(COLS, dtype=np.int8)
    player = 1
    for ch in sequence:
        col = int(ch) - 1
        board[heights[col], col] = player
        heights[col] += 1
        player = 2 if player == 1 else 1

    to_move = float(player == 1)
    p1 = (board == 1).astype(np.float32)
    p2 = (board == 2).astype(np.float32)
    tensor = np.stack([
        np.full((ROWS, COLS), to_move, dtype=np.float32),
        p1,
        p2,
    ])
    return torch.from_numpy(tensor)


def sequences_to_batch(sequences: list[str]) -> torch.Tensor:
    """Return (N, 3, 6, 7) batch tensor."""
    return torch.stack([sequence_to_tensor(s) for s in sequences])


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class NetConfig:
    channels: int = 3       # input channels
    filters: int = 32       # conv filters throughout body
    n_residuals: int = 3    # residual blocks (increase for stronger play)
    n_fc_layers: int = 2    # FC layers in value head


@dataclass
class TrainConfig:
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 1e-4
    batch_size: int = 512
    epochs: int = 10
    lr_milestones: list[int] = field(default_factory=lambda: [5, 8])
    lr_gamma: float = 0.1
    use_gpu: bool = True


# ---------------------------------------------------------------------------
# Network modules
# ---------------------------------------------------------------------------

def _conv_bn_relu(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(inplace=True),
    )


class ResidualBlock(nn.Module):
    def __init__(self, filters: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(filters)
        self.relu  = nn.LeakyReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)


class PolicyHead(nn.Module):
    """Output: (N, 7) softmax probabilities."""
    def __init__(self, filters: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(filters, 2, kernel_size=1)
        self.bn   = nn.BatchNorm2d(2)
        self.relu = nn.LeakyReLU(inplace=True)
        self.fc   = nn.Linear(2 * AREA, COLS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn(self.conv(x)))
        x = x.view(x.size(0), -1)
        return torch.softmax(self.fc(x), dim=1)


class ValueHead(nn.Module):
    """Output: (N,) scalar in [0, 1] (win probability from current player's view)."""
    def __init__(self, filters: int, n_fc: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(filters, 1, kernel_size=1)
        self.bn   = nn.BatchNorm2d(1)
        self.relu = nn.LeakyReLU(inplace=True)
        fc = []
        for _ in range(n_fc):
            fc.extend([nn.Linear(AREA, AREA), nn.LeakyReLU(inplace=True)])
        fc.append(nn.Linear(AREA, 1))
        self.fc   = nn.Sequential(*fc)
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn(self.conv(x)))
        x = x.view(x.size(0), -1)
        x = self.tanh(self.fc(x))
        return ((x + 1) * 0.5).view(-1)  # tanh → [0, 1]


class Connect4Net(nn.Module):
    def __init__(self, cfg: NetConfig = NetConfig()) -> None:
        super().__init__()
        self.body = nn.Sequential(
            _conv_bn_relu(cfg.channels, cfg.filters),
            *[ResidualBlock(cfg.filters) for _ in range(cfg.n_residuals)],
        )
        self.policy_head = PolicyHead(cfg.filters)
        self.value_head  = ValueHead(cfg.filters, cfg.n_fc_layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.body(x)
        return self.policy_head(features), self.value_head(features)


# ---------------------------------------------------------------------------
# Inference wrapper  (compatible with MLPolicyClient interface)
# ---------------------------------------------------------------------------

class Connect4NetClient:
    """Wraps Connect4Net for inference. best_move() matches MLPolicyClient interface."""

    def __init__(
        self,
        model_path: str,
        net_config: Optional[NetConfig] = None,
        difficulty: str = "hard",
        device: Optional[str] = None,
    ) -> None:
        import random
        self._rng = random.Random()
        self.difficulty = difficulty

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        cfg = net_config or NetConfig()
        self.net = Connect4Net(cfg)
        checkpoint = torch.load(model_path, map_location=self.device)
        state = checkpoint.get("net_state_dict", checkpoint)
        self.net.load_state_dict(state)
        self.net.to(self.device)
        self.net.eval()

    @torch.no_grad()
    def best_move(self, sequence: str) -> int:
        """Return 0-based column. Legal mask applied before selection."""
        x = sequence_to_tensor(sequence).unsqueeze(0).to(self.device)
        policy, _ = self.net(x)
        probs = policy.cpu().numpy()[0]  # shape (7,)

        # Legal mask
        heights = [0] * COLS
        for ch in sequence:
            heights[int(ch) - 1] += 1
        legal = np.array([h < ROWS for h in heights])
        probs[~legal] = 0.0
        if probs.sum() == 0:
            probs = legal.astype(np.float32)
        probs = probs / probs.sum()

        if self.difficulty == "hard":
            return int(np.argmax(probs))
        legal_cols = np.flatnonzero(legal).tolist()
        legal_probs = probs[legal_cols].tolist()
        if self.difficulty == "medium":
            return self._rng.choices(legal_cols, weights=legal_probs, k=1)[0]
        # easy: temperature softening
        log_p = np.log(np.maximum(probs[legal], 1e-8))
        soft = np.exp(log_p / 3.0)
        soft /= soft.sum()
        return self._rng.choices(legal_cols, weights=soft.tolist(), k=1)[0]

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# MCTS inference wrapper  (AlphaZero-style: network + tree search)
# ---------------------------------------------------------------------------

class MCTSClient:
    """Connect4Net + MCTS for stronger inference. Drop-in for Connect4NetClient."""

    def __init__(
        self,
        model_path: str,
        net_config: Optional[NetConfig] = None,
        simulations: int = 200,
        c_puct: float = 1.25,
        device: Optional[str] = None,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        cfg = net_config or NetConfig()
        net = Connect4Net(cfg)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        state = checkpoint.get("net_state_dict", checkpoint)
        net.load_state_dict(state)

        from src.fine_tune.mcts import MCTS
        self.mcts = MCTS(net, simulations=simulations, c_puct=c_puct, device=device)
        self.simulations = simulations

    def best_move(self, sequence: str) -> int:
        """Return 0-based column via MCTS search."""
        return self.mcts.best_move(sequence)

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = NetConfig(filters=32, n_residuals=3)
    net = Connect4Net(cfg)
    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    dummy = sequences_to_batch(["", "4", "44", "4416721"])
    policy, value = net(dummy)
    print(f"Policy shape: {policy.shape}  Value shape: {value.shape}")
    print(f"Policy sum (should be ~1): {policy.sum(dim=1)}")
    print(f"Value range: [{value.min():.3f}, {value.max():.3f}]")
