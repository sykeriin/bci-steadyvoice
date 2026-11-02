"""Per-day input adapters. All are identity-initialised and are the ONLY thing that adapts."""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

State = Dict[str, torch.Tensor]


class AffineAdapter(nn.Module):
    kind = "affine"

    def __init__(self, C: int, **_):
        super().__init__()
        self.W = nn.Parameter(torch.eye(C))
        self.b = nn.Parameter(torch.zeros(C))

    def forward(self, x):
        return x @ self.W + self.b


class DiagAdapter(nn.Module):
    kind = "diag"

    def __init__(self, C: int, **_):
        super().__init__()
        self.g = nn.Parameter(torch.ones(C))
        self.b = nn.Parameter(torch.zeros(C))

    def forward(self, x):
        return x * self.g + self.b


class LowRankAdapter(nn.Module):
    """x + (x U) V + b with V = 0 at init (identity)."""
    kind = "lowrank"

    def __init__(self, C: int, rank: int = 16, **_):
        super().__init__()
        self.U = nn.Parameter(torch.randn(C, rank) / C ** 0.5)
        self.V = nn.Parameter(torch.zeros(rank, C))
        self.b = nn.Parameter(torch.zeros(C))

    def forward(self, x):
        return x + (x @ self.U) @ self.V + self.b


_KINDS = {"affine": AffineAdapter, "diag": DiagAdapter, "lowrank": LowRankAdapter}


def make_adapter(kind: str, C: int, rank: int = 16) -> nn.Module:
    return _KINDS[kind](C, rank=rank)


def get_state(a: nn.Module) -> State:
    return {k: v.detach().clone().cpu() for k, v in a.state_dict().items()}


def load_state(a: nn.Module, s: State) -> nn.Module:
    a.load_state_dict({k: v.to(next(a.parameters()).device) for k, v in s.items()})
    return a


def clone_state(s: State) -> State:
    return {k: v.clone() for k, v in s.items()}


def state_dist(a: State, b: State) -> float:
    return float(sum(((a[k] - b[k]) ** 2).sum() for k in a) ** 0.5)
