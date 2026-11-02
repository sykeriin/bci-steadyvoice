"""Small shared helpers: batching, seeding, edit distance, PER."""
from __future__ import annotations

import random
from typing import Sequence

import editdistance
import numpy as np
import torch

BLANK = 0


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pad_batch(xs: Sequence[np.ndarray], device: str | torch.device = "cpu"):
    """list of [T_i, C] -> ([B, Tmax, C] float tensor, [B] long lengths)."""
    lens = torch.tensor([len(x) for x in xs], dtype=torch.long)
    out = torch.zeros(len(xs), int(lens.max()), xs[0].shape[1], dtype=torch.float32)
    for i, x in enumerate(xs):
        out[i, : len(x)] = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
    return out.to(device), lens.to(device)


def norm_edit(a: Sequence[int], b: Sequence[int]) -> float:
    """Edit distance normalised by the longer sequence, in [0, 1]."""
    n = max(len(a), len(b))
    return 0.0 if n == 0 else editdistance.eval(list(a), list(b)) / n


def per(refs: Sequence[Sequence[int]], hyps: Sequence[Sequence[int]]) -> float:
    """Corpus-level phoneme error rate."""
    errs = sum(editdistance.eval(list(r), list(h)) for r, h in zip(refs, hyps))
    tot = sum(len(r) for r in refs)
    return errs / max(tot, 1)
