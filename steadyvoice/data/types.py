"""Data containers.

`DayData` is the full record and lives only inside SessionStream / LabelVault.
`DayBatch` is the strictly label-free view that adaptation methods receive.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class DayData:
    session_id: int
    date: int  # calendar day index (days since first session)
    adapt_x: List[np.ndarray]  # [T, C] each; unlabelled at test time
    adapt_y: List[List[int]]
    eval_x: List[np.ndarray]
    eval_y: List[List[int]]


@dataclass(frozen=True)
class DayBatch:
    """What an adaptation method may see on day d: features only, no labels."""

    session_id: int
    date: int
    adapt_x: Tuple[np.ndarray, ...]
