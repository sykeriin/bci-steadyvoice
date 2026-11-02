"""Forward-in-time stream with labels sealed in a vault.

Leak-proofing by construction:
  * adaptation methods get a `DayBatch` (features only; eval trials of the
    same day are withheld, as they are future utterances in deployment);
  * labels are only reachable through the runner-owned `LabelVault`, or through
    a `CalibrationService`, which counts every labelled sentence it hands out.
"""
from __future__ import annotations

from typing import List, Tuple

from .types import DayBatch, DayData


class LabelVault:
    def __init__(self, days: List[DayData]):
        self._days = days

    def eval_labels(self, i: int):
        return self._days[i].eval_y

    def adapt_labels(self, i: int):
        return self._days[i].adapt_y


class CalibrationService:
    """The only channel through which a method can obtain labels. Every call is charged."""

    def __init__(self, days: List[DayData], day_idx: int):
        self._days, self._i = days, day_idx
        self.labels_used = 0

    def request(self, n: int) -> Tuple[list, list]:
        d = self._days[self._i]
        n = min(n, len(d.adapt_x))
        self.labels_used += n
        return list(d.adapt_x[:n]), [list(y) for y in d.adapt_y[:n]]


class SessionStream:
    def __init__(self, days: List[DayData]):
        self.days = days
        self.vault = LabelVault(days)

    def __len__(self) -> int:
        return len(self.days)

    def batch(self, i: int) -> DayBatch:
        d = self.days[i]
        return DayBatch(d.session_id, d.date, tuple(d.adapt_x))

    def eval_x(self, i: int):  # runner only
        return self.days[i].eval_x

    def calibration(self, i: int) -> CalibrationService:
        return CalibrationService(self.days, i)
