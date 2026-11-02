from __future__ import annotations

from typing import Dict, Tuple

from ..data.types import DayBatch
from ..models.adapters import State
from .common import Ctx


class AdaptMethod:
    name = "base"

    def start(self, ctx: Ctx, init_state: State) -> None:
        """Called once before the forward simulation (e.g. to reset guard memory)."""

    def adapt(self, ctx: Ctx, state: State, batch: DayBatch) -> Tuple[State, Dict]:
        """Return (state used to decode today, diagnostics). `batch` has NO labels.
        Only methods that explicitly call ctx.calib.request(n) ever see labels, and are charged for it."""
        raise NotImplementedError
