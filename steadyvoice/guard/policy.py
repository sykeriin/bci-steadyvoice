"""Collapse guard: choose among {carry-over, candidate, last-good}; fall back to a labelled
calibration prompt when nothing looks healthy. Every decision uses label-free signals only,
except the (counted) calibration request."""
from __future__ import annotations

from typing import Dict, Optional

from ..adapt.common import Ctx, supervised_fit
from ..models.adapters import State, clone_state
from .predictor import PERPredictor
from .signals import compute_signals


class Guard:
    def __init__(self, predictor: PERPredictor, hp: Optional[dict] = None):
        self.pred, self.hp = predictor, hp or {}
        self.last_good: Optional[State] = None

    def start(self, init_state: State):
        self.last_good = clone_state(init_state)

    def step(self, ctx: Ctx, xs, carry: State, cand: State):
        hp = ctx.hp
        opts = {"carry": carry, "cand": cand}
        if self.last_good is not None:
            opts["last_good"] = self.last_good
        pred = {k: self.pred.predict(compute_signals(ctx, s, xs)) for k, s in opts.items()}
        choice = "carry"
        for k in ("cand", "last_good"):
            if k in pred and pred[k] < pred[choice] - hp["margin"]:
                choice = k
        action = {"carry": "reject", "cand": "accept", "last_good": "rollback"}[choice]
        state = opts[choice]
        info: Dict = {"pred_carry": pred["carry"], "pred_cand": pred["cand"],
                      "pred_chosen": pred[choice]}
        if pred[choice] > hp["tau_cal"]:      # nothing healthy -> ask the user for a short calibration
            xl, yl = ctx.calib.request(hp["oracle_n"])
            state = supervised_fit(ctx, state, xl, yl)
            action = "calibrate"
            self.last_good = clone_state(state)
        elif pred[choice] < hp["tau_ok"]:
            self.last_good = clone_state(state)
        info["action"] = action
        return state, info
