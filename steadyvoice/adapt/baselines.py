"""Baselines: none, oracle (supervised daily refit), closed-form CORAL, Tent, CORP-style."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ..models.adapters import clone_state, get_state
from ..train.backbone_train import ctc_loss
from ..utils import pad_batch
from .base import AdaptMethod
from .common import Ctx, fit_adapter, supervised_fit


class NoAdapt(AdaptMethod):
    name = "none"

    def adapt(self, ctx, state, batch):
        return state, {}


class Oracle(AdaptMethod):
    """Daily labelled recalibration (today's protocol): N sentences, supervised refit, chained."""
    name = "oracle"

    def adapt(self, ctx, state, batch):
        xs, ys = ctx.calib.request(ctx.hp["oracle_n"])
        return supervised_fit(ctx, state, xs, ys), {"action": "calibrate"}


def _sqrtm(S, p, floor=1e-6):
    w, V = np.linalg.eigh(S)
    return (V * np.maximum(w, floor) ** p) @ V.T


def coral_state(ctx: Ctx, state, xs, shrink: float):
    """Closed-form CORAL (mean + covariance matching) of adapter-output features to the
    training-day reference. Only defined for the affine adapter; composes into W, b."""
    if ctx.kind != "affine" or ctx.ref is None:
        return clone_state(state)
    ad = ctx.new_adapter(state)
    with torch.no_grad():
        f = torch.cat([ad(torch.from_numpy(x).to(ctx.device)) for x in xs]).double().cpu().numpy()
    mu, S = f.mean(0), np.cov(f.T)
    Sr, mur = ctx.ref.feat_cov, ctx.ref.feat_mu
    C = len(mu)
    S = (1 - shrink) * S + shrink * np.trace(S) / C * np.eye(C)
    Sr = (1 - shrink) * Sr + shrink * np.trace(Sr) / C * np.eye(C)
    A = _sqrtm(S, -0.5) @ _sqrtm(Sr, 0.5)
    W, b = state["W"].double().numpy(), state["b"].double().numpy()
    return {"W": torch.from_numpy(W @ A).float(), "b": torch.from_numpy((b - mu) @ A + mur).float()}


class Coral(AdaptMethod):
    name = "coral"

    def adapt(self, ctx, state, batch):
        return coral_state(ctx, state, batch.adapt_x, ctx.hp["coral_shrink"]), {}


class Tent(AdaptMethod):
    """Entropy minimisation on the adapter. Known to collapse for CTC; kept as a negative control."""
    name = "tent"

    def adapt(self, ctx, state, batch):
        xs = list(batch.adapt_x)

        def loss_fn(ad, xb, lb, idx):
            logits, ol = ctx.model(xb, lb, ad)
            p = F.softmax(logits.float(), -1)
            ent = -(p * torch.log(p + 1e-8)).sum(-1)
            m = (torch.arange(ent.shape[1], device=ent.device)[None] < ol[:, None]).float()
            return (ent * m).sum() / m.sum()

        return fit_adapter(ctx, state, xs, loss_fn, ctx.hp["tent_epochs"], ctx.hp["tent_lr"]), {}


class Corp(AdaptMethod):
    """CORP-style: decode -> LM refine -> train on ALL refined decodes (no gating). Adapter-only
    here for a like-for-like comparison with SteadyVoice (full-network variant is an ablation)."""
    name = "corp"

    def adapt(self, ctx, state, batch):
        xs = list(batch.adapt_x)
        hp = ctx.hp
        for r in range(hp["corp_rounds"]):
            targets = ctx.lm_decode(ctx.logp(state, xs))

            def loss_fn(ad, xb, lb, idx, targets=targets):
                logits, ol = ctx.model(xb, lb, ad)
                return ctc_loss(logits, ol, [targets[i] or [1] for i in idx])

            state = fit_adapter(ctx, state, xs, loss_fn, hp["corp_epochs"], hp["corp_lr"], seed=r)
        return state, {}


class PeriodicOracle(AdaptMethod):
    """Fixed-schedule supervised recalibration every `every` calendar days (RDumb-style reset,
    but with a labelled refit). No adaptation in between."""

    def __init__(self, every: int = 7):
        self.every, self.name = every, f"periodic{every}"
        self.last = None

    def start(self, ctx, init_state):
        self.last = None

    def adapt(self, ctx, state, batch):
        if self.last is None:
            self.last = batch.date - 0  # first test day counts from training end: set below
            self._t0 = batch.date
        if batch.date - self.last >= self.every:
            xs, ys = ctx.calib.request(ctx.hp["oracle_n"])
            self.last = batch.date
            return supervised_fit(ctx, state, xs, ys), {"action": "calibrate"}
        return state, {}


class GuardedNoAdapt(AdaptMethod):
    """Guard + labelled-calibration fallback but NO label-free adaptation (candidate == carry-over).
    Isolates what the label-free adaptation adds on top of the guard."""
    name = "guard_only"

    def __init__(self, guard):
        self.guard = guard

    def start(self, ctx, init_state):
        self.guard.start(init_state)

    def adapt(self, ctx, state, batch):
        return self.guard.step(ctx, list(batch.adapt_x), carry=state, cand=state)
