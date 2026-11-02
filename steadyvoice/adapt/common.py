"""Shared context + helpers for adaptation methods."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from ..decode.ctc import forward_logp, greedy
from ..decode.lm import PhonemeNGramLM, prefix_beam_search
from ..models.adapters import State, get_state, load_state, make_adapter
from ..train.backbone_train import ctc_loss
from ..utils import pad_batch

DEFAULT_HP = dict(
    bs=10,
    # supervised oracle / calibration refit
    oracle_n=20, oracle_epochs=25, oracle_lr=3e-3,
    # LM decode
    lm_alpha=0.6, lm_beta=0.3, beam=8, topk=6,
    # CORP-style (ungated pseudo-labels)
    corp_rounds=2, corp_epochs=3, corp_lr=1e-3,
    # Tent
    tent_epochs=3, tent_lr=1e-3,
    # SteadyVoice
    sv_rounds=2, sv_epochs=3, sv_lr=2e-3, sv_coral_warm=True, coral_shrink=0.1,
    lam_pl=1.0, lam_al=5.0, lam_cons=0.5, lam_prox=1e-2,
    tau_agree=0.7, conf_min=0.6, chan_drop=0.15, cons_conf=0.8,
    use_pl=True, use_al=True, use_cons=True,
    # guard
    sig_utts=10, tau_cal=0.30, tau_ok=0.15, margin=0.02,
)


@dataclass
class Reference:
    feat_mu: np.ndarray  # [C]   adapter-output feature mean on training days
    feat_cov: np.ndarray  # [C,C]
    lat_mu: torch.Tensor  # [H]  GRU-latent mean
    lat_cov: torch.Tensor  # [H,H]


@dataclass
class Ctx:
    model: torch.nn.Module
    C: int
    kind: str
    rank: int
    device: str
    lm: Optional[PhonemeNGramLM]
    ref: Optional[Reference] = None
    hp: Dict = field(default_factory=lambda: dict(DEFAULT_HP))
    calib: object = None  # CalibrationService for the current day (set by runner)

    def new_adapter(self, state: State):
        return load_state(make_adapter(self.kind, self.C, self.rank).to(self.device), state)

    def logp(self, state: State, xs) -> List[np.ndarray]:
        return forward_logp(self.model, self.new_adapter(state), xs, self.device)

    def lm_decode(self, lps) -> List[List[int]]:
        h = self.hp
        return [prefix_beam_search(lp, self.lm, h["lm_alpha"], h["lm_beta"], h["beam"], h["topk"]) for lp in lps]

    def greedy(self, lps):
        return [greedy(lp) for lp in lps]


def latent_moments(h: torch.Tensor, out_lens: torch.Tensor):
    """Mean/cov of GRU latents over valid frames. h [B,T,H]."""
    mask = torch.arange(h.shape[1], device=h.device)[None] < out_lens[:, None]
    v = h[mask]
    mu = v.mean(0)
    d = v - mu
    return mu, d.T @ d / max(len(v) - 1, 1)


@torch.no_grad()
def compute_reference(ctx: Ctx, days, states, n_utts: int = 60) -> Reference:
    fs, fs2, nf = 0.0, 0.0, 0
    ls, ls2, nl = 0.0, 0.0, 0
    for d, st in zip(days, states):
        ad = ctx.new_adapter(st)
        xs = d.adapt_x[:n_utts]
        for i in range(0, len(xs), 32):
            x, l = pad_batch(xs[i:i + 32], ctx.device)
            logits, ol, h = ctx.model(x, l, ad, return_latent=True)
            xa = ad(x)
            m = torch.arange(x.shape[1], device=x.device)[None] < l[:, None]
            v = xa[m].double(); fs = fs + v.sum(0); fs2 = fs2 + v.T @ v; nf += len(v)
            mo = torch.arange(h.shape[1], device=x.device)[None] < ol[:, None]
            u = h[mo].double(); ls = ls + u.sum(0); ls2 = ls2 + u.T @ u; nl += len(u)
    fm = fs / nf; lm_ = ls / nl
    return Reference(
        (fm).cpu().numpy(), ((fs2 / nf) - torch.outer(fm, fm)).cpu().numpy(),
        lm_.float().to(ctx.device), (((ls2 / nl) - torch.outer(lm_, lm_))).float().to(ctx.device))


def fit_adapter(ctx: Ctx, state: State, xs, loss_fn: Callable, epochs: int, lr: float,
                seed: int = 0) -> State:
    """Generic mini-batch Adam fit of the adapter only. loss_fn(adapter, xb, lens, idx) -> scalar."""
    ad = ctx.new_adapter(state)
    opt = torch.optim.Adam(ad.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    bs = ctx.hp["bs"]
    for _ in range(epochs):
        perm = rng.permutation(len(xs))
        for j in range(0, len(xs), bs):
            idx = perm[j:j + bs]
            xb, lb = pad_batch([xs[i] for i in idx], ctx.device)
            loss = loss_fn(ad, xb, lb, idx)
            if not torch.isfinite(loss):
                continue
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ad.parameters(), 5.0)
            opt.step()
    return get_state(ad)


def supervised_fit(ctx: Ctx, state: State, xs, ys, epochs=None, lr=None) -> State:
    hp = ctx.hp
    def loss_fn(ad, xb, lb, idx):
        logits, ol = ctx.model(xb, lb, ad)
        return ctc_loss(logits, ol, [ys[i] for i in idx])
    return fit_adapter(ctx, state, xs, loss_fn, epochs or hp["oracle_epochs"], lr or hp["oracle_lr"])
