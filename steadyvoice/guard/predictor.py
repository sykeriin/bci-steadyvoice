"""Learn 'signals -> true PER' from labelled TRAINING days only.

Training examples (all from days whose labels are legitimately available):
  (1) natural drift: adapter of training day i applied to training day j's data;
  (2) synthetic perturbations of training-day data (gain, offset, rotation, dead channels).
Caveat (stated in the write-up): on real data the perturbation family must be validated
against real cross-session drift; (1) is the assumption-free part.
"""
from __future__ import annotations

from typing import List

import numpy as np
from scipy.linalg import expm
from scipy.stats import spearmanr

from ..adapt.common import Ctx
from ..decode.ctc import greedy
from ..utils import per
from .signals import SIGNAL_NAMES, compute_signals, to_vec


class PERPredictor:
    def __init__(self, ridge: float = 1.0):
        self.ridge, self.mu, self.sd, self.w, self.b = ridge, None, None, None, 0.0

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-8
        Z = (X - self.mu) / self.sd
        A = Z.T @ Z + self.ridge * np.eye(Z.shape[1])
        self.b = y.mean()
        self.w = np.linalg.solve(A, Z.T @ (y - self.b))
        return self

    def predict_vec(self, x: np.ndarray) -> float:
        return float(np.clip(((x - self.mu) / self.sd) @ self.w + self.b, 0.0, 1.0))

    def predict(self, sig) -> float:
        return self.predict_vec(to_vec(sig))


def perturb(xs: List[np.ndarray], rng: np.random.Generator, s: float) -> List[np.ndarray]:
    C = xs[0].shape[1]
    allx = np.concatenate(xs)
    mu, sd = allx.mean(0), allx.std(0) + 1e-6
    kinds = rng.choice(4, size=rng.integers(1, 4), replace=False)
    gain = np.exp(rng.normal(scale=0.6 * s, size=C)) if 0 in kinds else np.ones(C)
    off = rng.normal(scale=1.5 * s, size=C) * sd if 1 in kinds else np.zeros(C)
    Q = np.eye(C)
    if 2 in kinds:
        A = rng.normal(size=(C, C)); A = (A - A.T) / np.sqrt(C)
        Q = expm(A * 2.0 * s)
    dead = rng.random(C) < 0.35 * s if 3 in kinds else np.zeros(C, bool)
    out = []
    for x in xs:
        y = ((x - mu) * gain) @ Q + mu + off
        y[:, dead] = mu[dead]
        out.append(y.astype(np.float32))
    return out


def fit_predictor(ctx: Ctx, days, states, n_perturb: int = 80, n_utts: int = 12, seed: int = 0,
                  log=print):
    """Returns (predictor, diagnostics dict with in-sample Spearman per signal)."""
    rng = np.random.default_rng(seed)
    X, y = [], []

    def add(state, xs, ys):
        lps = ctx.logp(state, xs)
        y.append(per(ys, [greedy(lp) for lp in lps]))
        X.append(to_vec(compute_signals(ctx, state, xs)))

    for i, st in enumerate(states):          # (1) natural cross-day drift
        for j, d in enumerate(days):
            add(st, d.adapt_x[:n_utts], d.adapt_y[:n_utts])
    for _ in range(n_perturb):               # (2) synthetic perturbations
        j = int(rng.integers(len(days)))
        d = days[j]
        idx = rng.choice(len(d.adapt_x), n_utts, replace=False)
        xs = perturb([d.adapt_x[t] for t in idx], rng, float(rng.random()))
        add(states[j], xs, [d.adapt_y[t] for t in idx])
    X, y = np.array(X), np.array(y)
    pred = PERPredictor().fit(X, y)
    diag = {n: float(spearmanr(X[:, k], y)[0]) for k, n in enumerate(SIGNAL_NAMES)}
    fit = np.array([pred.predict_vec(x) for x in X])
    diag["fit_spearman"] = float(spearmanr(fit, y)[0])
    log(f"  guard predictor fit on {len(y)} labelled-train examples; "
        f"spearman(signal, PER)={ {k: round(v, 2) for k, v in diag.items()} }")
    return pred, diag
