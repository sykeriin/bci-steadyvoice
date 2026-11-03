"""Synthetic drifting speech-BCI data (for tests and pipeline development).

Latent phoneme code -> fixed mixing -> channels, then a per-day drift
(gain, offset, rotation, dead channels, occasional jump) that random-walks
across days. A language (lexicon + sparse word bigram) gives the LM real signal.
This is NOT a substitute for real data; it exists so every component can be
verified end to end (drift hurts, oracle fixes it, guard rolls back, no leakage).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy.linalg import expm
from scipy.ndimage import gaussian_filter1d

from .types import DayData


@dataclass
class SynthConfig:
    C: int = 64
    L: int = 12
    n_phon: int = 40  # phoneme ids are 1..n_phon (0 = CTC blank)
    n_words: int = 60
    seed: int = 0
    n_days: int = 40
    n_train_days: int = 5
    n_adapt_train: int = 150
    n_adapt: int = 40
    n_eval: int = 30
    noise: float = 1.3
    min_dur: int = 4
    max_dur: int = 8
    # drift (per-day random-walk scales)
    gain_sd: float = 0.07
    off_sd: float = 0.12
    rot_sd: float = 0.22
    dead_prob: float = 0.15
    dead_frac: float = 0.05
    jump_prob: float = 0.04
    jump_scale: float = 4.0
    drift_family: str = "linear"   # "nonlinear" adds per-channel curvature (not in the guard's perturbation family)
    curv_sd: float = 0.04


class SyntheticWorld:
    def __init__(self, cfg: SynthConfig):
        self.cfg = cfg
        r = np.random.default_rng(cfg.seed)
        self.emb = r.normal(size=(cfg.n_phon + 1, cfg.L)).astype(np.float32)
        self.emb[0] = 0.0
        self.M = (r.normal(size=(cfg.L, cfg.C)) / np.sqrt(cfg.L)).astype(np.float32)
        self.lexicon = [
            list(r.integers(1, cfg.n_phon + 1, size=r.integers(2, 6))) for _ in range(cfg.n_words)
        ]
        # sparse word bigram: each word has ~4 likely successors
        self.succ = [r.choice(cfg.n_words, size=4, replace=False) for _ in range(cfg.n_words)]

    def sentence(self, r: np.random.Generator) -> List[int]:
        w = int(r.integers(self.cfg.n_words))
        ph: List[int] = []
        for _ in range(int(r.integers(3, 6))):
            ph += [int(p) for p in self.lexicon[w]]
            w = int(r.choice(self.succ[w]))
        return ph

    def corpus(self, n: int, seed: int) -> List[List[int]]:
        r = np.random.default_rng(seed)
        return [self.sentence(r) for _ in range(n)]

    def render(self, phons: List[int], drift: "Drift", r: np.random.Generator) -> np.ndarray:
        c = self.cfg
        durs = r.integers(c.min_dur, c.max_dur + 1, size=len(phons))
        z = np.concatenate(
            [np.zeros((3, c.L), np.float32)]
            + [np.repeat(self.emb[p][None], d, axis=0) for p, d in zip(phons, durs)]
            + [np.zeros((3, c.L), np.float32)]
        )
        z = gaussian_filter1d(z, sigma=1.2, axis=0) * 2.0
        x = z @ self.M + r.normal(scale=c.noise, size=(len(z), c.C)).astype(np.float32)
        return drift.apply(x)


class Drift:
    """Cumulative drift state for one day."""

    def __init__(self, C: int):
        self.log_gain = np.zeros(C)
        self.offset = np.zeros(C)
        self.Q = np.eye(C)
        self.dead = np.zeros(C, bool)
        self.curv = np.zeros(C)
        self.nonlinear = False

    def step(self, cfg: SynthConfig, r: np.random.Generator, first: bool):
        if first:
            return
        C = cfg.C
        s = cfg.jump_scale if r.random() < cfg.jump_prob else 1.0
        self.log_gain += r.normal(scale=cfg.gain_sd * s, size=C)
        self.offset += r.normal(scale=cfg.off_sd * s, size=C)
        A = r.normal(size=(C, C))
        A = (A - A.T) / np.sqrt(C)
        self.Q = self.Q @ expm(A * cfg.rot_sd * s)
        if cfg.drift_family == "nonlinear":
            self.nonlinear = True
            self.curv += r.normal(scale=cfg.curv_sd * s, size=C)
        if r.random() < cfg.dead_prob:
            self.dead[r.choice(C, size=max(1, int(cfg.dead_frac * C)), replace=False)] = True

    def apply(self, x: np.ndarray) -> np.ndarray:
        y = (x * np.exp(self.log_gain)) @ self.Q + self.offset
        if self.nonlinear:
            y = y + self.curv * y * np.abs(y)
        y[:, self.dead] = 0.1 * x[:, self.dead]
        return y.astype(np.float32)

    def copy(self) -> "Drift":
        d = Drift(len(self.offset))
        d.log_gain, d.offset, d.Q, d.dead = (
            self.log_gain.copy(), self.offset.copy(), self.Q.copy(), self.dead.copy())
        d.curv, d.nonlinear = self.curv.copy(), self.nonlinear
        return d


def make_dataset(cfg: SynthConfig) -> Tuple[SyntheticWorld, List[DayData]]:
    world = SyntheticWorld(cfg)
    r = np.random.default_rng(cfg.seed + 1)
    drift = Drift(cfg.C)
    days: List[DayData] = []
    date = 0
    for i in range(cfg.n_days):
        drift.step(cfg, r, first=(i == 0))
        date += 0 if i == 0 else int(r.choice([1, 1, 1, 2, 3]))
        n_ad = cfg.n_adapt_train if i < cfg.n_train_days else cfg.n_adapt
        ay = [world.sentence(r) for _ in range(n_ad)]
        ey = [world.sentence(r) for _ in range(cfg.n_eval)]
        days.append(DayData(
            session_id=i, date=date,
            adapt_x=[world.render(p, drift, r) for p in ay], adapt_y=ay,
            eval_x=[world.render(p, drift, r) for p in ey], eval_y=ey))
    return world, days
