"""Label-free health signals for a candidate adapter on today's unlabelled data."""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from ..adapt.common import Ctx, latent_moments
from ..utils import norm_edit, pad_batch

SIGNAL_NAMES = ["entropy", "blank_frac", "unconf", "lm_rewrite", "latent_dist"]


@torch.no_grad()
def compute_signals(ctx: Ctx, state, xs) -> Dict[str, float]:
    xs = list(xs)[: ctx.hp["sig_utts"]]
    ad = ctx.new_adapter(state)
    x, l = pad_batch(xs, ctx.device)
    logits, ol, h = ctx.model(x, l, ad, return_latent=True)
    p = torch.softmax(logits.float(), -1)
    m = (torch.arange(p.shape[1], device=p.device)[None] < ol[:, None]).float()
    n = m.sum()
    ent = (-(p * torch.log(p + 1e-8)).sum(-1) * m).sum() / n
    blank = ((p.argmax(-1) == 0).float() * m).sum() / n
    unconf = ((1 - p.max(-1).values) * m).sum() / n
    lps = [torch.log(p[i, : int(ol[i])] + 1e-12).cpu().numpy() for i in range(len(xs))]
    g, lmd = ctx.greedy(lps), ctx.lm_decode(lps)
    rewrite = float(np.mean([norm_edit(a, b) for a, b in zip(g, lmd)]))
    lat = float("nan")
    if ctx.ref is not None:
        mu, cov = latent_moments(h, ol)
        H = len(mu)
        lat = float(((mu - ctx.ref.lat_mu) ** 2).sum() / H
                    + ((cov - ctx.ref.lat_cov) ** 2).sum() / (H * H) ** 0.5 / H)
    return dict(entropy=float(ent), blank_frac=float(blank), unconf=float(unconf),
                lm_rewrite=rewrite, latent_dist=float(np.log1p(lat * 10)))


def to_vec(s: Dict[str, float]) -> np.ndarray:
    return np.array([s[k] for k in SIGNAL_NAMES], dtype=np.float64)
