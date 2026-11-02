"""SteadyVoice: adapter-only adaptation from three label-free signals + collapse guard."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ..guard.policy import Guard
from ..models.adapters import clone_state
from ..train.backbone_train import ctc_loss
from ..utils import norm_edit
from .base import AdaptMethod
from .baselines import coral_state
from .common import Ctx, fit_adapter, latent_moments


def _aug(ctx: Ctx, xb, shift: int):
    """Strong view: channel dropout (to batch mean), gain jitter, causal time shift by `shift` frames."""
    B, T, C = xb.shape
    mean = xb.mean((0, 1), keepdim=True)
    keep = (torch.rand(B, 1, C, device=xb.device) > ctx.hp["chan_drop"]).float()
    x = xb * keep + mean * (1 - keep)
    x = x * (1 + 0.05 * torch.randn(B, 1, C, device=xb.device))
    s = shift * ctx.model.stride
    if s > 0:
        x = torch.cat([x[:, :1].expand(-1, s, -1), x[:, :-s]], 1)
    elif s < 0:
        x = torch.cat([x[:, -s:], x[:, -1:].expand(-1, -s, -1)], 1)
    return x


class SteadyVoice(AdaptMethod):
    name = "steadyvoice"

    def __init__(self, guard: Guard | None = None, name: str | None = None, **hp_override):
        self.guard, self.hp_override = guard, hp_override
        if name:
            self.name = name

    def start(self, ctx, init_state):
        ctx.hp.update(self.hp_override)
        if self.guard:
            self.guard.start(init_state)

    # --- one day -----------------------------------------------------------------------
    def adapt(self, ctx: Ctx, state, batch):
        hp, xs = ctx.hp, list(batch.adapt_x)
        s0 = coral_state(ctx, state, xs, hp["coral_shrink"]) if hp["sv_coral_warm"] else clone_state(state)
        s = s0
        diag = {"kept_frac": float("nan")}
        for r in range(hp["sv_rounds"]):
            lps = ctx.logp(s, xs)
            g, lmd = ctx.greedy(lps), ctx.lm_decode(lps)
            keep = np.array([
                (1 - norm_edit(a, b) >= hp["tau_agree"]) and len(b) > 0
                and float(np.exp(lp).max(-1).mean()) >= hp["conf_min"]
                for a, b, lp in zip(g, lmd, lps)])
            diag["kept_frac"] = float(keep.mean())
            targets = [b if k else None for b, k in zip(lmd, keep)]
            ref = {k: v.to(ctx.device) for k, v in s0.items()}

            def loss_fn(ad, xb, lb, idx, targets=targets, ref=ref):
                logits, ol, h = ctx.model(xb, lb, ad, return_latent=True)
                loss = xb.new_zeros(())
                sel = [i for i, t in enumerate(idx) if targets[t] is not None]
                if hp["use_pl"] and sel:
                    loss = loss + hp["lam_pl"] * ctc_loss(logits[sel], ol[sel], [targets[idx[i]] for i in sel])
                if hp["use_al"] and ctx.ref is not None:
                    mu, cov = latent_moments(h, ol)
                    H = len(mu)
                    loss = loss + hp["lam_al"] * (((mu - ctx.ref.lat_mu) ** 2).sum() / H
                                                  + ((cov - ctx.ref.lat_cov) ** 2).sum() / (H * H))
                if hp["use_cons"]:
                    sh = int(np.random.randint(-1, 2))
                    lw = F.softmax(logits.detach().float(), -1)
                    ls, ols = ctx.model(_aug(ctx, xb, sh), lb, ad)
                    lss = F.log_softmax(ls.float(), -1)
                    a, b = (lw, lss)
                    if sh > 0:
                        a, b = lw[:, :-sh], lss[:, sh:]
                    elif sh < 0:
                        a, b = lw[:, -sh:], lss[:, :sh]
                    m = (torch.arange(a.shape[1], device=a.device)[None] < (ol - abs(sh))[:, None]).float()
                    m = m * (a.max(-1).values >= hp["cons_conf"]).float()
                    kl = (a * (torch.log(a + 1e-8) - b)).sum(-1)
                    loss = loss + hp["lam_cons"] * (kl * m).sum() / m.sum().clamp(min=1)
                prox = sum(((p - ref[k]) ** 2).sum() for k, p in ad.named_parameters())
                return loss + hp["lam_prox"] * prox

            s = fit_adapter(ctx, s, xs, loss_fn, hp["sv_epochs"], hp["sv_lr"], seed=r)
        if self.guard is None:
            return s, {**diag, "action": "accept"}
        out, info = self.guard.step(ctx, xs, carry=state, cand=s)
        return out, {**diag, **info}
