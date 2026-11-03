"""Linear-probe diagnostic (analysis only; uses labels, never feeds back into any method).

Question: after adaptation, are phonemes still linearly decodable from the adapted latents?
Refit ONLY a fresh linear CTC head on the day's labelled adapt trials, score on the day's eval trials.
  probe_per << frozen-head PER  -> information preserved, readout misaligned (adapter is fine, just imperfect)
  probe_per high                -> adapter destroyed information (silent failure)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..decode.ctc import greedy
from ..utils import pad_batch, per


@torch.no_grad()
def _latents(ctx, state, xs):
    ad, out = ctx.new_adapter(state), []
    for i in range(0, len(xs), 32):
        x, l = pad_batch(xs[i:i + 32], ctx.device)
        _, ol, h = ctx.model(x, l, ad, return_latent=True)
        out += [h[j, : int(ol[j])] for j in range(len(h))]
    return out


def linear_probe_per(ctx, state, day, steps: int = 150, lr: float = 0.05, n_classes: int = 41) -> float:
    H = ctx.model.head.in_features
    tr, te = _latents(ctx, state, day.adapt_x), _latents(ctx, state, day.eval_x)
    head = torch.nn.Linear(H, n_classes).to(ctx.device)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    torch.manual_seed(0)
    for _ in range(steps):
        idx = torch.randperm(len(tr))[:16].tolist()
        lens = torch.tensor([len(tr[i]) for i in idx])
        pad = torch.nn.utils.rnn.pad_sequence([tr[i] for i in idx], batch_first=True)
        lp = F.log_softmax(head(pad), -1).transpose(0, 1)
        tg = torch.tensor([p for i in idx for p in day.adapt_y[i]])
        loss = F.ctc_loss(lp, tg, lens, torch.tensor([len(day.adapt_y[i]) for i in idx]), zero_infinity=True)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        hyp = [greedy(F.log_softmax(head(t), -1).cpu().numpy()) for t in te]
    return per(day.eval_y, hyp)
