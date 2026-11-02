"""Train the causal GRU backbone with one affine adapter per TRAINING day (labelled days only)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.types import DayData
from ..decode.ctc import forward_logp, greedy
from ..models.adapters import get_state, make_adapter
from ..models.backbone import CausalGRUDecoder
from ..utils import pad_batch, per, seed_everything


@dataclass
class TrainCfg:
    hidden: int = 128
    layers: int = 2
    patch: int = 4
    stride: int = 2
    dropout: float = 0.1
    adapter: str = "affine"
    rank: int = 16
    steps: int = 1500
    bs: int = 16
    lr: float = 2e-3
    noise: float = 0.1
    seed: int = 0
    amp: bool = False


def ctc_loss(logits, out_lens, targets):
    lp = F.log_softmax(logits.float(), -1).transpose(0, 1)
    tl = torch.tensor([len(t) for t in targets])
    tg = torch.tensor([p for t in targets for p in t])
    return F.ctc_loss(lp, tg, out_lens.cpu(), tl, blank=0, zero_infinity=True)


def train_backbone(days: List[DayData], n_classes: int, cfg: TrainCfg, device="cpu", log=print):
    seed_everything(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    C = days[0].adapt_x[0].shape[1]
    model = CausalGRUDecoder(C, cfg.hidden, cfg.layers, n_classes, cfg.patch, cfg.stride, cfg.dropout).to(device)
    adapters = nn.ModuleList([make_adapter(cfg.adapter, C, cfg.rank) for _ in days]).to(device)
    opt = torch.optim.AdamW(list(model.parameters()) + list(adapters.parameters()), lr=cfg.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, cfg.lr, total_steps=cfg.steps, pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=cfg.amp and device != "cpu")
    model.train()
    for step in range(cfg.steps):
        di = int(rng.integers(len(days)))
        d = days[di]
        idx = rng.choice(len(d.adapt_x), cfg.bs, replace=False)
        x, l = pad_batch([d.adapt_x[i] for i in idx], device)
        x = x + cfg.noise * torch.randn_like(x)
        with torch.autocast(device_type="cuda" if device != "cpu" else "cpu", enabled=cfg.amp and device != "cpu"):
            logits, ol = model(x, l, adapters[di])
        loss = ctc_loss(logits, ol, [d.adapt_y[i] for i in idx])
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(list(model.parameters()) + list(adapters.parameters()), 5.0)
        scaler.step(opt); scaler.update(); sched.step()
        if step % 250 == 0 or step == cfg.steps - 1:
            log(f"  backbone step {step:5d} loss {loss.item():.3f}")
    states = [get_state(a) for a in adapters]
    # validation on the held-out eval trials of training days
    pers = []
    for d, st in zip(days, states):
        a = make_adapter(cfg.adapter, C, cfg.rank).to(device)
        a.load_state_dict({k: v.to(device) for k, v in st.items()})
        lps = forward_logp(model, a, d.eval_x, device)
        pers.append(per(d.eval_y, [greedy(lp) for lp in lps]))
    log(f"  backbone train-day val PER (greedy, LM-off): {np.round(pers, 3).tolist()}")
    return model, states, pers
