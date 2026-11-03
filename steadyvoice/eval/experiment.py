"""Shared experiment core: train backbone on first k labelled days -> fit guard on TRAIN days ->
forward-simulate methods. Used by scripts/run_real.py (and mirrors scripts/run_synthetic.py)."""
from __future__ import annotations

import json
import os
from typing import List

import numpy as np
import torch

from ..adapt.baselines import Coral, Corp, GuardedNoAdapt, NoAdapt, Oracle, PeriodicOracle, Tent
from ..adapt.common import DEFAULT_HP, Ctx, compute_reference
from ..adapt.steadyvoice import SteadyVoice
from ..data.stream import SessionStream
from ..decode.lm import PhonemeNGramLM
from ..guard.policy import Guard
from ..guard.predictor import fit_predictor
from ..models.backbone import freeze_for_adaptation
from ..train.backbone_train import TrainCfg, train_backbone
from .metrics import summarize
from .plots import plot_per_vs_days
from .probe import linear_probe_per
from .runner import run_forward, run_gap_jump

ABLATIONS = {"sv_nowarm": dict(sv_coral_warm=False), "sv_nopl": dict(use_pl=False),
             "sv_noal": dict(use_al=False), "sv_nocons": dict(use_cons=False)}


def build_method(m: str, pred):
    simple = {"none": NoAdapt, "oracle": Oracle, "coral": Coral, "tent": Tent, "corp": Corp}
    if m in simple:
        return simple[m]()
    if m == "guard_only":
        return GuardedNoAdapt(Guard(pred))
    if m.startswith("periodic"):
        return PeriodicOracle(int(m[8:]))
    if m in ABLATIONS:
        return SteadyVoice(None, name=m, **ABLATIONS[m])
    return SteadyVoice(Guard(pred) if m == "steadyvoice" else None, name=m)


def run_experiment(days, n_phon: int, k: int, tcfg: TrainCfg, methods: List[str], lm_extra, out_dir: str,
                   tag: str, device: str, n_utts_ref: int = 60, log=print, n_perturb: int = 80,
                   gaps=(), probe_methods=(), hp_override=None, return_ctx: bool = False, min_rho: float = 0.5):
    os.makedirs(out_dir, exist_ok=True)
    C = days[0].adapt_x[0].shape[1]
    train_days = days[:k]
    log(f"[1/4] backbone on sessions 0..{k-1} (labelled); dates {[d.date for d in train_days]}")
    model, states, _ = train_backbone(train_days, n_phon + 1, tcfg, device, log)
    freeze_for_adaptation(model)
    # LM sees ONLY labelled training-session transcripts (+ caller-supplied external text, which the
    # caller must have filtered of benchmark sentences)
    lm = PhonemeNGramLM(list(lm_extra) + [y for d in train_days for y in d.adapt_y], n_phon)
    ctx = Ctx(model, C, tcfg.adapter, tcfg.rank, device, lm)
    ctx.ref = compute_reference(ctx, train_days, states, n_utts_ref)
    log("[2/4] guard predictor on labelled TRAINING days only")
    pred, pdiag = fit_predictor(ctx, train_days, states, n_perturb=n_perturb, seed=tcfg.seed, log=log, min_rho=min_rho)
    stream, rows = SessionStream(days), {}
    log("[3/4] forward simulation")
    for m in methods:
        ctx.hp = {**DEFAULT_HP, **(hp_override or {})}
        rows[m] = run_forward(stream, ctx, states[-1], build_method(m, pred), k, log=lambda s: None,
                              probe_fn=linear_probe_per if m in probe_methods else None)
        r = rows[m]
        log(f"  {m:22s} mean PER {np.mean([x['per'] for x in r]):.3f} worst {max(x['per'] for x in r):.3f} "
            f"labels {sum(x['labels_used'] for x in r)}")
    gap_rows = {}
    for m in (methods if gaps else []):
        ctx.hp = {**DEFAULT_HP, **(hp_override or {})}
        gap_rows[m] = run_gap_jump(stream, ctx, states[-1], build_method(m, pred), k, gaps)
    summ = summarize(rows)
    json.dump({"rows": rows, "gap_rows": gap_rows, "summary": summ, "guard_signal_spearman": pdiag},
              open(f"{out_dir}/results_{tag}.json", "w"), indent=1)
    plot_per_vs_days(rows, f"{out_dir}/per_vs_days_{tag}.png", title=f"{tag} (LM-off PER)")
    return (rows, summ, ctx, pred, states) if return_ctx else (rows, summ)
