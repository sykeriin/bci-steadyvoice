"""Forward-in-time simulation. The ONLY place that touches labels for scoring, and only
AFTER the method has returned its state for that day."""
from __future__ import annotations

import time
from typing import Dict, List

import numpy as np

from ..adapt.base import AdaptMethod
from ..adapt.common import Ctx
from ..data.stream import SessionStream
from ..models.adapters import State, clone_state
from ..utils import per


def run_forward(stream: SessionStream, ctx: Ctx, init_state: State, method: AdaptMethod, k: int,
                with_lm_score: bool = True, log=print) -> List[Dict]:
    method.start(ctx, clone_state(init_state))
    state = clone_state(init_state)
    t_train_end = stream.days[k - 1].date
    last_calib_date = t_train_end
    rows = []
    for i in range(k, len(stream)):
        ctx.calib = stream.calibration(i)
        batch = stream.batch(i)                      # features only
        t0 = time.perf_counter()
        state, diag = method.adapt(ctx, state, batch)
        dt = time.perf_counter() - t0
        d = stream.days[i]
        if diag.get("action") == "calibrate" or ctx.calib.labels_used > 0:
            last_calib_date = d.date
        ex = stream.eval_x(i)
        ey = stream.vault.eval_labels(i)             # labels touched only now, for scoring
        lps = ctx.logp(state, ex)
        row = dict(method=method.name, day=i, date=d.date, days_since_train=d.date - t_train_end,
                   days_since_calib=d.date - last_calib_date,
                   per=per(ey, ctx.greedy(lps)), labels_used=ctx.calib.labels_used,
                   adapt_seconds=dt, audio_seconds=sum(len(x) for x in batch.adapt_x) * 0.02,
                   **{k_: v for k_, v in diag.items() if isinstance(v, (int, float, str))})
        if with_lm_score and ctx.lm is not None:
            row["per_lm"] = per(ey, ctx.lm_decode(lps))
        rows.append(row)
        log(f"    [{method.name}] day {i:2d} (+{row['days_since_train']:2d}d) PER {row['per']:.3f} "
            f"labels {row['labels_used']:2d} {diag.get('action', '')}")
    return rows
