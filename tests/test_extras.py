import copy
import numpy as np, torch

from steadyvoice.adapt.baselines import Corp, NoAdapt
from steadyvoice.data.stream import SessionStream
from steadyvoice.eval.metrics import summarize
from steadyvoice.eval.probe import linear_probe_per
from steadyvoice.eval.runner import run_gap_jump


def test_gap_jump_ignores_intermediate_days_and_uses_no_labels(tiny):
    ctx, st, k = tiny["ctx"], tiny["states"][-1], tiny["k"]
    days = tiny["days"]
    bad = copy.deepcopy(days)
    for d in bad[k:k + 2]:                                   # days strictly between training end and target day (k+3-1)
        d.adapt_x = [x + 50 for x in d.adapt_x]; d.eval_x = [x - 9 for x in d.eval_x]
    for M in (NoAdapt, Corp):
        a = run_gap_jump(SessionStream(days), ctx, st, M(), k, gaps=(3,))
        b = run_gap_jump(SessionStream(bad), ctx, st, M(), k, gaps=(3,))
        assert a[0]["per"] == b[0]["per"] and a[0]["labels_used"] == 0 and a[0]["day"] == k + 2


def test_gap_longer_than_stream_is_skipped(tiny):
    r = run_gap_jump(SessionStream(tiny["days"]), tiny["ctx"], tiny["states"][-1], NoAdapt(), tiny["k"], gaps=(1, 99))
    assert [x["gap"] for x in r] == [1]


def test_linear_probe_returns_valid_per(tiny):
    v = linear_probe_per(tiny["ctx"], tiny["states"][-1], tiny["days"][4], steps=15)
    assert 0.0 <= v <= 1.5 and np.isfinite(v)


def test_days_gt_2x_oracle_metric():
    mk = lambda m, pers: [dict(method=m, day=i, per=p, labels_used=0, adapt_seconds=1.0, audio_seconds=1.0) for i, p in enumerate(pers)]
    rows = {"none": mk("none", [0.1, 0.5, 0.5]), "oracle": mk("oracle", [0.01, 0.01, 0.01]), "x": mk("x", [0.03, 0.03, 0.2])}
    s = {r["method"]: r for r in summarize(rows)}
    assert s["x"]["days_gt_2x_oracle"] == 1                  # only the 0.2 day exceeds 2*max(oracle, 0.02)=0.04... 0.03 does not
    assert s["none"]["days_gt_2x_oracle"] == 3
