import copy, dataclasses
import numpy as np, torch

from steadyvoice.data.types import DayBatch
from steadyvoice.decode.lm import PhonemeNGramLM, prefix_beam_search, EOS
from steadyvoice.models.adapters import make_adapter
from steadyvoice.models.backbone import CausalGRUDecoder


def test_adapters_identity_at_init():
    x = torch.randn(2, 7, 12)
    for kind in ("affine", "diag", "lowrank"):
        assert torch.allclose(make_adapter(kind, 12, 4)(x), x, atol=1e-6), kind


def test_backbone_is_causal():
    m = CausalGRUDecoder(8, 16, 2, 41, patch=4, stride=2, dropout=0.0).eval()
    x = torch.randn(1, 40, 8); l = torch.tensor([40])
    y1, _ = m(x, l)
    x2 = x.clone(); x2[:, 25:] += torch.randn_like(x2[:, 25:]) * 5   # change the future only
    y2, _ = m(x2, l)
    n_safe = 25 // 2  # frames whose receptive field ends before bin 25
    assert torch.allclose(y1[:, :n_safe], y2[:, :n_safe], atol=1e-5)
    assert not torch.allclose(y1[:, -3:], y2[:, -3:], atol=1e-3)


def test_lm_normalises_and_beam_decodes():
    sents = [[1, 2, 3], [1, 2, 3], [1, 2, 4], [3, 2, 1]] * 20
    lm = PhonemeNGramLM(sents, n_phon=5)
    for hist in [(), (1,), (1, 2)]:
        tot = sum(lm.logprob(hist, t) and np.exp(lm.logprob(hist, t)) for t in [1, 2, 3, 4, 5, EOS])
        assert abs(tot - 1) < 1e-6
    T, K = 12, 6
    lp = np.full((T, K), np.log(0.01)); seq = [1, 0, 2, 0, 3]
    for t, s in zip(range(0, 10, 2), seq): lp[t, s] = np.log(0.95)
    lp[1::2, 0] = np.log(0.95); lp = lp - np.logaddexp.reduce(lp, 1, keepdims=True)
    assert prefix_beam_search(lp, lm) == [1, 2, 3]


def test_synthetic_drift_changes_features(tiny):
    d = tiny["days"]
    f = lambda i: np.concatenate(d[i].adapt_x).mean(0)
    assert np.linalg.norm(f(0) - f(7)) > 2 * np.linalg.norm(f(0) - f(1)) * 0.5 and np.linalg.norm(f(0) - f(7)) > 0.1


def test_coral_matches_reference_stats(tiny):
    from steadyvoice.adapt.baselines import coral_state
    ctx, days, states = tiny["ctx"], tiny["days"], tiny["states"]
    st = coral_state(ctx, states[-1], days[5].adapt_x, shrink=0.0)
    ad = ctx.new_adapter(st)
    with torch.no_grad():
        f = torch.cat([ad(torch.from_numpy(x)) for x in days[5].adapt_x]).double().numpy()
    assert np.allclose(f.mean(0), ctx.ref.feat_mu, atol=1e-3)
    assert np.allclose(np.cov(f.T), ctx.ref.feat_cov, atol=1e-2 * np.abs(ctx.ref.feat_cov).max() + 1e-3)


# ---- leakage -------------------------------------------------------------------------------
def test_daybatch_has_no_labels():
    assert {f.name for f in dataclasses.fields(DayBatch)} == {"session_id", "date", "adapt_x"}


def test_adaptation_code_never_touches_the_vault():
    import glob, os
    root = os.path.join(os.path.dirname(__file__), "..", "steadyvoice")
    for p in glob.glob(f"{root}/adapt/*.py") + glob.glob(f"{root}/guard/*.py"):
        src = open(p).read().lower()
        assert "vault" not in src and "eval_labels" not in src and "adapt_labels" not in src, p


class _Spy:
    labels_used = 0
    def request(self, n): raise AssertionError("label-free method requested labels")


def _methods():
    from steadyvoice.adapt.baselines import Coral, Corp, NoAdapt, Tent
    from steadyvoice.adapt.steadyvoice import SteadyVoice
    return [NoAdapt(), Coral(), Tent(), Corp(), SteadyVoice(None, name="sv")]


def test_label_free_methods_never_request_labels(tiny):
    ctx, days, st = tiny["ctx"], tiny["days"], tiny["states"][-1]
    for m in _methods():
        ctx.calib = _Spy(); m.start(ctx, st)
        m.adapt(ctx, st, DayBatch(4, days[4].date, tuple(days[4].adapt_x)))


def test_day_d_state_depends_only_on_past(tiny):
    """Corrupt every future day's features AND all labels; day-d output must be bit-identical."""
    from steadyvoice.data.stream import SessionStream
    from steadyvoice.adapt.baselines import Corp, Coral
    ctx, st = tiny["ctx"], tiny["states"][-1]
    days = tiny["days"]
    def run(ds, method):
        s = SessionStream(ds); cur = st
        for i in (3, 4):
            ctx.calib = _Spy(); cur, _ = method.adapt(ctx, cur, s.batch(i))
        return cur
    bad = copy.deepcopy(days)
    for d in bad[5:]:
        d.adapt_x = [x + 99 for x in d.adapt_x]; d.eval_x = [x - 7 for x in d.eval_x]
    for d in bad:
        d.adapt_y = [[1] for _ in d.adapt_y]; d.eval_y = [[2] for _ in d.eval_y]
    for M in (Corp, Coral):
        a, b = run(days, M()), run(bad, M())
        assert all(torch.equal(a[k], b[k]) for k in a), M.__name__


# ---- guard ----------------------------------------------------------------------------------
def test_guard_rollback_and_calibration(monkeypatch):
    from types import SimpleNamespace
    import steadyvoice.guard.policy as pol

    mk = lambda v: {"b": torch.tensor([float(v)])}
    monkeypatch.setattr(pol, "compute_signals", lambda ctx, s, xs: {"v": float(s["b"][0])})
    monkeypatch.setattr(pol, "supervised_fit", lambda ctx, s, xl, yl: mk(0.05))
    pred = SimpleNamespace(predict=lambda sig: sig["v"])
    calib = SimpleNamespace(labels_used=0)
    def req(n): calib.labels_used += n; return [None] * n, [None] * n
    calib.request = req
    ctx = SimpleNamespace(hp=dict(margin=0.02, tau_cal=0.30, tau_ok=0.15, oracle_n=20), calib=calib)
    g = pol.Guard(pred); g.start(mk(0.05))                       # last_good = healthy (0.05)

    s, info = g.step(ctx, [], carry=mk(0.10), cand=mk(0.05)); assert info["action"] == "accept"
    s, info = g.step(ctx, [], carry=mk(0.06), cand=mk(0.06)); assert info["action"] == "reject"   # no one beats carry by the margin
    s, info = g.step(ctx, [], carry=mk(0.25), cand=mk(0.28)); assert info["action"] == "rollback"   # last_good (0.05) wins
    assert abs(float(s["b"][0]) - 0.06) < 1e-6 and calib.labels_used == 0   # last_good tracked the healthy carry-over
    g.last_good = mk(0.6)                                         # nothing healthy any more
    s, info = g.step(ctx, [], carry=mk(0.5), cand=mk(0.45)); assert info["action"] == "calibrate"
    assert calib.labels_used == 20                                # and it is charged
