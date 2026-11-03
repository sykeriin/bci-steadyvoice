"""End-to-end synthetic experiment: train backbone on first k days, forward-simulate all methods."""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, torch

from steadyvoice.adapt.baselines import Coral, Corp, GuardedNoAdapt, NoAdapt, Oracle, PeriodicOracle, Tent
from steadyvoice.adapt.common import DEFAULT_HP
from steadyvoice.adapt.common import Ctx, compute_reference
from steadyvoice.adapt.steadyvoice import SteadyVoice
from steadyvoice.data.stream import SessionStream
from steadyvoice.data.synthetic import SynthConfig, make_dataset
from steadyvoice.decode.lm import PhonemeNGramLM
from steadyvoice.eval.metrics import summarize
from steadyvoice.eval.plots import plot_per_vs_days
from steadyvoice.eval.runner import run_forward
from steadyvoice.guard.policy import Guard
from steadyvoice.guard.predictor import fit_predictor
from steadyvoice.models.backbone import freeze_for_adaptation
from steadyvoice.train.backbone_train import TrainCfg, train_backbone

ap = argparse.ArgumentParser()
ap.add_argument("--k", type=int, default=5)
ap.add_argument("--days", type=int, default=26)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--steps", type=int, default=1500)
ap.add_argument("--methods", default="none,oracle,coral,tent,corp,steadyvoice_noguard,steadyvoice")
ap.add_argument("--drift", default="linear")
ap.add_argument("--natural-only", action="store_true", help="fit guard predictor on natural cross-day pairs only")
ap.add_argument("--out", default="results/synthetic")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
torch.set_num_threads(4)
dev = "cuda" if torch.cuda.is_available() else "cpu"

cfg = SynthConfig(n_days=a.days, n_train_days=a.k, seed=a.seed, drift_family=a.drift)
world, days = make_dataset(cfg)
train_days = days[: a.k]
print(f"[1/4] backbone on days 0..{a.k-1} (labelled), dates {[d.date for d in train_days]}")
model, states, _ = train_backbone(train_days, cfg.n_phon + 1, TrainCfg(steps=a.steps, seed=a.seed), dev)
freeze_for_adaptation(model)

# LM trained ONLY on text we may see: world-language corpus + labelled training-day transcripts
lm = PhonemeNGramLM(world.corpus(3000, seed=123) + [y for d in train_days for y in d.adapt_y], cfg.n_phon)
ctx = Ctx(model, cfg.C, "affine", 16, dev, lm)
ctx.ref = compute_reference(ctx, train_days, states)
print("[2/4] fitting guard predictor on labelled TRAINING days only")
pred, pdiag = fit_predictor(ctx, train_days, states, n_perturb=0 if a.natural_only else 80, seed=a.seed)

stream = SessionStream(days)
init = states[-1]
print("[3/4] forward simulation")
rows = {}
for m in a.methods.split(","):
    simple = {"none": NoAdapt, "oracle": Oracle, "coral": Coral, "tent": Tent, "corp": Corp}
    abl = {"sv_nowarm": dict(sv_coral_warm=False), "sv_nopl": dict(use_pl=False),
           "sv_noal": dict(use_al=False), "sv_nocons": dict(use_cons=False)}
    if m in simple: method = simple[m]()
    elif m == "guard_only": method = GuardedNoAdapt(Guard(pred))
    elif m.startswith("periodic"): method = PeriodicOracle(int(m[8:]))
    elif m in abl: method = SteadyVoice(None, name=m, **abl[m])       # label-free core, no guard
    else: method = SteadyVoice(Guard(pred) if m == "steadyvoice" else None, name=m)
    ctx.hp = dict(DEFAULT_HP)
    rows[m] = run_forward(stream, ctx, init, method, a.k, log=lambda s: None)
    r = rows[m]
    print(f"  {m:22s} mean PER {np.mean([x['per'] for x in r]):.3f}  worst {max(x['per'] for x in r):.3f}  "
          f"labels {sum(x['labels_used'] for x in r)}")
summ = summarize(rows)
json.dump({"rows": rows, "summary": summ, "guard_signal_spearman": pdiag, "args": vars(a)},
          open(f"{a.out}/results_k{a.k}_s{a.seed}.json", "w"), indent=1)
plot_per_vs_days(rows, f"{a.out}/per_vs_days_k{a.k}_s{a.seed}.png",
                 title=f"Synthetic drift, k={a.k}, seed={a.seed} (LM-off PER)")
print("[4/4] summary"); [print(" ", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()}) for s in summ]
