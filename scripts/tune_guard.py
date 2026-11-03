"""Tune guard (margin, tau_cal) on DEV seeds only (100-102); the reporting seeds (0-9) are never used here.
Objective fixed BEFORE looking at results:  cost = mean_PER + 0.05 * labels_used / (20 * n_days)
(i.e. spending one calibration every day costs +0.05 PER-equivalents). Writes results/tune/best.json. Resumable."""
import itertools, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, torch

from steadyvoice.adapt.common import DEFAULT_HP
from steadyvoice.data.synthetic import SynthConfig, make_dataset
from steadyvoice.eval.experiment import build_method
from steadyvoice.eval.experiment import run_experiment
from steadyvoice.eval.runner import run_forward
from steadyvoice.data.stream import SessionStream
from steadyvoice.train.backbone_train import TrainCfg

DEV, K, OUT = [100, 101, 102], 5, "results/tune"
GRID = list(itertools.product([0.0, 0.02, 0.05], [0.2, 0.3, 0.4]))   # margin x tau_cal
os.makedirs(OUT, exist_ok=True)
torch.set_num_threads(4)
for seed in DEV:
    f = f"{OUT}/dev_s{seed}.json"
    if os.path.exists(f):
        continue
    cfg = SynthConfig(n_days=26, n_train_days=K, seed=seed)
    world, days = make_dataset(cfg)
    rows, _, ctx, pred, states = run_experiment(
        days, cfg.n_phon, K, TrainCfg(steps=1500, seed=seed), ["none"], world.corpus(3000, 123), OUT,
        f"dev_base_s{seed}", "cpu", log=lambda s: None, return_ctx=True)
    stream, res = SessionStream(days), []
    for margin, tau_cal in GRID:
        ctx.hp = {**DEFAULT_HP, "margin": margin, "tau_cal": tau_cal}
        r = run_forward(stream, ctx, states[-1], build_method("steadyvoice", pred), K, log=lambda s: None)
        mp, lab = float(np.mean([x["per"] for x in r])), sum(x["labels_used"] for x in r)
        res.append(dict(margin=margin, tau_cal=tau_cal, mean_per=mp, labels=lab, cost=mp + 0.05 * lab / (20 * len(r))))
        print(seed, res[-1], flush=True)
    json.dump(res, open(f, "w"))
allr = [json.load(open(f"{OUT}/dev_s{s}.json")) for s in DEV]
best = min(range(len(GRID)), key=lambda i: np.mean([r[i]["cost"] for r in allr]))
summ = [dict(margin=GRID[i][0], tau_cal=GRID[i][1], cost=float(np.mean([r[i]["cost"] for r in allr])),
             mean_per=float(np.mean([r[i]["mean_per"] for r in allr])), labels=float(np.mean([r[i]["labels"] for r in allr])))
        for i in range(len(GRID))]
json.dump(dict(best=summ[best], grid=summ, default=dict(margin=DEFAULT_HP["margin"], tau_cal=DEFAULT_HP["tau_cal"])),
          open(f"{OUT}/best.json", "w"), indent=1)
print("BEST (dev seeds):", summ[best]); [print(" ", s) for s in summ]
