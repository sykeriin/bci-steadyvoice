"""Single entry point for synthetic experiments (supersedes run_synthetic.py). Resumable (skips finished seeds).
Guard params (margin, tau_cal) come from results/tune/best.json (dev seeds 100-102) unless --no-tuned."""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch

from steadyvoice.data.synthetic import SynthConfig, make_dataset
from steadyvoice.eval.experiment import run_experiment
from steadyvoice.train.backbone_train import TrainCfg

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", default="0")
ap.add_argument("--k", type=int, default=5)
ap.add_argument("--days", type=int, default=26)
ap.add_argument("--steps", type=int, default=1500)
ap.add_argument("--drift", default="linear")
ap.add_argument("--natural-only", action="store_true")
ap.add_argument("--no-tuned", action="store_true")
ap.add_argument("--gaps", default="1,5,10,20")
ap.add_argument("--probe", default="none,steadyvoice_noguard,steadyvoice")
ap.add_argument("--methods", default="none,oracle,periodic7,corp,steadyvoice_noguard,guard_only,steadyvoice")
ap.add_argument("--min-rho", type=float, default=0.5, help="predictor reliability gate; -1 disables the fallback (control)")
ap.add_argument("--out", default="results/final")
a = ap.parse_args()
torch.set_num_threads(4)
hp = {}
if not a.no_tuned and os.path.exists("results/tune/best.json"):
    b = json.load(open("results/tune/best.json"))["best"]
    hp = dict(margin=b["margin"], tau_cal=b["tau_cal"])
print("guard hp override:", hp or "(defaults)", flush=True)
for seed in map(int, a.seeds.split(",")):
    f = f"{a.out}/results_k{a.k}_s{seed}.json"
    if os.path.exists(f):
        print("skip", f); continue
    cfg = SynthConfig(n_days=a.days, n_train_days=a.k, seed=seed, drift_family=a.drift)
    world, days = make_dataset(cfg)
    run_experiment(days, cfg.n_phon, a.k, TrainCfg(steps=a.steps, seed=seed), a.methods.split(","),
                   world.corpus(3000, 123), a.out, f"k{a.k}_s{seed}", "cpu", n_perturb=0 if a.natural_only else 80,
                   gaps=tuple(int(g) for g in a.gaps.split(",") if g), probe_methods=tuple(a.probe.split(",")),
                   hp_override=hp, min_rho=a.min_rho)
    print("finished seed", seed, flush=True)
