"""Run the forward-drift experiment on real T15 data (run this on your RTX 3050).

  python scripts/run_real.py --root /path/to/t15_copyTask_neuralData/hdf5_data_final --k 5

Backbone is retrained from scratch on the first k sessions ONLY. Do NOT use the released all-session
weights for drift curves: they were trained on every session's labels (leakage).
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch

from steadyvoice.data.t15 import load_t15
from steadyvoice.eval.experiment import run_experiment
from steadyvoice.train.backbone_train import TrainCfg

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--k", type=int, default=5)
ap.add_argument("--max-sessions", type=int, default=None)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--hidden", type=int, default=512)      # 3050-friendly; paper-size is 768
ap.add_argument("--layers", type=int, default=4)
ap.add_argument("--steps", type=int, default=20000)
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--methods", default="none,oracle,periodic7,coral,corp,steadyvoice_noguard,guard_only,steadyvoice")
ap.add_argument("--out", default="results/t15")
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
days = load_t15(a.root, a.max_sessions)
print(f"{len(days)} sessions, device={dev}")
tcfg = TrainCfg(hidden=a.hidden, layers=a.layers, patch=14, stride=4, dropout=0.3, steps=a.steps,
                bs=a.bs, lr=1e-3, noise=0.8, seed=a.seed, amp=(dev == "cuda"))
run_experiment(days, 40, a.k, tcfg, a.methods.split(","), [], a.out, f"t15_k{a.k}_s{a.seed}", dev)
