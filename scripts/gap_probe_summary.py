"""Summarise day-gap jump results and linear-probe diagnostics from results/final/*.json (mean +/- std over seeds)."""
import glob, json, sys
from collections import defaultdict
import numpy as np

d = sys.argv[1] if len(sys.argv) > 1 else "results/final"
runs = [json.load(open(f)) for f in sorted(glob.glob(f"{d}/results_k*_s*.json"))]
gap = defaultdict(lambda: defaultdict(list)); lab = defaultdict(lambda: defaultdict(list))
for r in runs:
    for m, rows in r.get("gap_rows", {}).items():
        for x in rows:
            gap[m][x["gap"]].append(x["per"]); lab[m][x["gap"]].append(x["labels_used"])
gaps = sorted({g for m in gap for g in gap[m]})
out = [f"Day-gap jump: adapt ONCE directly on day k+g-1 (intermediate days unseen). PER mean ± std over {len(runs)} seeds (labels used in brackets).\n",
       "| method | " + " | ".join(f"g={g}" for g in gaps) + " |", "|---|" + "---|" * len(gaps)]
for m in gap:
    out.append(f"| {m} | " + " | ".join(f"{np.mean(gap[m][g]):.3f} ± {np.std(gap[m][g]):.3f} [{np.mean(lab[m][g]):.0f}]" for g in gaps) + " |")
out += ["\nLinear probe (analysis only, uses labels): fresh linear CTC head refit on the day's labelled trials, on the adapted latents.",
        "probe << frozen-head PER => information preserved, readout misaligned; probe high => information destroyed.\n",
        "| method | frozen-head PER | probe PER | days |", "|---|---|---|---|"]
for m in ["none", "steadyvoice_noguard", "steadyvoice"]:
    fr, pr = [], []
    for r in runs:
        for x in r["rows"].get(m, []):
            if "probe_per" in x: fr.append(x["per"]); pr.append(x["probe_per"])
    if fr:
        out.append(f"| {m} | {np.mean(fr):.3f} | {np.mean(pr):.3f} | {len(fr)} |")
open(f"{d}/gap_probe.md", "w").write("\n".join(out)); print("\n".join(out))
