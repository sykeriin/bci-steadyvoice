"""Aggregate per-seed results into a mean +/- std table (Markdown) and a per-seed paired comparison."""
import glob, json, sys
import numpy as np

d_ = d = sys.argv[1] if len(sys.argv) > 1 else "results/synthetic"
files = sorted(glob.glob(f"{d}/results_k*_s*.json"))
runs = [json.load(open(f))["summary"] for f in files]
methods = [s["method"] for s in runs[0]]
cols = [("mean_per", "mean PER"), ("mean_per_moderate", "PER moderate-drift days"), ("mean_per_severe", "PER severe-drift days"),
        ("worst_per", "worst-day PER"), ("days_gt_2x_oracle", "days >2x oracle"), ("labels_used", "labels used"), ("collapse_events", "collapse events"),
        ("rollback_or_reject", "reject/rollback days")]
out = [f"Seeds: {len(runs)} ({', '.join(f.split('_s')[-1][:-5] for f in files)}). mean ± std across seeds. LM-off phoneme error rate.\n",
       "| method | " + " | ".join(c[1] for c in cols) + " |", "|---|" + "---|" * len(cols)]
get = lambda r, m, k: next(s for s in r if s["method"] == m).get(k, float("nan"))
for m in methods:
    cells = []
    for k, _ in cols:
        v = np.array([get(r, m, k) for r in runs], float)
        cells.append(f"{np.nanmean(v):.3f} ± {np.nanstd(v):.3f}" if k.endswith("per") or "per_" in k else f"{np.nanmean(v):.1f} ± {np.nanstd(v):.1f}")
    out.append(f"| {m} | " + " | ".join(cells) + " |")
out.append("\nPaired per-seed mean PER (rows = seeds):\n")
out.append("| seed | " + " | ".join(methods) + " |"); out.append("|---|" + "---|" * len(methods))
for f, r in zip(files, runs):
    out.append(f"| {f.split('_s')[-1][:-5]} | " + " | ".join(f"{get(r, m, 'mean_per'):.3f}" for m in methods) + " |")
open(f"{d}/summary.md", "w").write("\n".join(out)); print("\n".join(out))


# ---- paired bootstrap over seeds: mean PER difference (method - steadyvoice) --------------------------------
rng = np.random.default_rng(0)
if "steadyvoice" in methods:
    ref = np.array([get(r, "steadyvoice", "mean_per") for r in runs])
    lines = ["\nPaired difference in mean PER vs guarded SteadyVoice (positive = method worse). "
             "95% bootstrap CI over seeds; with few seeds this is coarse.\n", "| method | mean diff | 95% CI | seeds worse / total |", "|---|---|---|---|"]
    for m in methods:
        if m == "steadyvoice":
            continue
        d = np.array([get(r, m, "mean_per") for r in runs]) - ref
        bs = [rng.choice(d, len(d)).mean() for _ in range(5000)]
        lines.append(f"| {m} | {d.mean():+.3f} | [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}] | {(d > 0).sum()}/{len(d)} |")
    open(f"{d_}/summary.md", "a").write("\n".join(lines)); print("\n".join(lines))
