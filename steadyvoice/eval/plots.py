from __future__ import annotations

from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STYLE = {"none": ("#888888", "--"), "oracle": ("#1b9e77", "-"), "coral": ("#7570b3", "-"),
         "tent": ("#e6ab02", ":"), "corp": ("#d95f02", "-"), "steadyvoice": ("#1f78b4", "-"),
         "steadyvoice_noguard": ("#a6cee3", "-")}


def plot_per_vs_days(rows_by_method: Dict[str, List[Dict]], path: str, key: str = "per",
                     title: str = "") -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for m, rows in rows_by_method.items():
        c, ls = STYLE.get(m, (None, "-"))
        ax.plot([r["days_since_train"] for r in rows], [r[key] for r in rows], ls, color=c,
                marker="o", ms=3, lw=1.6, label=m)
    ax.set_xlabel("days since last labelled calibration (training end)")
    ax.set_ylabel("phoneme error rate (LM off)" if key == "per" else key)
    ax.set_ylim(0, 1); ax.grid(alpha=.3); ax.legend(frameon=False, fontsize=8)
    ax.set_title(title, fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
