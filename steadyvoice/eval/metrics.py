from __future__ import annotations

from typing import Dict, List

import numpy as np


def summarize(rows_by_method: Dict[str, List[Dict]], baseline: str = "none", n_cal: int = 20) -> List[Dict]:
    base = {r["day"]: r["per"] for r in rows_by_method[baseline]}
    out = []
    for m, rows in rows_by_method.items():
        pers = np.array([r["per"] for r in rows])
        prev = np.r_[np.nan, pers[:-1]]
        # collapse: >2x worse than no adaptation that day, and >5 PER points worse in absolute terms
        coll = sum(1 for r in rows if r["per"] > 2 * base[r["day"]] and r["per"] - base[r["day"]] > 0.05)
        acts = [r.get("action", "") for r in rows]
        used = sum(1 for r in rows if r.get("action") in ("calibrate",)) if m != "oracle" else len(rows)
        mod = [r["per"] for r in rows if base[r["day"]] < 0.30]   # 'moderate drift' days (post-hoc, analysis only)
        sev = [r["per"] for r in rows if base[r["day"]] >= 0.30]
        out.append(dict(
            method=m, mean_per=float(pers.mean()),
            mean_per_moderate=float(np.mean(mod)) if mod else float("nan"),
            mean_per_severe=float(np.mean(sev)) if sev else float("nan"), worst_per=float(pers.max()),
            final_third_per=float(pers[len(pers) * 2 // 3:].mean()),
            mean_per_lm=float(np.mean([r["per_lm"] for r in rows])) if "per_lm" in rows[0] else float("nan"),
            collapse_events=coll,
            calib_days=sum(1 for r in rows if r["labels_used"] > 0),
            labels_used=int(sum(r["labels_used"] for r in rows)),
            labels_saved_vs_daily=int(n_cal * len(rows) - sum(r["labels_used"] for r in rows)),
            rollback_or_reject=sum(a in ("reject", "rollback") for a in acts),
            rtf=float(sum(r["adapt_seconds"] for r in rows) / sum(r["audio_seconds"] for r in rows)),
        ))
    return out
