"""Brain-to-Text '25 (T15, Card et al. NEJM 2024) loader.

!!! UNVERIFIED AGAINST REAL DATA !!!  The Dryad download was blocked in the build environment
(403/401), so this follows the layout I recall from the public baseline repo
(Neuroprosthetics-Lab/nejm-brain-to-text) and is only tested against a FAKE file with that
assumed schema (tests/test_t15_loader.py). Run `python scripts/inspect_hdf5.py <root>` first;
the loader raises a clear error if the schema differs, rather than guessing.

Assumed layout:  <root>/<session_name>/data_{train,val,test}.hdf5, session_name like 't15.2023.08.11'
  group 'trial_0000' -> dataset 'input_features' [T, 512] float32,
                        dataset 'seq_class_ids' [<=500] int (0 = blank/pad), attr 'seq_len'
Phoneme ids 1..40 (40 = SIL), 0 = CTC blank  => 41 classes, matching steadyvoice's decoder.

Protocol: per session, `adapt` trials = train split (labels sealed in the vault; unlabelled to methods),
`eval` trials = val split (scored only after adaptation). Test split is excluded in v1 (labels hidden).
"""
from __future__ import annotations

import datetime as dt
import glob
import os
import re
from typing import List

import h5py
import numpy as np

from .types import DayData


def _parse_date(name: str) -> dt.date:
    m = re.search(r"(\d{4})\.(\d{2})\.(\d{2})", name)
    if not m:
        raise ValueError(f"cannot parse a date from session folder name {name!r}")
    return dt.date(*map(int, m.groups()))


def _read_split(path: str):
    xs, ys = [], []
    with h5py.File(path, "r") as f:
        for key in sorted(f.keys()):
            g = f[key]
            for need in ("input_features", "seq_class_ids"):
                if need not in g:
                    raise KeyError(f"{path}:{key} has no dataset {need!r}; found {list(g.keys())}. "
                                   "Schema differs from the assumed one - run scripts/inspect_hdf5.py")
            n = int(g.attrs["seq_len"]) if "seq_len" in g.attrs else int((g["seq_class_ids"][:] > 0).sum())
            xs.append(np.asarray(g["input_features"][:], dtype=np.float32))
            ys.append([int(v) for v in g["seq_class_ids"][:n]])
    return xs, ys


def load_t15(root: str, max_sessions: int | None = None) -> List[DayData]:
    sess = sorted(glob.glob(os.path.join(root, "t15.*")), key=lambda p: _parse_date(os.path.basename(p)))
    if not sess:
        raise FileNotFoundError(f"no t15.* session folders under {root}")
    days, d0 = [], _parse_date(os.path.basename(sess[0]))
    for i, p in enumerate(sess[:max_sessions]):
        tr, va = os.path.join(p, "data_train.hdf5"), os.path.join(p, "data_val.hdf5")
        if not (os.path.exists(tr) and os.path.exists(va)):
            continue                                   # some sessions have no val split
        ax, ay = _read_split(tr)
        ex, ey = _read_split(va)
        days.append(DayData(i, (_parse_date(os.path.basename(p)) - d0).days, ax, ay, ex, ey))
    return days
