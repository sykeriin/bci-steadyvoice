from __future__ import annotations

from typing import List

import numpy as np
import torch

from ..utils import BLANK


def greedy(logp: np.ndarray) -> List[int]:
    """logp [T,K] -> collapsed phoneme sequence."""
    a = logp.argmax(-1)
    out, prev = [], BLANK
    for t in a:
        if t != prev and t != BLANK:
            out.append(int(t))
        prev = t
    return out


@torch.no_grad()
def forward_logp(model, adapter, xs, device="cpu", bs: int = 32):
    """Frozen-model forward. Returns list of [T', K] log-prob arrays, one per utterance."""
    from ..utils import pad_batch
    out = []
    for i in range(0, len(xs), bs):
        x, l = pad_batch(xs[i:i + bs], device)
        logits, ol = model(x, l, adapter)
        lp = torch.log_softmax(logits.float(), -1).cpu().numpy()
        out += [lp[j, : int(ol[j])] for j in range(len(lp))]
    return out
