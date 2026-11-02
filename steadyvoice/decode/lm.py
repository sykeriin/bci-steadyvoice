"""Phoneme n-gram LM (Witten-Bell interpolation) + CTC prefix beam search.

v1 deviation from the plan: a phoneme-level LM replaces word-level KenLM. It needs no
G2P back-mapping, runs on CPU in pure python, and is trained ONLY on text we are
allowed to see (labelled training-session transcripts, plus optional external text with
benchmark sentences removed). A word-level KenLM can be dropped in behind the same
`logprob(history, token)` interface later.
"""
from __future__ import annotations

import math
from collections import defaultdict
from functools import lru_cache
from typing import Dict, List, Sequence, Tuple

import numpy as np

BOS, EOS = -1, -2
NEG = float("-inf")


class PhonemeNGramLM:
    def __init__(self, sentences: Sequence[Sequence[int]], n_phon: int, order: int = 3):
        self.order, self.V = order, n_phon + 1  # phonemes + EOS
        self.counts: List[Dict[tuple, Dict[int, int]]] = [defaultdict(lambda: defaultdict(int)) for _ in range(order)]
        for s in sentences:
            toks = [BOS] * (order - 1) + list(s) + [EOS]
            for i in range(order - 1, len(toks)):
                for n in range(order):  # context length n
                    self.counts[n][tuple(toks[i - n:i])][toks[i]] += 1
        self._tot = [{h: sum(c.values()) for h, c in lvl.items()} for lvl in self.counts]
        self.prob = lru_cache(maxsize=2_000_000)(self._prob)

    def _prob(self, hist: tuple, tok: int) -> float:
        p = 1.0 / self.V
        for n in range(self.order):
            h = hist[len(hist) - n:] if n else ()
            c = self.counts[n].get(h)
            if not c:
                continue
            tot, types = self._tot[n][h], len(c)
            p = (c.get(tok, 0) + types * p) / (tot + types)
        return p

    def logprob(self, prefix: Tuple[int, ...], tok: int) -> float:
        hist = ((BOS,) * (self.order - 1) + tuple(prefix))[-(self.order - 1):] if self.order > 1 else ()
        return math.log(self.prob(hist, tok))


def _lae(a: float, b: float) -> float:
    if a == NEG:
        return b
    if b == NEG:
        return a
    return a + math.log1p(math.exp(b - a)) if a >= b else b + math.log1p(math.exp(a - b))


def prefix_beam_search(logp: np.ndarray, lm: PhonemeNGramLM | None, alpha: float = 0.6,
                       beta: float = 0.3, beam: int = 8, topk: int = 6) -> List[int]:
    """CTC prefix beam search. logp: [T, K] log-probs, blank = 0. Returns a phoneme sequence."""
    beams: Dict[tuple, List[float]] = {(): [0.0, NEG]}  # prefix -> [log p_blank, log p_nonblank]
    for t in range(len(logp)):
        lp = logp[t]
        cand = [int(c) for c in np.argpartition(lp, -topk)[-topk:] if c != 0]
        nxt: Dict[tuple, List[float]] = defaultdict(lambda: [NEG, NEG])
        for pre, (pb, pnb) in beams.items():
            tot = _lae(pb, pnb)
            nxt[pre][0] = _lae(nxt[pre][0], tot + lp[0])
            for c in cand:
                p = float(lp[c])
                new = pre + (c,)
                bonus = (alpha * lm.logprob(pre, c) if lm else 0.0) + beta
                if pre and pre[-1] == c:
                    nxt[pre][1] = _lae(nxt[pre][1], pnb + p)
                    nxt[new][1] = _lae(nxt[new][1], pb + p + bonus)
                else:
                    nxt[new][1] = _lae(nxt[new][1], tot + p + bonus)
        beams = dict(sorted(nxt.items(), key=lambda kv: -_lae(*kv[1]))[:beam])
    def final(kv):
        pre, (pb, pnb) = kv
        return _lae(pb, pnb) + (alpha * lm.logprob(pre, EOS) if lm else 0.0)
    return list(max(beams.items(), key=final)[0])
