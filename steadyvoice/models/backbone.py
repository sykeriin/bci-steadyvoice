"""Causal (unidirectional) GRU + CTC phoneme decoder with causal patching."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalGRUDecoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, layers: int = 2, n_classes: int = 41,
                 patch: int = 4, stride: int = 2, dropout: float = 0.1):
        super().__init__()
        self.patch, self.stride, self.in_dim = patch, stride, in_dim
        self.proj = nn.Linear(patch * in_dim, hidden)
        self.drop = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden, hidden, layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, n_classes)

    def out_lengths(self, lengths):
        return (lengths - 1) // self.stride + 1

    def _patchify(self, x):
        # left-pad so frame j only sees bins <= j*stride  => causal
        x = F.pad(x, (0, 0, self.patch - 1, 0))
        return x.unfold(1, self.patch, self.stride).permute(0, 1, 3, 2).flatten(2)

    def forward(self, x, lengths, adapter=None, return_latent=False):
        if adapter is not None:
            x = adapter(x)
        h = self.drop(torch.nn.functional.gelu(self.proj(self._patchify(x))))
        h, _ = self.gru(h)
        logits = self.head(self.drop(h))
        ol = self.out_lengths(lengths)
        return (logits, ol, h) if return_latent else (logits, ol)


def freeze_for_adaptation(model: CausalGRUDecoder) -> CausalGRUDecoder:
    """Freeze weights and turn dropout off but stay in train() mode (cuDNN GRU backward needs it)."""
    model.train()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = 0.0
    model.gru.dropout = 0.0
    for p in model.parameters():
        p.requires_grad_(False)
    return model
