from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from xettat.models.common import _out


class CharBiLSTM(nn.Module):
    """Character embedding -> stacked BiLSTM -> per-character logit (convert?)."""

    def __init__(self, vocab_size: int, emb_dim: int = 128, hidden: int = 256, layers: int = 3, dropout: float = 0.2,
                 num_classes: int = 1):
        super().__init__()
        self.num_classes = num_classes
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            emb_dim, hidden, num_layers=layers, bidirectional=True, batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(2 * hidden, num_classes)

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.emb(ids))
        if torch.onnx.is_in_onnx_export():
            # packed sequences don't export cleanly; the browser runs BiLSTM one line at a time (no padding)
            out, _ = self.lstm(x)
            logits = _out(self.head(self.drop(out)), self.num_classes)
            m = mask if self.num_classes == 1 else mask.unsqueeze(-1)
            return logits.masked_fill(~m, -1e4)  # uses `mask`, so ONNX keeps it as a graph input
        lengths = mask.sum(1).clamp(min=1).cpu()
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=ids.size(1))
        return _out(self.head(self.drop(out)), self.num_classes)
