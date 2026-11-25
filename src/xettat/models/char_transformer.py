from __future__ import annotations

import torch
import torch.nn as nn


class CharTransformer(nn.Module):
    """Character embedding + learned positions -> pre-norm Transformer encoder -> logit."""

    def __init__(
        self, vocab_size: int, d_model: int = 256, heads: int = 4, layers: int = 6,
        ff: int = 1024, dropout: float = 0.1, max_len: int = 512,
    ):
        super().__init__()
        self.max_len = max_len
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, heads, ff, dropout=dropout, batch_first=True, norm_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        T = ids.size(1)
        if T > self.max_len:
            raise ValueError(f"sequence length {T} > max_len {self.max_len}")
        pos = torch.arange(T, device=ids.device).unsqueeze(0)
        x = self.emb(ids) + self.pos(pos)
        x = self.encoder(x, src_key_padding_mask=~mask)
        return self.head(self.norm(x)).squeeze(-1)
