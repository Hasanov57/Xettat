from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from xettat.models.common import _out


class CharTransformer(nn.Module):
    """Character embedding + learned positions -> pre-norm Transformer encoder -> logit."""

    def __init__(
        self, vocab_size: int, d_model: int = 256, heads: int = 4, layers: int = 6,
        ff: int = 1024, dropout: float = 0.1, max_len: int = 512, num_classes: int = 1,
    ):
        super().__init__()
        self.max_len = max_len
        self.num_classes = num_classes
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, heads, ff, dropout=dropout, batch_first=True, norm_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        T = ids.size(1)
        if T > self.max_len:
            raise ValueError(f"sequence length {T} > max_len {self.max_len}")
        pos = torch.arange(T, device=ids.device).unsqueeze(0)
        x = self.emb(ids) + self.pos(pos)
        if torch.onnx.is_in_onnx_export():
            x = self._export_encoder(x, mask)
        else:
            x = self.encoder(x, src_key_padding_mask=~mask)
        return _out(self.head(self.norm(x)), self.num_classes)

    def _export_encoder(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Same math as the pre-norm nn.TransformerEncoderLayer stack (same parameters), written so the
        ONNX tracer keeps batch and sequence length dynamic (nn.MultiheadAttention bakes them in)."""
        attn_mask = mask[:, None, None, :]  # [B,1,1,T], True = attend
        for layer in self.encoder.layers:
            sa = layer.self_attn
            h = layer.norm1(x)
            q, k, v = F.linear(h, sa.in_proj_weight, sa.in_proj_bias).chunk(3, dim=-1)
            q, k, v = (t.unflatten(-1, (sa.num_heads, sa.head_dim)).transpose(1, 2) for t in (q, k, v))
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
            x = x + sa.out_proj(o.transpose(1, 2).flatten(2))
            x = x + layer.linear2(layer.activation(layer.linear1(layer.norm2(x))))
        return x
