from __future__ import annotations

import torch
import torch.nn as nn

from xettat.vocab import CharVocab

BYTE_OFFSET = 3  # ByT5: 0 pad, 1 eos, 2 unk, bytes start at 3
UNK = 2


class ByT5Tagger(nn.Module):
    """Pretrained ByT5 encoder + per-character head, over CharVocab ids.

    Each character is expanded to its UTF-8 bytes inside the model (vectorised), the encoder runs on
    bytes, and every character reads the hidden state of its first byte. Same contract as the char models:
    forward(ids[B,T], mask[B,T]) -> logits[B,T].
    """

    def __init__(self, vocab_size: int, name: str = "google/byt5-small", dropout: float = 0.1,
                 pretrained: bool = True, config: dict | None = None, grad_ckpt: bool = False):
        super().__init__()
        from transformers import T5Config, T5EncoderModel

        if pretrained:
            self.encoder = T5EncoderModel.from_pretrained(name)
        else:
            cfg = T5Config(**config) if config else T5Config.from_pretrained(name)
            self.encoder = T5EncoderModel(cfg)
        if grad_ckpt:
            self.encoder.gradient_checkpointing_enable()
        d = self.encoder.config.d_model
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d, 1)

        vocab = CharVocab()
        assert vocab.size == vocab_size, "ByT5Tagger expects the default CharVocab"
        byte_len = torch.zeros(vocab_size, dtype=torch.long)
        byte_tab = torch.zeros(vocab_size, 4, dtype=torch.long)
        byte_len[CharVocab.UNK], byte_tab[CharVocab.UNK, 0] = 1, UNK
        for ch, i in vocab.index.items():
            b = ch.encode("utf-8")
            byte_len[i] = len(b)
            byte_tab[i, : len(b)] = torch.tensor([x + BYTE_OFFSET for x in b])
        self.register_buffer("byte_len", byte_len, persistent=False)
        self.register_buffer("byte_tab", byte_tab, persistent=False)

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        lens = self.byte_len[ids] * mask.long()                      # [B,T] bytes per char
        offs = lens.cumsum(1) - lens                                 # first-byte position per char
        totals = lens.sum(1)                                         # [B]
        L = max(int(totals.max().item()), 1)
        byte_ids = torch.zeros(B, L, dtype=torch.long, device=ids.device)
        rows = torch.arange(B, device=ids.device).unsqueeze(1).expand(B, T)
        tab = self.byte_tab[ids]                                     # [B,T,4]
        for k in range(4):
            valid = lens > k
            byte_ids[rows[valid], (offs + k)[valid]] = tab[..., k][valid]
        attn = torch.arange(L, device=ids.device).unsqueeze(0) < totals.unsqueeze(1)
        h = self.encoder(input_ids=byte_ids, attention_mask=attn.long()).last_hidden_state  # [B,L,d]
        idx = offs.clamp(max=L - 1).unsqueeze(-1).expand(-1, -1, h.size(-1))
        g = h.gather(1, idx)                                         # [B,T,d]
        return self.head(self.drop(g)).squeeze(-1)
