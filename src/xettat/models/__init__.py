from __future__ import annotations

import torch.nn as nn


def build_model(cfg: dict, vocab_size: int) -> nn.Module:
    kind = cfg["type"]
    params = {k: v for k, v in cfg.items() if k != "type"}
    if kind == "bilstm":
        from xettat.models.char_bilstm import CharBiLSTM

        return CharBiLSTM(vocab_size, **params)
    if kind == "transformer":
        from xettat.models.char_transformer import CharTransformer

        return CharTransformer(vocab_size, **params)
    raise ValueError(f"unknown model type {kind!r}")
