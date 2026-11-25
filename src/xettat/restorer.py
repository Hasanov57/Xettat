"""High-level inference API used by evaluation, the Docker service and export."""

from __future__ import annotations

import torch

from xettat.infer import predict_texts
from xettat.models import build_model
from xettat.text import nfc
from xettat.vocab import CharVocab


class Restorer:
    def __init__(self, model, vocab: CharVocab, max_len: int, device="cpu"):
        self.model = model.eval()
        self.vocab, self.max_len, self.device = vocab, max_len, torch.device(device)

    @classmethod
    def from_checkpoint(cls, path: str, device: str = "cpu") -> "Restorer":
        st = torch.load(path, map_location=device, weights_only=False)
        cfg = st["cfg"]
        vocab = CharVocab.from_json(st["vocab"])
        mcfg = dict(cfg["model"])
        if mcfg["type"] == "byt5":
            mcfg["pretrained"] = False  # weights come from the checkpoint; only the config is fetched
            mcfg.pop("grad_ckpt", None)
        model = build_model(mcfg, vocab.size)
        model.load_state_dict(st["model"])
        model.to(device)
        max_len = min(cfg["data"]["max_len"], cfg["model"].get("max_len", 10**9))
        return cls(model, vocab, max_len, device)

    def restore_batch(self, texts: list[str], threshold: float = 0.5) -> list[str]:
        texts = [nfc(t) for t in texts]
        # restore line by line so a single newline-heavy text is not one giant window
        flat, owners = [], []
        for ti, t in enumerate(texts):
            for line in t.split("\n"):
                flat.append(line)
                owners.append(ti)
        nonempty = [i for i, l in enumerate(flat) if l.strip()]
        preds = predict_texts(self.model, self.vocab, [flat[i] for i in nonempty], self.device,
                              self.max_len, threshold=threshold)
        for i, p in zip(nonempty, preds):
            flat[i] = p
        out = [[] for _ in texts]
        for line, ti in zip(flat, owners):
            out[ti].append(line)
        return ["\n".join(lines) for lines in out]

    def restore(self, text: str, threshold: float = 0.5) -> str:
        return self.restore_batch([text], threshold)[0]
