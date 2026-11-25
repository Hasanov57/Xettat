"""Batched inference with sliding windows for inputs longer than the model's max_len."""

from __future__ import annotations

import torch

from xettat.text import CANDIDATES, TO_AZ
from xettat.vocab import CharVocab


def windows(n: int, max_len: int, overlap: int) -> list[tuple[int, int]]:
    if n <= max_len:
        return [(0, n)]
    step = max_len - overlap
    out, s = [], 0
    while True:
        e = min(s + max_len, n)
        out.append((e - max_len, e))
        if e == n:
            return out
        s += step


@torch.no_grad()
def predict_texts(model, vocab: CharVocab, texts: list[str], device, max_len: int,
                  batch_size: int = 128, threshold: float = 0.5, overlap: int = 64) -> list[str]:
    """Return restored texts; only candidate characters may change."""
    pieces = []  # (text_idx, start, end)
    for ti, t in enumerate(texts):
        for s, e in windows(len(t), max_len, min(overlap, max_len // 2)):
            pieces.append((ti, s, e))
    probs = [torch.zeros(len(t)) for t in texts]
    best_margin = [torch.full((len(t),), -1.0) for t in texts]
    order = sorted(range(len(pieces)), key=lambda i: pieces[i][2] - pieces[i][1])
    use_amp = getattr(device, "type", str(device)) == "cuda"
    for b in range(0, len(order), batch_size):
        chunk = [pieces[i] for i in order[b:b + batch_size]]
        T = max(e - s for _, s, e in chunk)
        ids = torch.zeros(len(chunk), T, dtype=torch.long)
        mask = torch.zeros(len(chunk), T, dtype=torch.bool)
        for r, (ti, s, e) in enumerate(chunk):
            ids[r, : e - s] = torch.tensor(vocab.encode(texts[ti][s:e]), dtype=torch.long)
            mask[r, : e - s] = True
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            logits = model(ids.to(device), mask.to(device))
        p = torch.sigmoid(logits.float()).cpu()
        for r, (ti, s, e) in enumerate(chunk):
            L = e - s
            pos = torch.arange(L)
            margin = torch.minimum(pos, L - 1 - pos).float()  # distance from window edge
            better = margin > best_margin[ti][s:e]
            probs[ti][s:e] = torch.where(better, p[r, :L], probs[ti][s:e])
            best_margin[ti][s:e] = torch.where(better, margin, best_margin[ti][s:e])
    out = []
    for t, pr in zip(texts, probs):
        out.append("".join(TO_AZ[c] if c in CANDIDATES and pr[i] > threshold else c for i, c in enumerate(t)))
    return out
