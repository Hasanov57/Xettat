"""Batched inference with sliding windows for inputs longer than the model's max_len.

Works for both schemes:
  v1 (binary, logits [B,T])       -> CONVERT where sigmoid > threshold, only on v1 candidates
  v2 (3-class, logits [B,T,3])    -> masked argmax over KEEP/CONVERT/DELETE allowed for that character
Protected spans (URLs, e-mails, @mentions, #hashtags) are never changed.
"""

from __future__ import annotations

import torch

from xettat.text import CANDIDATES, CONVERT, DELETE, KEEP, allowed_actions, apply_actions, protected_mask
from xettat.vocab import CharVocab

NEG = -1e9


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


def _actions_v1(text: str, logits: torch.Tensor, threshold: float) -> list[int]:
    p = torch.sigmoid(logits)
    return [CONVERT if c in CANDIDATES and p[i] > threshold else KEEP for i, c in enumerate(text)]


def _actions_v2(text: str, offset: int, logits: torch.Tensor, a_bias: float) -> list[int]:
    """logits [L,3] for text[offset:offset+L]; allowed actions use the FULL text (s|h across windows)."""
    L = logits.size(0)
    allowed = torch.zeros(L, 3, dtype=torch.bool)
    is_a = torch.zeros(L, dtype=torch.bool)
    for i in range(L):
        for a in allowed_actions(text, offset + i):
            allowed[i, a] = True
        is_a[i] = text[offset + i] in "aA"
    allowed[:, KEEP] = True  # keeping is always possible
    row = logits.clone()
    row[:, CONVERT] += a_bias * is_a  # negative bias = more conservative about a -> ə
    return row.masked_fill(~allowed, NEG).argmax(-1).tolist()


@torch.no_grad()
def predict_texts(model, vocab: CharVocab, texts: list[str], device, max_len: int,
                  batch_size: int = 128, threshold: float = 0.5, overlap: int = 64, a_bias: float = 0.0) -> list[str]:
    """Return restored texts; only allowed characters may change (see module docstring)."""
    pieces = []  # (text_idx, start, end)
    for ti, t in enumerate(texts):
        for s, e in windows(len(t), max_len, min(overlap, max_len // 2)):
            pieces.append((ti, s, e))
    actions = [[KEEP] * len(t) for t in texts]
    best_margin = [[-1] * len(t) for t in texts]
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
        logits = logits.float().cpu()
        for r, (ti, s, e) in enumerate(chunk):
            L = e - s
            if logits.dim() == 2:
                acts = _actions_v1(texts[ti][s:e], logits[r, :L], threshold)
            else:
                acts = _actions_v2(texts[ti], s, logits[r, :L], a_bias)
            for i in range(L):
                margin = min(i, L - 1 - i)  # distance from window edge
                if margin > best_margin[ti][s + i]:
                    best_margin[ti][s + i] = margin
                    actions[ti][s + i] = acts[i]
    return [apply_actions(t, a, protected_mask(t)) for t, a in zip(texts, actions)]
