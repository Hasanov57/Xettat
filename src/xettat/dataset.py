"""Torch dataset: targets on disk, inputs generated on the fly (stripping + augmentation)."""

from __future__ import annotations

import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from xettat.data.augment import make_input
from xettat.text import CANDIDATES, IGNORE, TO_ASCII, make_labels, make_variant_pair, nfc
from xettat.vocab import CharVocab


def labels_for(inp: str, target: str) -> list[int]:
    """Labels for a (possibly partially correct) input against its target."""
    out = []
    for c, t in zip(inp, target):
        if c in CANDIDATES:
            out.append(1 if t in TO_ASCII else 0)
        else:
            out.append(IGNORE)
    return out


class SentenceDataset(Dataset):
    def __init__(self, path: str | Path, vocab: CharVocab, max_len: int, augment: bool, seed: int = 0,
                 limit: int | None = None, scheme: str = "v1"):
        lines = [nfc(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
        self.sentences = lines[:limit] if limit else lines
        self.vocab, self.max_len, self.augment = vocab, max_len, augment
        self.seed = seed
        self.scheme = scheme
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, i: int) -> tuple[list[int], list[int]]:
        target = self.sentences[i]
        if len(target) > self.max_len:
            start = self._rng.randrange(len(target) - self.max_len + 1) if self.augment else 0
            target = target[start:start + self.max_len]
        if self.scheme == "v2":
            if self.augment:
                inp, labels, _ = make_variant_pair(target, self._rng)
            else:
                inp, labels, _ = make_variant_pair(target, self._rng, style="std", p_partial=0, p_upper=0, p_english=0)
            inp, labels = inp[: self.max_len], labels[: self.max_len]  # digraphs can make input longer
        elif self.augment:
            inp, target = make_input(target, self._rng)
            labels = labels_for(inp, target)
        else:
            inp, labels = make_labels(target)
        return self.vocab.encode(inp), labels


def worker_init(worker_id: int) -> None:
    info = torch.utils.data.get_worker_info()
    ds = info.dataset
    ds._rng = random.Random(ds.seed * 1000 + worker_id + int(torch.initial_seed() % 10**6))


def collate(batch: list[tuple[list[int], list[int]]]):
    T = max(len(ids) for ids, _ in batch)
    B = len(batch)
    ids_t = torch.zeros(B, T, dtype=torch.long)
    lab_t = torch.full((B, T), IGNORE, dtype=torch.long)
    mask = torch.zeros(B, T, dtype=torch.bool)
    for b, (ids, labels) in enumerate(batch):
        n = len(ids)
        ids_t[b, :n] = torch.tensor(ids, dtype=torch.long)
        lab_t[b, :n] = torch.tensor(labels, dtype=torch.long)
        mask[b, :n] = True
    return ids_t, mask, lab_t
