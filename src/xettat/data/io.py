"""Torch-free loading of evaluation splits."""

from __future__ import annotations

from pathlib import Path

from xettat.text import strip


def load_split(data: Path, split: str, real_path: Path, tsv: str | None = None) -> tuple[list[str], list[str]]:
    if tsv or split == "real":
        path = Path(tsv) if tsv else real_path
        inputs, targets = [], []
        for line in path.read_text(encoding="utf-8").splitlines()[1:]:
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            k = len(parts) // 2  # legacy files may contain tabs inside sentences: split in the middle
            inputs.append("\t".join(parts[:k]))
            targets.append("\t".join(parts[k:]))
        return inputs, targets
    targets = [l for l in (data / f"{split}.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    return [strip(t) for t in targets], targets
