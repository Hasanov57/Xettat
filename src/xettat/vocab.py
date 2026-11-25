"""Fixed, data-independent character vocabulary (so checkpoints never depend on corpus)."""

from __future__ import annotations

import json

from xettat.text import AZ_LETTERS

_EXTRA = "«»–—…’‘“”№°"


class CharVocab:
    PAD = 0
    UNK = 1

    def __init__(self, chars: list[str] | None = None) -> None:
        if chars is None:
            chars = [chr(c) for c in range(32, 127)] + sorted(AZ_LETTERS) + list(_EXTRA) + ["\t", "\n"]
        self.chars = chars
        self.index = {ch: i + 2 for i, ch in enumerate(chars)}

    @property
    def size(self) -> int:
        return len(self.chars) + 2

    def encode(self, s: str) -> list[int]:
        return [self.index.get(ch, self.UNK) for ch in s]

    def to_json(self) -> str:
        return json.dumps(self.chars, ensure_ascii=False)

    @classmethod
    def from_json(cls, s: str) -> "CharVocab":
        return cls(json.loads(s))
