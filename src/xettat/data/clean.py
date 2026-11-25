"""Sentence splitting and quality filtering for Azerbaijani text (no external deps)."""

from __future__ import annotations

import re

from xettat.text import AZ_LETTERS, nfc

_UPPER_START = "A-ZƏÖÜÇŞĞİ0-9\"«“(\\-–—"
# split after . ! ? … when followed by whitespace and an uppercase/digit/quote start
_SENT_BOUNDARY = re.compile(rf"(?<=[.!?…])\s+(?=[{_UPPER_START}])")
_LATIN_LETTER = re.compile(r"[A-Za-zƏəÖöÜüÇçŞşĞğİıIi]")


def split_sentences(text: str) -> list[str]:
    out: list[str] = []
    for line in nfc(text).splitlines():
        line = line.strip()
        if not line:
            continue
        out.extend(s.strip() for s in _SENT_BOUNDARY.split(line) if s.strip())
    return out


def is_good_sentence(s: str, min_chars: int = 15, max_chars: int = 400, min_az_ratio: float = 0.02) -> bool:
    if not (min_chars <= len(s) <= max_chars):
        return False
    letters = [ch for ch in s if ch.isalpha()]
    if not letters:
        return False
    latin = sum(1 for ch in letters if _LATIN_LETTER.match(ch))
    if latin / len(letters) < 0.9:
        return False
    az = sum(1 for ch in letters if ch in AZ_LETTERS)
    return az / len(letters) >= min_az_ratio
