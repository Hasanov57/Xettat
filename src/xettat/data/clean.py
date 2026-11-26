"""Sentence splitting and quality filtering for Azerbaijani text (no external deps)."""

from __future__ import annotations

import re

from xettat.text import AZ_LETTERS, nfc

_UPPER_START = "A-ZƏÖÜÇŞĞİ0-9\"«“(\\-–—"
# split after . ! ? … when followed by whitespace and an uppercase/digit/quote start
_SENT_BOUNDARY = re.compile(rf"(?<=[.!?…])\s+(?=[{_UPPER_START}])")
_LATIN_LETTER = re.compile(r"[A-Za-zƏəÖöÜüÇçŞşĞğİıIi]")
# pre-2001 / non-standard spellings of ə (Latin ä, Cyrillic schwa) — such text teaches wrong targets
_NONSTANDARD = frozenset("äÄәӘ")
_DETOK = [
    (re.compile(r"\s+([.,!?;:%)\]»…])"), r"\1"),
    (re.compile(r"([(\[«])\s+"), r"\1"),
]


def split_sentences(text: str) -> list[str]:
    out: list[str] = []
    for line in nfc(text).splitlines():
        line = line.strip()
        if not line:
            continue
        out.extend(s.strip() for s in _SENT_BOUNDARY.split(line) if s.strip())
    return out


def az_ratio(s: str) -> float:
    """Share of letters that are Azerbaijani-specific (normal running text: ~0.12-0.18)."""
    letters = [ch for ch in s if ch.isalpha()]
    return sum(ch in AZ_LETTERS for ch in letters) / len(letters) if letters else 0.0


def detokenize(s: str) -> str:
    """Undo OPUS-style tokenisation ("Salam , necəsən ?" -> "Salam, necəsən?")."""
    for pat, rep in _DETOK:
        s = pat.sub(rep, s)
    return s.strip()


def is_good_sentence(s: str, min_chars: int = 15, max_chars: int = 400, min_az_ratio: float = 0.02) -> bool:
    if not (min_chars <= len(s) <= max_chars):
        return False
    if any(ch in _NONSTANDARD for ch in s):
        return False
    letters = [ch for ch in s if ch.isalpha()]
    if not letters:
        return False
    latin = sum(1 for ch in letters if _LATIN_LETTER.match(ch))
    if latin / len(letters) < 0.9:
        return False
    az = sum(1 for ch in letters if ch in AZ_LETTERS)
    return az / len(letters) >= min_az_ratio
