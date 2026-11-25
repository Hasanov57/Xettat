"""Azerbaijani text core: diacritic stripping, labels, Azerbaijani-aware casing.

Task framing: every Azerbaijani-specific letter has exactly one ASCII stand-in, so
stripping preserves length. A model only has to decide, for each *candidate* ASCII
letter, whether to keep it (0) or convert it (1). Everything else never changes.
"""

from __future__ import annotations

import unicodedata
from typing import Sequence

TO_AZ: dict[str, str] = {
    "e": "ə", "i": "ı", "o": "ö", "u": "ü", "c": "ç", "s": "ş", "g": "ğ",
    "E": "Ə", "I": "İ", "O": "Ö", "U": "Ü", "C": "Ç", "S": "Ş", "G": "Ğ",
}
TO_ASCII: dict[str, str] = {v: k for k, v in TO_AZ.items()}
CANDIDATES: frozenset[str] = frozenset(TO_AZ)
AZ_LETTERS: frozenset[str] = frozenset(TO_ASCII)

IGNORE = -100

_STRIP_TABLE = str.maketrans(TO_ASCII)
_LOWER_SPECIAL = str.maketrans({"I": "ı", "İ": "i"})
_UPPER_SPECIAL = str.maketrans({"i": "İ", "ı": "I"})


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def az_lower(s: str) -> str:
    """Lowercase with Azerbaijani rules (I→ı, İ→i)."""
    return s.translate(_LOWER_SPECIAL).lower()


def az_upper(s: str) -> str:
    """Uppercase with Azerbaijani rules (i→İ, ı→I)."""
    return s.translate(_UPPER_SPECIAL).upper()


def strip(s: str) -> str:
    """Replace every Azerbaijani-specific letter by its ASCII stand-in."""
    return nfc(s).translate(_STRIP_TABLE)


def make_labels(target: str) -> tuple[str, list[int]]:
    """Return (stripped input, per-char labels) for a correctly written target."""
    target = nfc(target)
    src = target.translate(_STRIP_TABLE)
    labels = []
    for t, c in zip(target, src):
        if t in AZ_LETTERS:
            labels.append(1)
        elif c in CANDIDATES:
            labels.append(0)
        else:
            labels.append(IGNORE)
    return src, labels


def apply_labels(src: str, labels: Sequence[int]) -> str:
    """Convert candidate characters whose label is 1; leave everything else."""
    return "".join(
        TO_AZ[c] if lab == 1 and c in CANDIDATES else c for c, lab in zip(src, labels)
    )
