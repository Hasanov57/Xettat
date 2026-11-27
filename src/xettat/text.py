"""Azerbaijani text core: diacritic stripping, labels, Azerbaijani-aware casing.

Task framing: every Azerbaijani-specific letter has exactly one ASCII stand-in, so
stripping preserves length. A model only has to decide, for each *candidate* ASCII
letter, whether to keep it (0) or convert it (1). Everything else never changes.
"""

from __future__ import annotations

import random
import re
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


# ---------------------------------------------------------------------------------------------
# v2 scheme: per-character actions KEEP / CONVERT / DELETE (spelling variants: sh/ch/gh, a for ə)
# ---------------------------------------------------------------------------------------------

KEEP, CONVERT, DELETE = 0, 1, 2
TO_AZ_EXT: dict[str, str] = {**TO_AZ, "a": "ə", "A": "Ə"}
DIGRAPH_HEADS = frozenset("sScCgG")
_DIGRAPH_OF = {"ş": "s", "ç": "c", "ğ": "g"}

# Python's \w is Unicode-aware; the JS twin uses [\p{L}\p{N}_] with the "u" flag.
_PROTECTED = re.compile(
    r"(?:https?://|www\.)\S+"            # URLs
    r"|[\w.+-]+@\w+(?:\.\w+)+"          # e-mails
    r"|(?<!\w)[@#]\w+"                   # @mentions, #hashtags
)


_WORD = re.compile(r"[^\W\d_]+")  # letters only (mirrors JS \p{L}+)
_GUARD: frozenset[str] | None = None


def english_guard() -> frozenset[str]:
    """English words that are never Azerbaijani words (built by scripts/build_english_guard.py)."""
    global _GUARD
    if _GUARD is None:
        from pathlib import Path

        p = Path(__file__).with_name("english_guard.txt")
        _GUARD = frozenset(p.read_text(encoding="utf-8").split()) if p.exists() else frozenset()
    return _GUARD


def protected_mask(text: str, english: frozenset[str] | set[str] | None = None) -> list[bool]:
    """True for characters that are never changed: URLs, e-mails, @mentions, #hashtags and whole words
    found in the English guard (case-insensitive)."""
    mask = [False] * len(text)
    for m in _PROTECTED.finditer(text):
        for i in range(m.start(), m.end()):
            mask[i] = True
    guard = english_guard() if english is None else english
    if guard:
        for m in _WORD.finditer(text):
            if m.group().lower() in guard:
                for i in range(m.start(), m.end()):
                    mask[i] = True
    return mask


def allowed_actions(src: str, i: int) -> set[int]:
    c = src[i]
    if c in TO_AZ_EXT:
        return {KEEP, CONVERT}
    if c in "hH":
        return {KEEP, DELETE} if i > 0 and src[i - 1] in DIGRAPH_HEADS else {KEEP}
    return set()


def apply_actions(src: str, labels: Sequence[int], protected: Sequence[bool] | None = None) -> str:
    out = []
    for i, (c, lab) in enumerate(zip(src, labels)):
        if protected is not None and protected[i]:
            out.append(c)
        elif lab == DELETE and c in "hH" and i > 0 and src[i - 1] in DIGRAPH_HEADS:
            continue
        elif lab == CONVERT and c in TO_AZ_EXT:
            out.append(TO_AZ_EXT[c])
        else:
            out.append(c)
    return "".join(out)


def _insert_english(target: str, rng: random.Random) -> str:
    from xettat.english import WORDS

    words = target.split(" ")
    for _ in range(rng.choice((1, 1, 2))):
        w = rng.choice(WORDS)
        if rng.random() < 0.2:
            w = w.capitalize()
        words.insert(rng.randrange(len(words) + 1), w)
    return " ".join(words)


def make_variant_pair(
    target: str,
    rng: random.Random,
    style: str | None = None,
    p_partial: float = 0.3,
    p_keep: float = 0.3,
    p_upper: float = 0.05,
    p_english: float = 0.15,
    p_digraph: float = 0.8,
    p_a: float = 0.8,
    styles: tuple[str, ...] = ("std", "digraph", "a", "mixed"),
    style_weights: tuple[float, ...] = (0.4, 0.25, 0.15, 0.2),
) -> tuple[str, list[int], str]:
    """Return (noisy input, per-input-char labels, target used). apply_actions(input, labels) == target."""
    target = nfc(target)
    if rng.random() < p_english:
        target = _insert_english(target, rng)
    if rng.random() < p_upper:
        target = az_upper(target)
    style = style or rng.choices(styles, style_weights)[0]
    partial = rng.random() < p_partial
    inp: list[str] = []
    labels: list[int] = []
    for k, t in enumerate(target):
        if t in AZ_LETTERS:
            if partial and rng.random() < p_keep:
                inp.append(t)
                labels.append(IGNORE)
                continue
            low = az_lower(t)
            base = TO_ASCII[t]
            if low == "ə" and ((style == "a" and rng.random() < p_a) or (style == "mixed" and rng.random() < 0.3)):
                base = "A" if t.isupper() else "a"
            if low in _DIGRAPH_OF and ((style == "digraph" and rng.random() < p_digraph)
                                       or (style == "mixed" and rng.random() < 0.4)):
                nxt = target[k + 1] if k + 1 < len(target) else ""
                h = "H" if t.isupper() and (nxt.isupper() or not nxt.isalpha()) else "h"
                inp += [base, h]
                labels += [CONVERT, DELETE]
                continue
            inp.append(base)
            labels.append(CONVERT)
        else:
            inp.append(t)
            if t in TO_AZ_EXT:
                labels.append(KEEP)
            elif t in "hH" and inp[-2:-1] and inp[-2] in DIGRAPH_HEADS:
                labels.append(KEEP)
            else:
                labels.append(IGNORE)
    return "".join(inp), labels, target
