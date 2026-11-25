"""Training-time input noise: real users sometimes type some Azerbaijani letters correctly."""

from __future__ import annotations

import random

from xettat.text import TO_ASCII, az_upper, nfc


def make_input(
    target: str,
    rng: random.Random,
    p_partial: float = 0.3,
    p_keep: float = 0.3,
    p_upper: float = 0.05,
) -> tuple[str, str]:
    """Return (noisy ASCII-ish input, target actually used). Lengths are equal."""
    target = nfc(target)
    if rng.random() < p_upper:
        target = az_upper(target)
    partial = rng.random() < p_partial
    chars = []
    for ch in target:
        if ch in TO_ASCII and not (partial and rng.random() < p_keep):
            chars.append(TO_ASCII[ch])
        else:
            chars.append(ch)
    return "".join(chars), target
