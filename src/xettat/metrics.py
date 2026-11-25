"""Evaluation metrics for diacritic restoration (inputs, preds, targets are aligned strings)."""

from __future__ import annotations

import re

from xettat.text import AZ_LETTERS, CANDIDATES, az_lower, nfc, strip

WORD = re.compile(r"\w+")
PAIRS = ["e/ə", "i/ı", "o/ö", "u/ü", "c/ç", "s/ş", "g/ğ"]
_PAIR_OF = {}
for _p in PAIRS:
    _a, _z = _p.split("/")
    _PAIR_OF[_a] = _PAIR_OF[_z] = _p


def _safe(n: float, d: float) -> float:
    return n / d if d else 0.0


def score(inputs: list[str], preds: list[str], targets: list[str], ambiguous: set[str] | None = None) -> dict:
    assert len(inputs) == len(preds) == len(targets)
    c_ok = c_tot = 0
    w_ok = w_tot = 0
    a_ok = a_tot = 0
    s_ok = 0
    werr = wall = 0
    corr_bad = corr_tot = 0
    pair = {p: {"tp": 0, "fp": 0, "fn": 0} for p in PAIRS}

    for inp, pred, tgt in zip(inputs, preds, targets):
        inp, pred, tgt = nfc(inp), nfc(pred), nfc(tgt)
        aligned = len(pred) == len(tgt)
        s_ok += pred == tgt
        src = strip(tgt)
        if aligned:
            for i, (p, t) in enumerate(zip(pred, tgt)):
                if src[i] in CANDIDATES:
                    c_tot += 1
                    c_ok += p == t
                    key = _PAIR_OF.get(az_lower(src[i]))
                    if key:
                        t_az = t in AZ_LETTERS
                        p_az = p in AZ_LETTERS
                        pair[key]["tp"] += t_az and p_az
                        pair[key]["fp"] += p_az and not t_az
                        pair[key]["fn"] += t_az and not p_az
                if i < len(inp) and inp[i] in AZ_LETTERS and inp[i] == t:
                    corr_tot += 1
                    corr_bad += p != t
        else:
            c_tot += sum(ch in CANDIDATES for ch in src)
        for m in WORD.finditer(tgt):
            tw = m.group()
            pw = pred[m.start():m.end()] if aligned else None
            wall += 1
            werr += pw != tw
            if any(ch in CANDIDATES for ch in strip(tw)):
                w_tot += 1
                w_ok += pw == tw
                if ambiguous is not None and az_lower(strip(tw)) in ambiguous:
                    a_tot += 1
                    a_ok += pw == tw

    per_pair = {}
    for k, v in pair.items():
        pr = _safe(v["tp"], v["tp"] + v["fp"])
        rc = _safe(v["tp"], v["tp"] + v["fn"])
        if v["tp"] + v["fp"] + v["fn"] == 0:
            pr = rc = 1.0
        per_pair[k] = {"p": pr, "r": rc, "f1": _safe(2 * pr * rc, pr + rc), "support": v["tp"] + v["fn"]}
    return {
        "char_acc": _safe(c_ok, c_tot),
        "word_acc": _safe(w_ok, w_tot),
        "ambig_word_acc": _safe(a_ok, a_tot) if ambiguous is not None else None,
        "sentence_acc": _safe(s_ok, len(targets)),
        "wer": _safe(werr, wall),
        "corruption_rate": _safe(corr_bad, corr_tot),
        "per_pair": per_pair,
        "n_sentences": len(targets),
    }
