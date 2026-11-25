"""Non-neural baselines: identity, word-frequency lexicon, lexicon + bigram context."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Iterable

from xettat.text import AZ_LETTERS, az_lower, az_upper, nfc, strip

WORD = re.compile(r"\w+")


def _recase(template: str, form: str) -> str:
    """Copy the per-character casing of `template` onto lowercase `form`."""
    return "".join(az_upper(f) if t.isupper() else f for t, f in zip(template, form))


def _compatible(typed_lower: str, form: str) -> bool:
    """Letters the user typed as Azerbaijani letters must be kept by the candidate form."""
    return all(t == f for t, f in zip(typed_lower, form) if t in AZ_LETTERS)


class Identity:
    name = "identity"

    def fit(self, sentences: Iterable[str]) -> "Identity":
        return self

    def predict(self, text: str) -> str:
        return text


class WordFreq:
    name = "wordfreq"

    def __init__(self) -> None:
        self.forms: dict[str, Counter] = defaultdict(Counter)

    def fit(self, sentences: Iterable[str]) -> "WordFreq":
        for s in sentences:
            for w in WORD.findall(az_lower(nfc(s))):
                self.forms[strip(w)][w] += 1
        return self

    def ambiguous(self) -> set[str]:
        return {k for k, c in self.forms.items() if len(c) > 1}

    def candidates(self, word: str) -> list[str]:
        low = az_lower(word)
        c = self.forms.get(strip(low))
        if not c:
            return []
        return [f for f, _ in c.most_common() if len(f) == len(word) and _compatible(low, f)]

    def _choose(self, word: str, prev: str | None) -> str:
        cands = self.candidates(word)
        return cands[0] if cands else az_lower(word)

    def predict(self, text: str) -> str:
        text = nfc(text)
        out, last, prev = [], 0, None
        for m in WORD.finditer(text):
            w = m.group()
            form = self._choose(w, prev)
            out.append(text[last:m.start()])
            out.append(_recase(w, form) if self.candidates(w) else w)
            last, prev = m.end(), form
        out.append(text[last:])
        return "".join(out)


class Bigram(WordFreq):
    name = "bigram"

    def __init__(self) -> None:
        super().__init__()
        self.bigrams: Counter = Counter()

    def fit(self, sentences: Iterable[str]) -> "Bigram":
        sentences = list(sentences)
        super().fit(sentences)
        amb = self.ambiguous()
        for s in sentences:
            words = WORD.findall(az_lower(nfc(s)))
            for a, b in zip(words, words[1:]):
                if strip(b) in amb:
                    self.bigrams[(a, b)] += 1
        return self

    def _choose(self, word: str, prev: str | None) -> str:
        cands = self.candidates(word)
        if not cands:
            return az_lower(word)
        if len(cands) > 1 and prev is not None:
            best = max(cands, key=lambda f: self.bigrams.get((prev, f), 0))
            if self.bigrams.get((prev, best), 0) > 0:
                return best
        return cands[0]
