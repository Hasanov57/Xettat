"""Fixed evaluation sets for each writing style (v2 scheme).

    python -m xettat.data.variants --data data/processed --split val --out data/variants --n 5000

Writes {split}_{style}.tsv with header "input\\ttarget". Styles:
  std      e/i/o/u/c/s/g stand-ins only (same inputs as v1 evaluation)
  digraph  ş/ç/ğ written as sh/ch/gh
  a        ə written as a
  mixed    random mix per character + partial correct letters + occasional uppercase
  english  std + 1-2 frequent English words inserted (they must stay untouched)
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from xettat.text import make_variant_pair

STYLES = ("std", "digraph", "a", "mixed", "english")
_KW = {
    "std": dict(style="std", p_partial=0, p_upper=0, p_english=0),
    "digraph": dict(style="digraph", p_partial=0, p_upper=0, p_english=0, p_digraph=1.0),
    "a": dict(style="a", p_partial=0, p_upper=0, p_english=0, p_a=1.0),
    "mixed": dict(style="mixed", p_english=0),
    "english": dict(style="std", p_partial=0, p_upper=0, p_english=1.0),
}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--out", default="data/variants")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)

    lines = [l for l in (Path(args.data) / f"{args.split}.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    sample = random.Random(args.seed).sample(lines, min(args.n, len(lines)))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for style in STYLES:
        rng = random.Random(f"{args.seed}:{style}")
        rows = ["input\ttarget"]
        for t in sample:
            inp, _, tgt = make_variant_pair(t.replace("\t", " "), rng, **_KW[style])  # TSV-safe
            rows.append(f"{inp}\t{tgt}")
        (out / f"{args.split}_{style}.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
        print(style, len(rows) - 1)


if __name__ == "__main__":
    main()
