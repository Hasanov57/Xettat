"""Mix several prepared corpora into one training set, keeping every source's held-out data clean.

    python -m xettat.data.mix --sources data/processed data/hplt data/subs --out data/mix

- train = union of all train.txt, minus any sentence that appears in ANY source's val/test (no leakage,
  e.g. web pages that copy Wikipedia), shuffled with a fixed seed.
- val   = up to --val-per-source sentences from each source's val.txt (used for early stopping).
- test sets are NOT copied: evaluate each domain on its own data dir (data/processed, data/hplt, ...).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def _lines(p: Path) -> list[str]:
    return [l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-per-source", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    sources = [Path(s) for s in args.sources if (Path(s) / "train.txt").exists()]
    held_out: set[str] = set()
    for s in sources:
        held_out.update(_lines(s / "val.txt"))
        held_out.update(_lines(s / "test.txt"))

    rng = random.Random(args.seed)
    train: dict[str, None] = {}
    val: list[str] = []
    stats = {"sources": {}}
    for s in sources:
        tr = [l for l in _lines(s / "train.txt") if l not in held_out]
        before = len(train)
        train.update(dict.fromkeys(tr))
        v = _lines(s / "val.txt")
        v = rng.sample(v, min(len(v), args.val_per_source))
        val += v
        stats["sources"][str(s)] = {"train_added": len(train) - before, "val": len(v)}
    train_list = list(train)
    rng.shuffle(train_list)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "train.txt").write_text("\n".join(train_list) + "\n", encoding="utf-8")
    (out / "val.txt").write_text("\n".join(val) + "\n", encoding="utf-8")
    stats.update(train=len(train_list), val=len(val), seed=args.seed)
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False))
    return stats


if __name__ == "__main__":
    main()
