"""Build the English-word guard: frequent English words that are NOT Azerbaijani words typed without
diacritics, so restoring them could only corrupt them (e.g. curve -> cürvə).

    python scripts/build_english_guard.py --en data/english/google-20k.txt --corpus data/mix/train.txt \
        --out src/xettat/english_guard.txt --copy-to web/shared/english-guard.txt extension/shared/english-guard.txt

A word is kept when: >= 3 letters, ASCII a-z only, contains a candidate letter (e i o u c s g a), and its
form appears fewer than --min-az-count times among Azerbaijani corpus words after stripping diacritics.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from xettat.text import az_lower, strip  # noqa: E402

WORD = re.compile(r"\w+")
CAND = set("eioucsga")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--en", required=True)
    ap.add_argument("--corpus", required=True, nargs="+")
    ap.add_argument("--min-az-count", type=int, default=3)
    ap.add_argument("--max-lines", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--copy-to", nargs="*", default=[], help="extra copies, e.g. web/shared/english-guard.txt")
    args = ap.parse_args(argv)

    en = [w.strip().lower() for w in Path(args.en).read_text(encoding="utf-8").splitlines()]
    en = [w for w in en if len(w) >= 3 and re.fullmatch(r"[a-z]+", w) and CAND & set(w)]
    wanted = set(en)
    az = Counter()
    for path in args.corpus:
        with open(path, encoding="utf-8") as f:
            for n, line in enumerate(f):
                if args.max_lines and n >= args.max_lines:
                    break
                for w in WORD.findall(line):
                    s = strip(az_lower(w))
                    if s in wanted:
                        az[s] += 1
    guard = sorted(w for w in wanted if az[w] < args.min_az_count)
    for target in [args.out, *args.copy_to]:
        Path(target).write_text("\n".join(guard) + "\n", encoding="utf-8")
    print(f"english candidates: {len(wanted)}, also Azerbaijani: {len(wanted) - len(guard)}, guard: {len(guard)}")


if __name__ == "__main__":
    main()
