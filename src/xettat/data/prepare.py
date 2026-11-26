"""Build frozen train/val/test sentence files from a corpus.

    python -m xettat.data.prepare --source wiki --out data/processed --seed 42
    python -m xettat.data.prepare --source hplt:4 --out data/hplt --doc-min-az-ratio 0.08 --max-train 3000000
    python -m xettat.data.prepare --source opensubs --out data/subs --doc-min-az-ratio 0.08
    python -m xettat.data.prepare --source txt:corpus.txt --out data/processed
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Iterator

from xettat.data.clean import az_ratio, detokenize, is_good_sentence, split_sentences

LICENSES = {
    "wiki": "CC BY-SA 4.0 (Wikimedia, wikimedia/wikipedia 20231101.az)",
    "hplt": "CC0-1.0 (HPLT/HPLT2.0_cleaned, azj_Latn)",
    "opensubs": "OPUS OpenSubtitles v2024 (opensubtitles.org; research use, cite Lison & Tiedemann 2016)",
}
HPLT_SHARD = "hf://datasets/HPLT/HPLT2.0_cleaned/azj_Latn/train-{:05d}-of-00054.parquet"
OPUS_SUBS = "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2024/mono/az.txt.gz"  # verified 200 on 2026-11-27
SUBS_CHUNK = 200  # subtitle lines per pseudo-document (split + doc-level filter unit)


def split_docs(doc_ids: list[str], seed: int, val_frac: float = 0.01, test_frac: float = 0.01) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    for d in doc_ids:
        h = int(hashlib.sha1(f"{seed}:{d}".encode()).hexdigest()[:12], 16) / 16**12
        if h < test_frac:
            out["test"].add(d)
        elif h < test_frac + val_frac:
            out["val"].add(d)
        else:
            out["train"].add(d)
    return out


def iter_docs(source: str, limit: int | None) -> Iterator[tuple[str, str]]:
    n = 0
    if source == "wiki":
        from datasets import load_dataset  # heavy import only when needed

        ds = load_dataset("wikimedia/wikipedia", "20231101.az", split="train")
        for row in ds:
            yield str(row["id"]), row["text"]
            n += 1
            if limit and n >= limit:
                return
    elif source.startswith("txt:"):
        text = Path(source[4:]).read_text(encoding="utf-8")
        for i, block in enumerate(b for b in text.split("\n\n") if b.strip()):
            yield f"txt{i}", block
            if limit and i + 1 >= limit:
                return
    elif source.startswith("hplt"):
        from datasets import load_dataset

        shards = int(source.split(":")[1]) if ":" in source else 2
        files = [HPLT_SHARD.format(i) for i in range(shards)]
        ds = load_dataset("parquet", data_files=files, split="train", columns=["text"])
        for i, row in enumerate(ds):
            yield f"hplt{i}", row["text"]
            n += 1
            if limit and n >= limit:
                return
    elif source == "opensubs":
        import gzip
        import urllib.request

        cache = Path(os.environ.get("XETTAT_CACHE", ".cache")) / "opensubs_az.txt.gz"
        cache.parent.mkdir(parents=True, exist_ok=True)
        if not cache.exists():
            urllib.request.urlretrieve(OPUS_SUBS, cache)
        with gzip.open(cache, "rt", encoding="utf-8", errors="replace") as f:
            buf: list[str] = []
            for line in f:
                line = detokenize(line.strip())
                if line:
                    buf.append(line)
                if len(buf) >= SUBS_CHUNK:
                    yield f"subs{n}", "\n".join(buf)
                    buf, n = [], n + 1
                    if limit and n >= limit:
                        return
            if buf:
                yield f"subs{n}", "\n".join(buf)
    else:
        raise ValueError(f"unknown source {source!r}")


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.01)
    ap.add_argument("--test-frac", type=float, default=0.01)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--doc-min-az-ratio", type=float, default=0.0,
                    help="drop whole documents typed (mostly) without Azerbaijani letters; normal text ~0.12-0.18")
    ap.add_argument("--max-train", type=int, default=None, help="random cap on train sentences")
    args = ap.parse_args(argv)

    docs, dropped = [], 0
    for doc_id, text in iter_docs(args.source, args.limit):
        if args.doc_min_az_ratio and az_ratio(text) < args.doc_min_az_ratio:
            dropped += 1
            continue
        docs.append((doc_id, text))
    split = split_docs([d for d, _ in docs], args.seed, args.val_frac, args.test_frac)
    which = {d: name for name, ids in split.items() for d in ids}

    sents: dict[str, dict[str, None]] = {"train": {}, "val": {}, "test": {}}  # ordered sets
    for doc_id, text in docs:
        bucket = sents[which[doc_id]]
        for s in split_sentences(text):
            if is_good_sentence(s):
                bucket[s] = None
    held_out = set(sents["val"]) | set(sents["test"])
    sents["train"] = {s: None for s in sents["train"] if s not in held_out}
    sents["val"] = {s: None for s in sents["val"] if s not in sents["test"]}
    if args.max_train and len(sents["train"]) > args.max_train:
        keep = random.Random(args.seed).sample(list(sents["train"]), args.max_train)
        sents["train"] = {s: None for s in keep}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, bucket in sents.items():
        (out / f"{name}.txt").write_text("\n".join(bucket) + "\n", encoding="utf-8")
    stats = {name: len(b) for name, b in sents.items()}
    stats.update(
        source=args.source, seed=args.seed, docs=len(docs), docs_dropped_low_az_ratio=dropped,
        docs_per_split={k: len(v) for k, v in split.items()},
        license=LICENSES.get(args.source.split(":")[0], "see source"),
    )
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False))
    return stats


if __name__ == "__main__":
    main()
