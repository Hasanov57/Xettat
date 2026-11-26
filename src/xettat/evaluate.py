"""Evaluate a checkpoint on val / test / real-user data.

    python -m xettat.evaluate --ckpt CKPT --data data/processed --split test --out runs/<run_id>
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

from xettat.baselines import WordFreq
from xettat.metrics import score
from xettat.restorer import Restorer
from xettat.text import strip


def load_split(data: Path, split: str, real_path: Path) -> tuple[list[str], list[str]]:
    if split == "real":
        rows = [l.split("\t") for l in real_path.read_text(encoding="utf-8").splitlines()[1:] if "\t" in l]
        return [r[0] for r in rows], [r[1] for r in rows]
    targets = [l for l in (data / f"{split}.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    return [strip(t) for t in targets], targets


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--split", default="val", choices=["val", "test", "real"])
    ap.add_argument("--real", default="data/real_test/real_test.tsv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--latency-n", type=int, default=200)
    ap.add_argument("--tag", default="", help="suffix for output files, e.g. the data domain (hplt, subs)")
    args = ap.parse_args(argv)

    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    inputs, targets = load_split(data, args.split, Path(args.real))
    r = Restorer.from_checkpoint(args.ckpt, device=args.device)
    preds = r.restore_batch(inputs)

    amb = None
    if (data / "train.txt").exists():
        amb = WordFreq().fit(l for l in (data / "train.txt").read_text(encoding="utf-8").splitlines()).ambiguous()
    m = score(inputs, preds, targets, ambiguous=amb)

    # CPU latency: one sentence at a time, the way the extension/API calls it
    cpu = Restorer.from_checkpoint(args.ckpt, device="cpu")
    torch.set_num_threads(1)
    sample = inputs[: args.latency_n] or ["salam"]
    t0 = time.perf_counter()
    for x in sample:
        cpu.restore(x)
    m["latency_ms_per_sentence_cpu"] = 1000 * (time.perf_counter() - t0) / len(sample)
    m["params"] = sum(p.numel() for p in cpu.model.parameters())
    m["ckpt_mb"] = os.path.getsize(args.ckpt) / 1e6
    m["ckpt"] = args.ckpt
    m["split"] = args.split
    suffix = f"{args.split}_{args.tag}" if args.tag else args.split
    m["tag"] = args.tag
    (out / f"final_{suffix}.json").write_text(json.dumps(m, ensure_ascii=False, indent=2))
    errors = [{"input": i, "pred": p, "target": t} for i, p, t in zip(inputs, preds, targets) if p != t][:500]
    (out / f"errors_{suffix}.jsonl").write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in errors))
    print(json.dumps({k: v for k, v in m.items() if not isinstance(v, dict)}, ensure_ascii=False))
    return m


if __name__ == "__main__":
    main()
