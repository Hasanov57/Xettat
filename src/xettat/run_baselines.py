"""Fit baselines on train and score them on a split.

    python -m xettat.run_baselines --data data/processed --split val --out runs/baselines
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from xettat.baselines import Bigram, Identity, WordFreq
from xettat.metrics import score
from xettat.text import strip


def read_lines(path: Path) -> list[str]:
    return [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--out", default="runs/baselines")
    args = ap.parse_args(argv)

    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train = read_lines(data / "train.txt")
    targets = read_lines(data / f"{args.split}.txt")
    inputs = [strip(t) for t in targets]

    results = {}
    wf_for_amb = WordFreq().fit(train)
    amb = wf_for_amb.ambiguous()
    for model in (Identity(), wf_for_amb, Bigram()):
        if not isinstance(model, WordFreq) or isinstance(model, Bigram):
            model.fit(train)
        t0 = time.perf_counter()
        preds = [model.predict(x) for x in inputs]
        dt = time.perf_counter() - t0
        r = score(inputs, preds, targets, ambiguous=amb)
        r["latency_ms_per_sentence_cpu"] = 1000 * dt / max(1, len(inputs))
        r["model"] = model.name
        (out / f"{model.name}_{args.split}.json").write_text(json.dumps(r, ensure_ascii=False, indent=2))
        results[model.name] = r
        print(f"{model.name:9s} word_acc={r['word_acc']:.4f} sent_acc={r['sentence_acc']:.4f}")
    return results


if __name__ == "__main__":
    main()
