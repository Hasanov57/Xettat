"""Collect runs/*/final_<split>.json and baseline JSONs into one markdown results table.

    python -m xettat.report_table --runs runs --split test --out report/tables/results.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

COLS = [("word_acc", "Word acc"), ("ambig_word_acc", "Ambig. word acc"), ("char_acc", "Char acc"),
        ("sentence_acc", "Sent. acc"), ("wer", "WER"), ("corruption_rate", "Corruption"),
        ("latency_ms_per_sentence_cpu", "CPU ms/sent"), ("params", "Params"), ("ckpt_mb", "Size MB")]


def _fmt(v) -> str:
    if v is None:
        return "–"
    if isinstance(v, int):
        return f"{v / 1e6:.1f}M" if v >= 1e5 else str(v)
    return f"{v:.4f}" if v < 10 else f"{v:.1f}"


def collect(runs: Path, split: str) -> list[tuple[str, dict]]:
    rows = []
    for f in sorted(runs.glob(f"*/final_{split}.json")):
        rows.append((f.parent.name, json.loads(f.read_text())))
    for f in sorted((runs / "baselines").glob(f"*_{split}.json")):
        d = json.loads(f.read_text())
        rows.append((d.get("model", f.stem), d))
    return sorted(rows, key=lambda r: r[1].get("word_acc", 0))


def main(argv: list[str] | None = None) -> str:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="report/tables/results.md")
    args = ap.parse_args(argv)
    rows = collect(Path(args.runs), args.split)
    lines = ["| Model | " + " | ".join(h for _, h in COLS) + " |", "|---" * (len(COLS) + 1) + "|"]
    for name, d in rows:
        lines.append(f"| {name} | " + " | ".join(_fmt(d.get(k)) for k, _ in COLS) + " |")
    md = f"Results on **{args.split}** split\n\n" + "\n".join(lines) + "\n"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    print(md)
    return md


if __name__ == "__main__":
    main()
