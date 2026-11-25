"""CPU deployment measurements for the report (Track 3): latency, size, memory.

    python scripts/benchmark.py --exported exported/bilstm_v1 --data data/processed --n 500 [--threads 1]
"""

from __future__ import annotations

import argparse
import json
import resource
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from xettat.text import strip  # noqa: E402
from xettat.vocab import CharVocab  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exported", required=True)
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()

    exp = Path(args.exported)
    meta = json.loads((exp / "meta.json").read_text())
    vocab = CharVocab(json.loads((exp / "vocab.json").read_text()))
    lines = [l for l in (Path(args.data) / "val.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    texts = [strip(l)[: meta["max_len"]] for l in lines[: args.n]]

    results = {"model_type": meta["model_type"], "params": meta["params"], "threads": args.threads, "n": len(texts)}
    for f in sorted(exp.glob("*.onnx")):
        so = ort.SessionOptions()
        so.intra_op_num_threads = args.threads
        sess = ort.InferenceSession(str(f), so, providers=["CPUExecutionProvider"])
        feed = lambda t: {"ids": np.array([vocab.encode(t)], dtype=np.int64),
                          "mask": np.ones((1, len(t)), dtype=bool)}
        for t in texts[:20]:
            sess.run(["logits"], feed(t))  # warm-up
        times = []
        for t in texts:
            t0 = time.perf_counter()
            sess.run(["logits"], feed(t))
            times.append(1000 * (time.perf_counter() - t0))
        q = statistics.quantiles(times, n=100)
        results[f.name] = {
            "size_mb": round(f.stat().st_size / 1e6, 2),
            "ms_mean": round(statistics.mean(times), 3), "ms_p50": round(q[49], 3), "ms_p95": round(q[94], 3),
            "chars_per_sec": round(sum(map(len, texts)) / (sum(times) / 1000)),
        }
    results["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    out = exp / "benchmark.json"
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
