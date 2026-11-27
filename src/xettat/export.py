"""Export a char-level checkpoint to ONNX for the browser (onnxruntime-web) and the API.

    python -m xettat.export --ckpt CKPT --out exported/<run_id> [--data data/processed]

Writes model.onnx, model.int8.onnx (if quantisation keeps decisions), vocab.json, meta.json.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from xettat.restorer import Restorer
from xettat.text import CANDIDATES, allowed_actions, strip

SAMPLE = [
    "uzumun suyu uzume sicradi",
    "sabah seher saat 9-da seherin merkezinde gorusek",
    "Ilham bu gun Istanbula ucur, gelende cicek getirecek.",
    "Azerbaycan Respublikasinin paytaxti Baki seheridir.",
]


def decisions(logits, texts: list[str]) -> list[int]:
    """Per-decision outcome: v1 -> convert?(bool) on candidates; v2 -> argmax class where >1 action is legal."""
    out = []
    for row, t in zip(logits, texts):
        if row.ndim == 1:
            out += [int(row[i] > 0) for i, c in enumerate(t) if c in CANDIDATES]
        else:
            out += [int(row[i].argmax()) for i in range(len(t)) if len(allowed_actions(t, i)) > 1]
    return out


def run_onnx(path: Path, vocab, texts: list[str]) -> np.ndarray:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    rows = []
    for t in texts:  # batch 1: identical to how the browser runs BiLSTM
        ids = np.array([vocab.encode(t)], dtype=np.int64)
        rows.append(sess.run(["logits"], {"ids": ids, "mask": np.ones_like(ids, dtype=bool)})[0][0])
    return rows


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=None, help="processed data dir; val.txt sentences check int8 agreement")
    ap.add_argument("--n-check", type=int, default=500)
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    r = Restorer.from_checkpoint(args.ckpt, device="cpu")
    st = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    kind = st["cfg"]["model"]["type"]
    if kind not in ("bilstm", "transformer"):
        raise SystemExit(f"ONNX export supports bilstm/transformer, not {kind!r} (serve it via the API instead)")
    model = r.model.eval()

    ids = torch.tensor([r.vocab.encode("seher geldi")], dtype=torch.long)
    mask = torch.ones_like(ids, dtype=torch.bool)
    if hasattr(torch.backends, "mha"):
        torch.backends.mha.set_fastpath_enabled(False)  # fused MHA kernel is not exportable
    onnx_path = out / "model.onnx"
    torch.onnx.export(
        model, (ids, mask), str(onnx_path), input_names=["ids", "mask"], output_names=["logits"],
        dynamic_axes={"ids": {0: "B", 1: "T"}, "mask": {0: "B", 1: "T"}, "logits": {0: "B", 1: "T"}},
        opset_version=17, dynamo=False,
    )

    texts = list(SAMPLE)
    if args.data and (Path(args.data) / "val.txt").exists():
        lines = (Path(args.data) / "val.txt").read_text(encoding="utf-8").splitlines()
        texts += [strip(l) for l in lines[: args.n_check] if l.strip()]
    texts = [t[: r.max_len] for t in texts]
    ref = decisions(run_onnx(onnx_path, r.vocab, texts), texts)

    default_file, agreement = "model.onnx", None
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        q_path = out / "model.int8.onnx"
        quantize_dynamic(str(onnx_path), str(q_path), weight_type=QuantType.QInt8)
        q = decisions(run_onnx(q_path, r.vocab, texts), texts)
        agreement = float(np.mean([a == b for a, b in zip(ref, q)])) if ref else 1.0
        if agreement >= 0.999:
            default_file = "model.int8.onnx"
    except Exception as e:  # quantisation is an optimisation, never a blocker
        print("int8 quantisation skipped:", e)

    (out / "vocab.json").write_text(json.dumps(r.vocab.chars, ensure_ascii=False), encoding="utf-8")
    meta = {
        "model_type": kind,
        "scheme": st["cfg"]["data"].get("scheme", "v1"),
        "num_classes": st["cfg"]["model"].get("num_classes", 1),
        "a_bias": st["cfg"].get("infer", {}).get("a_bias", 0.0),
        "max_len": r.max_len,
        "threshold": 0.5,
        "batch_size": 1 if kind == "bilstm" else 32,
        "default_file": default_file,
        "int8_decision_agreement": agreement,
        "params": sum(p.numel() for p in model.parameters()),
        "files_mb": {f.name: round(os.path.getsize(f) / 1e6, 2) for f in out.glob("*.onnx")},
        "source_ckpt": str(args.ckpt),
        "step": st.get("step"),
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return meta


if __name__ == "__main__":
    main()
