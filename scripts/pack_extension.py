"""Assemble the browser bundle: shared JS core, exported model, onnxruntime-web files.

    python scripts/fetch_ort.py                                   # once: downloads ORT into vendor/ort
    python scripts/pack_extension.py --model exported/bilstm_v1   # -> extension/ and web/ ready to load
    python scripts/pack_extension.py --sync-only                  # only copy web/shared -> extension/shared
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_FILES = ("model.onnx", "model.int8.onnx", "vocab.json", "meta.json")


def copy_tree_files(src: Path, dst: Path, names=None) -> list[str]:
    dst.mkdir(parents=True, exist_ok=True)
    copied = []
    for f in sorted(src.iterdir()):
        if f.is_file() and (names is None or f.name in names):
            shutil.copy2(f, dst / f.name)
            copied.append(f.name)
    return copied


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="exported model dir (from python -m xettat.export)")
    ap.add_argument("--vendor", default=str(ROOT / "vendor" / "ort"), help="onnxruntime-web files")
    ap.add_argument("--extension", default=str(ROOT / "extension"))
    ap.add_argument("--web", default=str(ROOT / "web"))
    ap.add_argument("--sync-only", action="store_true")
    args = ap.parse_args(argv)

    ext = Path(args.extension)
    print("shared ->", copy_tree_files(ROOT / "web" / "shared", ext / "shared"))
    if args.sync_only:
        return
    if not args.model:
        ap.error("--model is required unless --sync-only")
    vendor = Path(args.vendor)
    if not vendor.exists():
        raise SystemExit(f"{vendor} not found: run  python scripts/fetch_ort.py  first")
    for target in (ext, Path(args.web)):
        print(target, "model ->", copy_tree_files(Path(args.model), target / "model", MODEL_FILES))
        print(target, "vendor ->", copy_tree_files(vendor, target / "vendor"))


if __name__ == "__main__":
    main()
