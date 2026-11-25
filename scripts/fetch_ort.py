"""Download onnxruntime-web (WASM build) into vendor/ort. Needs internet; run on any machine with Python.

    python scripts/fetch_ort.py
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

VERSION = "1.20.1"
BASE = f"https://cdn.jsdelivr.net/npm/onnxruntime-web@{VERSION}/dist/"
FILES = ["ort.wasm.min.mjs", "ort-wasm-simd-threaded.mjs", "ort-wasm-simd-threaded.wasm"]
OUT = Path(__file__).resolve().parents[1] / "vendor" / "ort"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        dst = OUT / name
        print(f"downloading {name} ...", end=" ", flush=True)
        with urllib.request.urlopen(BASE + name, timeout=120) as r:
            dst.write_bytes(r.read())
        print(f"{dst.stat().st_size / 1e6:.1f} MB")
    (OUT / "VERSION").write_text(VERSION)
    print("done ->", OUT)


if __name__ == "__main__":
    main()
