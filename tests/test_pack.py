import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_pack_copies_model_shared_and_vendor(tmp_path):
    exported = tmp_path / "exported" / "bilstm_v1"
    exported.mkdir(parents=True)
    (exported / "model.onnx").write_bytes(b"onnx")
    (exported / "vocab.json").write_text("[]")
    (exported / "meta.json").write_text(json.dumps({"max_len": 512}))
    vendor = tmp_path / "vendor"
    vendor.mkdir()
    (vendor / "ort.wasm.min.mjs").write_text("// ort")
    ext = tmp_path / "ext"
    web = tmp_path / "web"
    for d in (ext, web):
        d.mkdir()
    subprocess.run([sys.executable, str(ROOT / "scripts" / "pack_extension.py"), "--model", str(exported),
                    "--vendor", str(vendor), "--extension", str(ext), "--web", str(web)], check=True)
    for d in (ext, web):
        assert (d / "model" / "model.onnx").read_bytes() == b"onnx"
        assert (d / "model" / "meta.json").exists()
        assert (d / "vendor" / "ort.wasm.min.mjs").exists()
    assert (ext / "shared" / "xettat-core.js").read_text() == (ROOT / "web" / "shared" / "xettat-core.js").read_text()


def test_pack_sync_only(tmp_path):
    ext = tmp_path / "ext"
    ext.mkdir()
    subprocess.run([sys.executable, str(ROOT / "scripts" / "pack_extension.py"), "--extension", str(ext),
                    "--sync-only"], check=True)
    assert (ext / "shared" / "xettat-ort.js").exists()
