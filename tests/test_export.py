import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")
import numpy as np  # noqa: E402

from xettat.models import build_model  # noqa: E402
from xettat.vocab import CharVocab  # noqa: E402

CFGS = {
    "bilstm": {"type": "bilstm", "emb_dim": 16, "hidden": 16, "layers": 2, "dropout": 0.0},
    "transformer": {"type": "transformer", "d_model": 32, "heads": 2, "layers": 1, "ff": 64, "dropout": 0.0, "max_len": 64},
}


@pytest.mark.parametrize("kind", ["bilstm", "transformer"])
def test_export_matches_torch(tmp_path, kind):
    from xettat.export import main

    v = CharVocab()
    m = build_model(CFGS[kind], v.size).eval()
    ck = tmp_path / "best.pt"
    torch.save({"model": m.state_dict(), "cfg": {"model": CFGS[kind], "data": {"max_len": 64}}, "vocab": v.to_json()}, ck)
    out = tmp_path / "exp"
    main(["--ckpt", str(ck), "--out", str(out)])
    meta = json.loads((out / "meta.json").read_text())
    assert meta["model_type"] == kind and meta["max_len"] == 64 and meta["default_file"] in {"model.onnx", "model.int8.onnx"}
    assert json.loads((out / "vocab.json").read_text()) == v.chars

    text = "seher saat 9-da şeherde gorusek"
    ids = torch.tensor([v.encode(text)])
    mask = torch.ones_like(ids, dtype=torch.bool)
    with torch.no_grad():
        ref = m(ids, mask).numpy()
    sess = ort.InferenceSession(str(out / "model.onnx"))
    got = sess.run(["logits"], {"ids": ids.numpy(), "mask": mask.numpy()})[0]
    np.testing.assert_allclose(got, ref, atol=1e-4)
