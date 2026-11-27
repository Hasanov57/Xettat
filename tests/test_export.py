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


def test_transformer_export_path_equals_module_path():
    """The hand-written export encoder must match nn.TransformerEncoder exactly (padding included)."""
    from unittest import mock

    v = CharVocab()
    m = build_model(CFGS["transformer"], v.size).eval()
    ids = torch.randint(2, 60, (3, 23))
    mask = torch.ones(3, 23, dtype=torch.bool)
    mask[1, 15:] = False
    with torch.no_grad():
        ref = m(ids, mask)
        with mock.patch("torch.onnx.is_in_onnx_export", return_value=True):
            got = m(ids, mask)
    torch.testing.assert_close(got[mask], ref[mask], atol=1e-5, rtol=1e-4)


def test_export_v2_three_classes(tmp_path):
    from xettat.export import main

    v = CharVocab()
    cfg = dict(CFGS["bilstm"], num_classes=3)
    m = build_model(cfg, v.size).eval()
    ck = tmp_path / "v2.pt"
    torch.save({"model": m.state_dict(), "cfg": {"model": cfg, "data": {"max_len": 64, "scheme": "v2"}},
                "vocab": v.to_json()}, ck)
    meta = main(["--ckpt", str(ck), "--out", str(tmp_path / "e")])
    assert meta["scheme"] == "v2" and meta["num_classes"] == 3
    ids = torch.tensor([v.encode("sheher adad")])
    mask = torch.ones_like(ids, dtype=torch.bool)
    with torch.no_grad():
        ref = m(ids, mask).numpy()
    got = ort.InferenceSession(str(tmp_path / "e" / "model.onnx")).run(["logits"], {"ids": ids.numpy(), "mask": mask.numpy()})[0]
    np.testing.assert_allclose(got, ref, atol=1e-4)
