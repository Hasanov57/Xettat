import json

import pytest

torch = pytest.importorskip("torch")

from xettat.infer import windows  # noqa: E402
from xettat.models import build_model  # noqa: E402
from xettat.restorer import Restorer  # noqa: E402
from xettat.vocab import CharVocab  # noqa: E402


def test_windows_cover_everything():
    assert windows(10, 64, 16) == [(0, 10)]
    w = windows(3000, 512, 64)
    assert w[0][0] == 0 and w[-1][1] == 3000
    assert all(e - s == 512 for s, e in w)
    covered = set()
    for s, e in w:
        covered |= set(range(s, e))
    assert covered == set(range(3000))


@pytest.fixture
def tiny_ckpt(tmp_path):
    cfg = {"model": {"type": "transformer", "d_model": 32, "heads": 2, "layers": 1, "ff": 64, "dropout": 0.0,
                     "max_len": 64}, "data": {"max_len": 64}}
    v = CharVocab()
    m = build_model(cfg["model"], v.size)
    p = tmp_path / "best.pt"
    torch.save({"model": m.state_dict(), "cfg": cfg, "vocab": v.to_json(), "step": 0}, p)
    return p


def test_restorer_invariants(tiny_ckpt):
    r = Restorer.from_checkpoint(str(tiny_ckpt))
    assert r.restore("") == ""
    assert r.restore("   \n ") == "   \n "
    s = "https://x.az 123 😀"
    assert r.restore(s).replace("ş", "s").replace("ç", "c") == s  # only candidates may change
    out = r.restore("səhər")
    assert out[1] == "ə" and out[3] in "eə" and len(out) == 5
    long = "seher " * 500
    assert len(r.restore(long)) == len(long)
    assert len(r.restore("bir\niki\n\nuc")) == len("bir\niki\n\nuc")


def test_restore_batch(tiny_ckpt):
    r = Restorer.from_checkpoint(str(tiny_ckpt))
    assert len(r.restore_batch(["a", "seher", ""])) == 3


def test_evaluate_cli(tiny_ckpt, data_dir, tmp_path):
    from xettat.evaluate import main

    out = tmp_path / "run"
    main(["--ckpt", str(tiny_ckpt), "--data", str(data_dir), "--split", "val", "--out", str(out)])
    r = json.loads((out / "final_val.json").read_text())
    assert "word_acc" in r and "latency_ms_per_sentence_cpu" in r and "params" in r


def test_evaluate_tag(tiny_ckpt, data_dir, tmp_path):
    from xettat.evaluate import main

    out = tmp_path / "run"
    main(["--ckpt", str(tiny_ckpt), "--data", str(data_dir), "--split", "val", "--out", str(out), "--tag", "hplt"])
    assert (out / "final_val_hplt.json").exists() and (out / "errors_val_hplt.jsonl").exists()


@pytest.fixture
def tiny_ckpt_v2(tmp_path):
    cfg = {"model": {"type": "bilstm", "emb_dim": 16, "hidden": 16, "layers": 1, "dropout": 0.0, "num_classes": 3},
           "data": {"max_len": 64, "scheme": "v2"}}
    v = CharVocab()
    p = tmp_path / "v2.pt"
    torch.save({"model": build_model(cfg["model"], v.size).state_dict(), "cfg": cfg, "vocab": v.to_json()}, p)
    return p


def test_restorer_v2_invariants(tiny_ckpt_v2):
    r = Restorer.from_checkpoint(str(tiny_ckpt_v2))
    for s in ["sheher chay agh", "SHEHER", "ata adad", "bax https://x.az/sehife @seher", "", "eh oh"]:
        out = r.restore(s)
        assert len(out) <= len(s)
    assert r.restore("bax https://x.az/sehife @seher").endswith("https://x.az/sehife @seher")
    assert r.restore("eh oh") in {"eh oh", "əh oh", "eh öh", "əh öh"}  # h after e/o is never deleted
    long = "sheher " * 300
    assert len(r.restore(long)) <= len(long)
