import json

import pytest

torch = pytest.importorskip("torch")

from xettat.dataset import SentenceDataset, collate  # noqa: E402
from xettat.models import build_model  # noqa: E402
from xettat.text import make_labels  # noqa: E402
from xettat.vocab import CharVocab  # noqa: E402

TINY = {
    "bilstm": {"type": "bilstm", "emb_dim": 16, "hidden": 16, "layers": 1, "dropout": 0.0},
    "transformer": {"type": "transformer", "d_model": 32, "heads": 2, "layers": 1, "ff": 64, "dropout": 0.0, "max_len": 64},
}


@pytest.mark.parametrize("kind", ["bilstm", "transformer"])
def test_model_shapes(kind):
    m = build_model(TINY[kind], CharVocab().size)
    ids = torch.randint(2, 50, (2, 17))
    mask = torch.ones(2, 17, dtype=torch.bool)
    mask[1, 10:] = False
    out = m(ids, mask)
    assert out.shape == (2, 17)


def test_collate_pads():
    ids, mask, labels = collate([([5, 6, 7], [1, 0, -100]), ([8], [0])])
    assert ids.tolist() == [[5, 6, 7], [8, 0, 0]]
    assert mask.tolist() == [[True, True, True], [True, False, False]]
    assert labels.tolist() == [[1, 0, -100], [0, -100, -100]]


def test_dataset_items(data_dir):
    v = CharVocab()
    ds = SentenceDataset(data_dir / "val.txt", v, max_len=64, augment=False, seed=0)
    ids, labels = ds[0]
    src, y = make_labels(ds.sentences[0])
    assert len(ids) == len(labels) == len(src)
    assert labels == y


def test_dataset_crops_long(tmp_path):
    p = tmp_path / "long.txt"
    p.write_text("şəhər " * 50 + "\n", encoding="utf-8")
    ds = SentenceDataset(p, CharVocab(), max_len=32, augment=True, seed=0)
    ids, labels = ds[0]
    assert len(ids) == len(labels) == 32


def _cfg(tmp_path, data_dir, max_steps):
    import yaml

    cfg = yaml.safe_load(open("configs/debug.yaml"))
    cfg["data"]["dir"] = str(data_dir)
    cfg["ckpt_root"] = str(tmp_path / "ckpt")
    cfg["runs_root"] = str(tmp_path / "runs")
    cfg["train"]["max_steps"] = max_steps
    p = tmp_path / f"cfg{max_steps}.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_train_and_resume(tmp_path, data_dir):
    from xettat.train import main

    main(["--config", str(_cfg(tmp_path, data_dir, 30)), "--run-id", "dbg"])
    run = tmp_path / "runs" / "dbg"
    assert (run / "metrics.jsonl").exists() and (run / "config.yaml").exists()
    assert (tmp_path / "ckpt" / "dbg" / "last.pt").exists()
    assert (tmp_path / "ckpt" / "dbg" / "best.pt").exists()
    n_before = len((run / "metrics.jsonl").read_text().splitlines())

    main(["--config", str(_cfg(tmp_path, data_dir, 40)), "--run-id", "dbg"])
    lines = [json.loads(l) for l in (run / "metrics.jsonl").read_text().splitlines()]
    new = [l for l in lines[n_before:] if "step" in l]
    assert new and min(l["step"] for l in new) > 30
