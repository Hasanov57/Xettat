import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from xettat.models import build_model  # noqa: E402
from xettat.vocab import CharVocab  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    cfg = {"model": {"type": "bilstm", "emb_dim": 16, "hidden": 16, "layers": 1, "dropout": 0.0}, "data": {"max_len": 64}}
    v = CharVocab()
    p = tmp_path / "best.pt"
    torch.save({"model": build_model(cfg["model"], v.size).state_dict(), "cfg": cfg, "vocab": v.to_json()}, p)
    monkeypatch.setenv("MODEL_FAST", str(p))
    monkeypatch.delenv("MODEL_ACCURATE", raising=False)
    import importlib

    import serve.app as app_mod

    importlib.reload(app_mod)
    return TestClient(app_mod.app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["models"]["fast"] is True


def test_restore_keeps_length(client):
    r = client.post("/restore", json={"text": "seher ə 123", "model": "fast"})
    assert r.status_code == 200
    body = r.json()
    assert len(body["text"]) == len("seher ə 123") and body["text"][6] == "ə"
    assert {"changed", "ms", "model"} <= set(body)


def test_accurate_falls_back_to_fast(client):
    r = client.post("/restore", json={"text": "seher", "model": "accurate"})
    assert r.status_code == 200 and r.json()["model"] == "fast"


def test_limits(client):
    assert client.post("/restore", json={"text": "a" * 20001}).status_code == 413
    assert client.post("/restore", content=b"not json", headers={"content-type": "application/json"}).status_code == 422
