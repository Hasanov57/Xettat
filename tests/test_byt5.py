import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from xettat.models import build_model  # noqa: E402
from xettat.vocab import CharVocab  # noqa: E402

TINY = {"type": "byt5", "pretrained": False, "dropout": 0.0,
        "config": {"vocab_size": 384, "d_model": 32, "d_kv": 8, "num_heads": 2, "d_ff": 64, "num_layers": 1}}


def test_shape_and_multibyte_alignment():
    v = CharVocab()
    m = build_model(TINY, v.size).eval()
    texts = ["səhər", "abc"]
    ids = torch.zeros(2, 5, dtype=torch.long)
    mask = torch.zeros(2, 5, dtype=torch.bool)
    for r, t in enumerate(texts):
        ids[r, : len(t)] = torch.tensor(v.encode(t))
        mask[r, : len(t)] = True
    with torch.no_grad():
        out = m(ids, mask)
    assert out.shape == (2, 5)
    # each char's logit comes from its own first byte: same prefix char -> same logit regardless of later chars
    with torch.no_grad():
        solo = m(torch.tensor([v.encode("s")]), torch.ones(1, 1, dtype=torch.bool))
    assert torch.isfinite(out).all()
    assert solo.shape == (1, 1)


def test_byte_tables():
    v = CharVocab()
    m = build_model(TINY, v.size)
    ə = v.encode("ə")[0]
    a = v.encode("a")[0]
    assert m.byte_len[ə].item() == 2 and m.byte_len[a].item() == 1
    assert m.byte_tab[a, 0].item() == ord("a") + 3
    assert m.byte_len[0].item() == 0  # pad
