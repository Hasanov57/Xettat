from xettat.metrics import score
from xettat.text import strip

T = ["Səhər şəhərə getdim.", "Bu gün hava gözəldir."]


def test_perfect():
    m = score([strip(t) for t in T], T, T)
    assert m["char_acc"] == 1.0
    assert m["word_acc"] == 1.0
    assert m["sentence_acc"] == 1.0
    assert m["wer"] == 0.0
    assert m["corruption_rate"] == 0.0
    assert m["per_pair"]["e/ə"]["f1"] == 1.0
    assert m["n_sentences"] == 2


def test_identity_on_seher():
    m = score(["seher"], ["seher"], ["səhər"])
    assert m["word_acc"] == 0.0
    assert m["sentence_acc"] == 0.0
    assert m["per_pair"]["e/ə"]["r"] == 0.0


def test_corruption():
    # user already typed "ə" correctly in the input, model turned it back to "e"
    m = score(["səher"], ["seher"], ["səhər"])
    assert m["corruption_rate"] == 1.0


def test_ambiguous_subset():
    m = score(["seher gel"], ["şəhər gəl"], ["səhər gəl"], ambiguous={"seher"})
    assert m["ambig_word_acc"] == 0.0
    assert m["word_acc"] == 0.5


def test_length_mismatch_counts_as_wrong():
    m = score(["seher"], ["səhərr"], ["səhər"])
    assert m["word_acc"] == 0.0 and m["sentence_acc"] == 0.0
