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


def test_misaligned_digraph_scored_by_words():
    # input had 'sh' (longer), prediction restored it -> same as target
    m = score(["sheher var"], ["şəhər var"], ["şəhər var"])
    assert m["word_acc"] == 1.0 and m["sentence_acc"] == 1.0
    # prediction failed to delete the 'h'
    m = score(["sheher gel"], ["şhəhər gəl"], ["şəhər gəl"])
    assert m["word_acc"] == 0.5 and m["sentence_acc"] == 0.0


def test_a_false_conversion_rate():
    m = score(["ata adad"], ["ətə ədəd"], ["ata ədəd"])
    assert m["a_false_conversion_rate"] == 1.0  # both genuine a's of 'ata' were converted
    m = score(["ata adad"], ["ata ədəd"], ["ata ədəd"])
    assert m["a_false_conversion_rate"] == 0.0


def test_english_corruption_rate():
    m = score(["meeting sabah"], ["meetıng sabah"], ["meeting sabah"])
    assert m["english_corruption_rate"] == 1.0
    m = score(["meeting sabah"], ["meeting sabah"], ["meeting sabah"])
    assert m["english_corruption_rate"] == 0.0
