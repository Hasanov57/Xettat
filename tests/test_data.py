import random

from xettat.data.augment import make_input
from xettat.data.clean import is_good_sentence, split_sentences
from xettat.data.prepare import main as prepare_main
from xettat.data.prepare import split_docs
from xettat.text import strip

AZ = "Bu gün hava çox gözəl idi və biz parka getdik."


def test_split_sentences():
    para = "Bakı Azərbaycanın paytaxtıdır. Şəhər Xəzər sahilindədir! Siz orada olmusunuz?"
    assert split_sentences(para) == [
        "Bakı Azərbaycanın paytaxtıdır.",
        "Şəhər Xəzər sahilindədir!",
        "Siz orada olmusunuz?",
    ]


def test_split_sentences_newlines_and_abbrev():
    text = "Alma, armud və s. meyvələr satılır.\nİkinci sətir burdadır."
    assert split_sentences(text) == [
        "Alma, armud və s. meyvələr satılır.",
        "İkinci sətir burdadır.",
    ]


def test_is_good_sentence():
    assert is_good_sentence(AZ)
    assert not is_good_sentence("ok")
    assert not is_good_sentence("Это предложение написано по-русски, а не по-азербайджански.")
    assert not is_good_sentence("This sentence is plain English without special letters.")


def test_split_docs_deterministic_disjoint():
    ids = [f"doc{i}" for i in range(10000)]
    a = split_docs(ids, seed=42)
    b = split_docs(ids, seed=42)
    assert a == b
    assert not (a["train"] & a["val"]) and not (a["train"] & a["test"]) and not (a["val"] & a["test"])
    assert len(a["train"]) + len(a["val"]) + len(a["test"]) == 10000
    assert 50 <= len(a["val"]) <= 150 and 50 <= len(a["test"]) <= 150


def test_prepare_txt_source(tmp_path):
    docs = []
    for d in range(300):
        docs.append(f"Bu {d} nömrəli məqalədir və çox maraqlıdır. Burada {d} haqqında söhbət gedir.")
    src = tmp_path / "corpus.txt"
    src.write_text("\n\n".join(docs), encoding="utf-8")
    out = tmp_path / "out"
    prepare_main(["--source", f"txt:{src}", "--out", str(out), "--seed", "42", "--val-frac", "0.1", "--test-frac", "0.1"])
    sets = {n: set((out / f"{n}.txt").read_text(encoding="utf-8").splitlines()) for n in ("train", "val", "test")}
    assert sets["train"] and sets["val"] and sets["test"]
    assert not (sets["train"] & sets["val"]) and not (sets["train"] & sets["test"]) and not (sets["val"] & sets["test"])
    assert (out / "stats.json").exists()


def test_make_input_modes():
    rng = random.Random(0)
    assert make_input(AZ, rng, p_partial=0.0, p_upper=0.0) == (strip(AZ), AZ)
    assert make_input(AZ, rng, p_partial=1.0, p_keep=1.0, p_upper=0.0) == (AZ, AZ)
    for _ in range(200):
        inp, tgt = make_input(AZ, rng)
        assert len(inp) == len(tgt) == len(AZ)
        assert strip(inp) == strip(tgt)


def test_make_input_upper():
    rng = random.Random(0)
    assert make_input("çiçək", rng, p_partial=0.0, p_upper=1.0) == ("CICEK", "ÇİÇƏK")
