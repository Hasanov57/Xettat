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


def test_is_good_sentence_rejects_old_orthography():
    assert not is_good_sentence("Män säni sevirәm, bu gün gözäl gündür.")


def test_az_ratio():
    from xettat.data.clean import az_ratio

    assert az_ratio("Mən səni sevirəm") > 0.15
    assert az_ratio("Men seni sevirem") == 0.0
    assert az_ratio("123 !!") == 0.0


def test_detokenize_opus():
    from xettat.data.clean import detokenize

    assert detokenize("Salam , necəsən ?") == "Salam, necəsən?"
    assert detokenize("« Gəl » dedi ( yavaşca ) .") == "«Gəl» dedi (yavaşca)."
    assert detokenize("Bu 100 % doğrudur !") == "Bu 100% doğrudur!"


def test_prepare_doc_level_filter(tmp_path):
    good = "Mən bu gün səhər tezdən çörək almağa getdim. Sonra çay içdik və söhbət etdik."
    bad = "Men bu gun seher tezden corek almaga getdim. Sonra cay icdik ve sohbet etdik ğ."
    src = tmp_path / "c.txt"
    src.write_text("\n\n".join([good] * 50 + [bad] * 50), encoding="utf-8")
    out = tmp_path / "o"
    prepare_main(["--source", f"txt:{src}", "--out", str(out), "--seed", "1", "--val-frac", "0", "--test-frac", "0",
                  "--doc-min-az-ratio", "0.08"])
    train = (out / "train.txt").read_text(encoding="utf-8")
    assert "Men bu gun" not in train and "Mən bu gün" in train


def test_prepare_max_train(tmp_path):
    src = tmp_path / "c.txt"
    src.write_text("\n\n".join(f"Bu {i} nömrəli cümlədir və çox gözəldir." for i in range(200)), encoding="utf-8")
    out = tmp_path / "o"
    prepare_main(["--source", f"txt:{src}", "--out", str(out), "--val-frac", "0", "--test-frac", "0",
                  "--max-train", "50"])
    assert len((out / "train.txt").read_text(encoding="utf-8").splitlines()) == 50


def test_mix(tmp_path):
    from xettat.data.mix import main as mix_main

    a, b = tmp_path / "a", tmp_path / "b"
    for d, tag in ((a, "vikipediya"), (b, "söhbət")):
        d.mkdir()
        (d / "train.txt").write_text("\n".join(f"{tag} cümləsi {i}" for i in range(100)) + "\n", encoding="utf-8")
        (d / "val.txt").write_text("\n".join(f"{tag} yoxlama {i}" for i in range(30)) + "\n", encoding="utf-8")
        (d / "test.txt").write_text(f"{tag} test\n", encoding="utf-8")
    # a leaked copy of a's held-out sentence inside b's train must be removed
    with open(b / "train.txt", "a", encoding="utf-8") as f:
        f.write("vikipediya yoxlama 3\n")
    out = tmp_path / "mix"
    mix_main(["--sources", str(a), str(b), "--out", str(out), "--val-per-source", "10", "--seed", "0"])
    train = (out / "train.txt").read_text(encoding="utf-8").splitlines()
    val = (out / "val.txt").read_text(encoding="utf-8").splitlines()
    assert len(train) == 200 and "vikipediya yoxlama 3" not in train
    assert len(val) == 20
    assert train != sorted(train)  # shuffled
    mix_main(["--sources", str(a), str(b), "--out", str(tmp_path / "mix2"), "--val-per-source", "10", "--seed", "0"])
    assert (tmp_path / "mix2" / "train.txt").read_text() == (out / "train.txt").read_text()  # deterministic
