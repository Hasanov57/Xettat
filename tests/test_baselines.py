from xettat.baselines import Bigram, Identity, WordFreq


def test_identity():
    assert Identity().predict("seher geldi") == "seher geldi"


def test_wordfreq_basic_and_case():
    wf = WordFreq().fit(["Səhər gəldi", "səhər tezdən", "şəhər böyükdür"])
    assert wf.predict("seher geldi") == "səhər gəldi"
    assert wf.predict("SEHER") == "SƏHƏR"
    assert wf.predict("Seher") == "Səhər"
    assert wf.predict("xyz 123, seher!") == "xyz 123, səhər!"


def test_wordfreq_keeps_typed_az_letters():
    wf = WordFreq().fit(["səhər"])
    # unknown word containing an Azerbaijani letter stays as typed
    assert wf.predict("qəlb") == "qəlb"


def test_wordfreq_ambiguous():
    wf = WordFreq().fit(["səhər", "şəhər", "gəl"])
    assert wf.ambiguous() == {"seher"}


def test_bigram_context():
    bg = Bigram().fit(["böyük şəhər", "böyük şəhər", "sabah səhər", "səhər", "səhər"])
    assert bg.predict("boyuk seher") == "böyük şəhər"
    assert bg.predict("sabah seher") == "sabah səhər"
    assert bg.predict("seher") == "səhər"


def test_run_baselines_cli(tmp_path):
    import json

    from xettat.run_baselines import main

    d = tmp_path / "data"
    d.mkdir()
    (d / "train.txt").write_text("Səhər şəhərə getdim.\nBöyük şəhər gözəldir.\n", encoding="utf-8")
    (d / "val.txt").write_text("Səhər getdim.\n", encoding="utf-8")
    out = tmp_path / "runs"
    main(["--data", str(d), "--split", "val", "--out", str(out)])
    for name in ("identity", "wordfreq", "bigram"):
        r = json.loads((out / f"{name}_val.json").read_text())
        assert "word_acc" in r
    assert json.loads((out / "wordfreq_val.json").read_text())["sentence_acc"] == 1.0
