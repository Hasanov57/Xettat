from xettat.vocab import CharVocab


def test_vocab_fixed_and_roundtrip():
    v = CharVocab()
    assert v.size == CharVocab().size  # data independent
    ids = v.encode("Səhər 1!😀")
    assert len(ids) == 9
    assert ids[-1] == CharVocab.UNK  # emoji unknown
    assert 0 not in ids  # 0 is padding only
    assert CharVocab.from_json(v.to_json()).encode("Səhər") == v.encode("Səhər")


def test_vocab_covers_all_az_letters():
    v = CharVocab()
    assert CharVocab.UNK not in v.encode("əıöüçşğƏİÖÜÇŞĞ«»–—…’“”")
