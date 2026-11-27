import random

from xettat.text import (CONVERT, DELETE, IGNORE, KEEP, allowed_actions, apply_actions, make_variant_pair,
                         protected_mask, strip)

AZ = "Şəhərdə çiçəkli bağçalar var, ata ilə ağac əkdik."


def test_apply_actions_digraph_and_a():
    assert apply_actions("sheher", [CONVERT, DELETE, CONVERT, KEEP, CONVERT, KEEP]) == "şəhər"
    assert apply_actions("adad", [CONVERT, KEEP, CONVERT, KEEP]) == "ədəd"
    assert apply_actions("SHEHER", [CONVERT, DELETE, CONVERT, KEEP, CONVERT, KEEP]) == "ŞƏHƏR"


def test_delete_only_after_s_c_g():
    # 'h' after 'e' may not be deleted even if asked
    assert apply_actions("eh", [KEEP, DELETE]) == "eh"
    assert allowed_actions("ah", 1) == {KEEP}
    assert allowed_actions("ch", 1) == {KEEP, DELETE}
    assert allowed_actions("ab", 0) == {KEEP, CONVERT}
    assert allowed_actions("1b", 0) == set()


def test_protected_mask():
    t = "bax https://x.az/sehife ve @seher #gozel mail: ali.veli@gmail.com son"
    m = protected_mask(t)
    for span in ("https://x.az/sehife", "@seher", "#gozel", "ali.veli@gmail.com"):
        i = t.index(span)
        assert all(m[i:i + len(span)]), span
    assert not any(m[: t.index("https")]) and not m[t.index("ve")]
    assert len(m) == len(t)


def test_protected_spans_are_never_changed():
    t = "seher @seher"
    labels = [CONVERT] * len(t)
    out = apply_actions(t, labels, protected=protected_mask(t))
    assert out.endswith("@seher")


def test_variant_pair_std_matches_v1_strip():
    inp, labels, tgt = make_variant_pair(AZ, random.Random(0), style="std", p_partial=0, p_upper=0, p_english=0)
    assert tgt == AZ and inp == strip(AZ)
    assert apply_actions(inp, labels) == AZ


def test_variant_pair_digraph():
    inp, labels, tgt = make_variant_pair("şəhər çay ağ", random.Random(0), style="digraph", p_partial=0,
                                         p_upper=0, p_english=0, p_digraph=1.0)
    assert inp == "sheher chay agh"
    assert labels[:2] == [CONVERT, DELETE]
    assert apply_actions(inp, labels) == tgt


def test_variant_pair_a_style():
    inp, labels, tgt = make_variant_pair("ədəd ata", random.Random(0), style="a", p_partial=0, p_upper=0,
                                         p_english=0, p_a=1.0)
    assert inp == "adad ata"
    assert labels[2] == KEEP or labels[0] == CONVERT
    assert labels[5] == KEEP  # genuine 'a' in 'ata' is a candidate labelled KEEP
    assert apply_actions(inp, labels) == tgt


def test_variant_pair_roundtrip_random():
    rng = random.Random(1)
    for _ in range(300):
        inp, labels, tgt = make_variant_pair(AZ, rng)
        assert len(inp) == len(labels)
        assert apply_actions(inp, labels) == tgt
        assert all(l == IGNORE or l in (KEEP, CONVERT, DELETE) for l in labels)


def test_genuine_sh_is_keep():
    inp, labels, tgt = make_variant_pair("İshaq", random.Random(0), style="std", p_partial=0, p_upper=0,
                                         p_english=0)
    assert inp == "Ishaq" and labels[2] == KEEP


def test_english_insertion_is_kept():
    rng = random.Random(3)
    seen = False
    for _ in range(50):
        inp, labels, tgt = make_variant_pair("bu gün görüş var", rng, style="std", p_english=1.0, p_partial=0,
                                             p_upper=0)
        assert apply_actions(inp, labels) == tgt
        if len(tgt.split()) > 4:
            seen = True
    assert seen


def test_english_guard_protects_whole_words_case_insensitive():
    t = "qrafikde Curve yarandi, curved deyil, curve-in"
    m = protected_mask(t, english={"curve"})
    i = t.index("Curve")
    assert all(m[i:i + 5])
    j = t.index("curved")
    assert not any(m[j:j + 6])  # only whole words
    k = t.index("curve-in")
    assert all(m[k:k + 5]) and not m[k + 6]
    assert not m[t.index("qrafikde")]


def test_build_english_guard(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    en = tmp_path / "en.txt"
    en.write_text("the\nson\ncurve\nmeeting\nbar\nok\nxyzzy\ngel\n", encoding="utf-8")
    corpus = tmp_path / "az.txt"
    # web text also contains the English words themselves ("meeting", "curve") — they must not count
    corpus.write_text("Mənim oğlum son dəfə bara getdi, gəl.\n" * 5 + "Bar və son. Sabah meeting var, curve.\n" * 9,
                      encoding="utf-8")
    out = tmp_path / "guard.txt"
    subprocess.run([sys.executable, str(root / "scripts" / "build_english_guard.py"), "--en", str(en),
                    "--corpus", str(corpus), "--min-az-count", "3", "--out", str(out)], check=True)
    words = set(out.read_text().split())
    assert {"curve", "meeting"} <= words  # English words seen as-is in Azerbaijani web text still protected
    assert "gel" not in words  # = gəl without diacritics: protecting it would block restoring gəl
    assert {"son", "bar"} <= words  # harmless: their correct Azerbaijani form has no diacritics anyway
    assert "ok" not in words  # shorter than 3 letters: too risky (Azerbaijani short words)
    assert "xyzzy" not in words  # no candidate letter -> nothing to protect
