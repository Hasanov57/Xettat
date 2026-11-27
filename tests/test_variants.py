from xettat.data.variants import STYLES, main
from xettat.text import strip


def test_variant_eval_sets(data_dir, tmp_path):
    out = tmp_path / "var"
    main(["--data", str(data_dir), "--split", "val", "--out", str(out), "--n", "20", "--seed", "7"])
    for style in STYLES:
        rows = [l.split("\t") for l in (out / f"val_{style}.tsv").read_text(encoding="utf-8").splitlines()[1:]]
        assert len(rows) == 20, style
        for inp, tgt in rows:
            if style == "std":
                assert inp == strip(tgt)
    digraph = (out / "val_digraph.tsv").read_text(encoding="utf-8")
    assert "sh" in digraph or "ch" in digraph or "gh" in digraph
    english = [l.split("\t")[1] for l in (out / "val_english.tsv").read_text(encoding="utf-8").splitlines()[1:]]
    assert any(len(t.split()) > 5 for t in english)
    # deterministic
    main(["--data", str(data_dir), "--split", "val", "--out", str(tmp_path / "v2"), "--n", "20", "--seed", "7"])
    assert (tmp_path / "v2" / "val_mixed.tsv").read_text() == (out / "val_mixed.tsv").read_text()
