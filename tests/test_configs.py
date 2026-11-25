import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_all_configs_have_required_keys():
    files = sorted((ROOT / "configs").glob("*.yaml"))
    assert {f.stem for f in files} >= {"debug", "bilstm", "transformer"}
    for f in files:
        cfg = yaml.safe_load(f.read_text())
        assert cfg["model"]["type"] in {"bilstm", "transformer"}, f
        assert {"dir", "max_len"} <= set(cfg["data"]), f
        assert {"batch_size", "lr", "max_steps", "eval_every", "ckpt_every"} <= set(cfg["train"]), f
        assert "seed" in cfg, f
        if cfg["model"]["type"] == "transformer":
            assert cfg["data"]["max_len"] <= cfg["model"]["max_len"], f


def test_report_table(tmp_path):
    from xettat.report_table import main

    runs = tmp_path / "runs"
    for name, acc in [("baselines", 0.8), ("bilstm_v1", 0.95)]:
        (runs / name).mkdir(parents=True)
    (runs / "baselines" / "wordfreq_test.json").write_text(json.dumps(
        {"model": "wordfreq", "word_acc": 0.8, "char_acc": 0.9, "sentence_acc": 0.3, "wer": 0.1,
         "ambig_word_acc": 0.5, "corruption_rate": 0.0, "latency_ms_per_sentence_cpu": 0.1}))
    (runs / "bilstm_v1" / "final_test.json").write_text(json.dumps(
        {"word_acc": 0.95, "char_acc": 0.98, "sentence_acc": 0.7, "wer": 0.03, "ambig_word_acc": 0.8,
         "corruption_rate": 0.001, "latency_ms_per_sentence_cpu": 2.0, "params": 5_000_000, "ckpt_mb": 20.0}))
    out = tmp_path / "results.md"
    main(["--runs", str(runs), "--split", "test", "--out", str(out)])
    md = out.read_text()
    assert "wordfreq" in md and "bilstm_v1" in md and "0.9500" in md
    assert md.index("wordfreq") < md.index("bilstm_v1")  # sorted by word_acc ascending
