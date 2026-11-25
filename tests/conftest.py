import itertools
import random

import pytest

SUBJ = ["Mən", "Sən", "Biz", "Qardaşım", "Müəllim", "Uşaqlar", "Şəhər əhalisi", "Gənc tələbə"]
TIME = ["səhər", "axşam", "bu gün", "dünən", "sabah", "gecə"]
OBJ = ["çörək", "üzüm", "kitab", "şəkil", "gül", "çay", "göz", "qəzet"]
VERB = ["aldı", "gördü", "sevdi", "gətirdi", "istədi", "seçdi"]
PLACE = ["bazarda", "şəhərdə", "evdə", "məktəbdə", "küçədə", "bağçada"]


@pytest.fixture
def az_sentences() -> list[str]:
    rng = random.Random(0)
    combos = list(itertools.product(SUBJ, TIME, OBJ, VERB, PLACE))
    rng.shuffle(combos)
    return [f"{s} {t} {p} {o} {v}." for s, t, o, v, p in combos[:240]]


@pytest.fixture
def data_dir(tmp_path, az_sentences):
    d = tmp_path / "data"
    d.mkdir()
    (d / "train.txt").write_text("\n".join(az_sentences[:200]) + "\n", encoding="utf-8")
    (d / "val.txt").write_text("\n".join(az_sentences[200:220]) + "\n", encoding="utf-8")
    (d / "test.txt").write_text("\n".join(az_sentences[220:]) + "\n", encoding="utf-8")
    return d
