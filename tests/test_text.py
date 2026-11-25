from xettat.text import apply_labels, az_lower, az_upper, make_labels, strip

SENTENCES = [
    "üzümün suyu üzümə sıçradı",
    "Şəhər çox gözəl idi.",
    "İLHAM ƏLİYEV BAKIDA",
    "Mən 2026-cı ildə https://x.az saytına baxdım 😀",
    "Bu gün weather çox soyuqdur, deyəsən qış gəlir.",
    "Ağac, ölkə, üzük, çiçək, şüşə, ığdır, Qızıl",
    "AZƏRBAYCAN RESPUBLİKASI",
    "  boşluqlar   və\tTAB\n yeni sətir ",
    "Привет, salam!",
    "İstanbul və İsmayıllı",
]


def test_strip_basic():
    assert strip("üzümün suyu üzümə sıçradı") == "uzumun suyu uzume sicradi"


def test_strip_uppercase():
    assert strip("İLHAM ƏLİYEV") == "ILHAM ELIYEV"


def test_az_case():
    assert az_lower("IŞIQ İL") == "ışıq il"
    assert az_upper("ışıq il") == "IŞIQ İL"


def test_make_labels_word():
    src, y = make_labels("Şəhər")
    assert src == "Seher"
    assert y == [1, 1, -100, 1, -100]


def test_roundtrip():
    for t in SENTENCES:
        src, y = make_labels(t)
        assert len(src) == len(t) == len(y)
        assert apply_labels(src, y) == t


def test_empty():
    assert make_labels("") == ("", [])
    assert apply_labels("", []) == ""


def test_nfc_decomposed_input():
    # "ü" written as u + combining diaeresis must be normalised first
    decomposed = "üzüm"
    src, y = make_labels(decomposed)
    assert src == "uzum"
    assert apply_labels(src, y) == "üzüm"


def test_make_labels_keep_label():
    src, y = make_labels("Sabah")
    assert src == "Sabah"
    assert y == [0, -100, -100, -100, -100]
