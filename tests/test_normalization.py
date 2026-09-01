from moocr.normalization import OUTPUT_V1, SCORING_V1, normalize


def test_idempotent():
    s = "ﻣﺪﺭﺳﺔ الأَطْفالِ ـــ ٢٠٢٣ ى"
    once = normalize(s, SCORING_V1)
    assert normalize(once, SCORING_V1) == once


def test_presentation_forms_folded():
    # U+FEE3 (MEEM medial) etc. should fold to plain letters under NFKC
    assert normalize("ﻣﺪﺭﺳﺔ", SCORING_V1) == normalize("مدرسة", SCORING_V1)


def test_scoring_folds_orthography():
    assert normalize("أحمد", SCORING_V1) == normalize("احمد", SCORING_V1)
    assert normalize("مدرسة", SCORING_V1) == normalize("مدرسه", SCORING_V1)
    assert normalize("مستشفى", SCORING_V1) == normalize("مستشفي", SCORING_V1)
    assert normalize("٢٠٢٣", SCORING_V1) == "2023"
    assert normalize("۲۰۲۳", SCORING_V1) == "2023"


def test_scoring_strips_diacritics_and_tatweel():
    assert normalize("كَـتَبَ", SCORING_V1) == "كتب"


def test_invisibles_stripped():
    assert normalize("اب‌جد‏", SCORING_V1) == "ابجد"


def test_output_preserves_meaningful_orthography():
    assert normalize("أحمد", OUTPUT_V1) == "أحمد"  # hamza kept
    assert normalize("كَتَبَ", OUTPUT_V1) == "كَتَبَ"  # diacritics kept
    assert normalize("مدرسة", OUTPUT_V1) == "مدرسة"  # taa marbuta kept
    assert normalize("٢٠٢٣", OUTPUT_V1) == "٢٠٢٣"  # digits kept
    assert normalize("كـتب", OUTPUT_V1) == "كتب"  # tatweel still stripped


def test_whitespace_collapse():
    assert normalize("  اب   جد \n", SCORING_V1) == "اب جد"
