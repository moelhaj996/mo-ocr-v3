import pytest

from moocr.metrics import (
    bidi_check,
    token_edit_breakdown,
    cer,
    confusion_report,
    fix_break_counts,
    paired_bootstrap_delta,
    score_corpus,
    wer,
)


def test_cer_known_values():
    assert cer("ابجد", "ابجد") == 0.0
    assert cer("ابجل", "ابجد") == 0.25
    assert cer("", "ابجد") == 1.0
    assert cer("اب", "ابجد") == 0.5


def test_cer_empty_ref_convention():
    assert cer("", "") == 0.0
    assert cer("اب", "") == 2.0  # len(hyp)/max(len(ref),1)


def test_wer():
    assert wer("اب جد", "اب جد") == 0.0
    assert wer("اب خد", "اب جد") == 0.5


def test_corpus_vs_macro():
    # corpus weights by ref length; macro doesn't
    s = score_corpus(["ا", "ابجدهوزح"], ["اب", "ابجدهوزح"])
    assert s.corpus_cer == pytest.approx(1 / 10)
    assert s.macro_cer == pytest.approx((0.5 + 0.0) / 2)
    assert s.exact_match == 0.5


def test_bidi_check_flags_reversed():
    refs = ["ابجد", "هوزح"]
    hyps = ["دجبا", "هوزح"]  # first is stored reversed
    out = bidi_check(hyps, refs)
    assert out["n_reversed_better"] == 1
    assert out["indices"] == [0]


def test_paired_bootstrap_separates_clear_winner():
    refs = ["ابجدهوز"] * 40
    a = ["ابجدهوز"] * 40  # perfect
    b = ["ابجدهول"] * 40  # one error each
    out = paired_bootstrap_delta(a, b, refs, n_resamples=2000)
    assert out["delta_corpus_cer"] < 0
    assert out["ci_95"][1] < 0  # CI excludes zero
    assert out["p_two_sided"] < 0.01


def test_paired_bootstrap_no_difference():
    refs = ["ابجد"] * 30
    a = ["ابجل"] * 30
    out = paired_bootstrap_delta(a, a, refs, n_resamples=500)
    assert out["delta_corpus_cer"] == 0.0
    assert out["p_two_sided"] == 1.0


def test_fix_break():
    refs = ["ابجد", "هوزح", "طيكل"]
    before = ["ابجل", "هوزح", "طيكل"]
    after = ["ابجد", "هوزح", "طيكم"]  # fixed 1, broke 1
    out = fix_break_counts(before, after, refs)
    assert out == {"fixed": 1, "broke": 1, "unchanged": 1}


def test_confusion_report_counts_substitution():
    out = confusion_report(["ابجل"], ["ابجد"])
    assert {"ref": "د", "hyp": "ل", "count": 1} in out


@pytest.mark.parametrize(
    "hyp,ref",
    [("ابجل", "ابجد"), ("", "ابجد"), ("ابجد زائد", "ابجد"), ("ابجد", "")],
)
def test_cer_matches_jiwer_when_available(hyp, ref):
    jiwer = pytest.importorskip("jiwer")
    if not ref:
        return  # jiwer rejects empty refs; our convention documented instead
    assert cer(hyp, ref) == pytest.approx(jiwer.cer(ref, hyp))


def test_corpus_cer_empty_ref_matches_documented_definition():
    # verification finding: empty refs must not inflate the denominator
    s = score_corpus(["x", "ab"], ["", "ab"])
    assert s.corpus_cer == pytest.approx(1 / 2)  # 1 edit / 2 true ref chars


WER_CASES = [
    ("كتاب", "كتاب", 0.0, (0, 0, 0)),  # exact match
    ("كتاب", "هذا كتاب", 1.0, (0, 0, 1)),  # reference word preceded by an extra word
    ("كتاب", "كتاب هذا", 1.0, (0, 0, 1)),  # reference word followed by an extra word
    ("كتاب", "كتب", 1.0, (1, 0, 0)),  # substitution
    ("كتاب", "", 1.0, (0, 1, 0)),  # empty hypothesis
]


@pytest.mark.parametrize("ref,hyp,expected_wer,breakdown", WER_CASES)
def test_wer_single_token_reference(ref, hyp, expected_wer, breakdown):
    assert wer(hyp, ref) == expected_wer
    b = token_edit_breakdown([hyp], [ref])
    assert (b["substitutions"], b["deletions"], b["insertions"]) == breakdown
    assert b["total"] == sum(breakdown)


def test_wer_is_not_the_exact_mismatch_rate():
    # the reference word plus one extra word: an exact mismatch, but WER counts
    # a single insertion, not a mismatch plus an insertion
    assert wer("هذا كتاب", "كتاب") == 1.0
    assert "هذا كتاب" != "كتاب"


@pytest.mark.parametrize("ref,hyp,expected_wer,breakdown", WER_CASES)
def test_wer_matches_jiwer_when_available(ref, hyp, expected_wer, breakdown):
    jiwer = pytest.importorskip("jiwer")
    assert wer(hyp, ref) == pytest.approx(jiwer.wer(ref, hyp))
    out = jiwer.process_words(ref, hyp)
    assert (out.substitutions, out.deletions, out.insertions) == breakdown


def test_token_edit_breakdown_rejects_multi_token_references():
    with pytest.raises(ValueError):
        token_edit_breakdown(["اب جد"], ["اب جد"])
