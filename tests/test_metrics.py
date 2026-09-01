import pytest

from moocr.metrics import (
    bidi_check,
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
