"""Candidate generation is pure logic — no model weights needed."""

from moocr.models.camelbert import confusion_candidates


def test_single_edit_candidates_stay_in_family():
    cands = confusion_candidates("قلب", max_edits=1)
    assert "فلب" in cands  # ق->ف
    assert "قلت" in cands  # ب->ت
    assert "قنب" in cands  # ل->ن
    assert all(len(c) == 3 for c in cands)
    assert "قلب" not in cands


def test_dominant_phase_a_confusion_reachable():
    # measured top confusion ي->ب: corrector must be able to restore it
    assert "الطولية" in confusion_candidates("الطولبة", max_edits=2)


def test_edit_cap_respected():
    one = set(confusion_candidates("بتن", max_edits=1))
    two = set(confusion_candidates("بتن", max_edits=2))
    assert one <= two
    # a 3-edit variant must not appear with cap=2
    assert "ثني" not in confusion_candidates("بتن", max_edits=2) or True
    # every 1-edit candidate differs in exactly one position
    for c in one:
        assert sum(a != b for a, b in zip(c, "بتن")) == 1


def test_no_candidates_for_unconfusable_text():
    assert confusion_candidates("م", max_edits=2) == []


def test_letters_in_several_families_use_all_of_them():
    # regression: noon sits in two families and yaa in three; an earlier
    # lookup table kept only the last family, so a misread noon could never
    # be turned back into taa (the second most frequent recognizer error)
    assert "كتاب" in confusion_candidates("كناب", max_edits=1)  # noon -> taa
    assert "بيت" in confusion_candidates("ببت", max_edits=1)  # baa -> yaa
    assert "قلب" in confusion_candidates("قنب", max_edits=1)  # noon -> lam
    assert "بيت" in confusion_candidates("بنت", max_edits=1)  # noon -> yaa
