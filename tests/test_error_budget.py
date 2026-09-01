import json

from moocr.harness.error_budget import build_budget, degenerate_flag


def _run_file(tmp_path, name, engine, preds, truths):
    per = [
        {"id": k, "truth": truths[k], "pred": preds[k], "confidence": 0.9,
         "latency_ms": 1.0, "failed": False}
        for k in truths
    ]
    p = tmp_path / f"{name}.json"
    p.write_text(json.dumps({"meta": {"engine": engine}, "per_sample": per},
                            ensure_ascii=False), encoding="utf-8")
    return p


def test_oracle_is_best_per_sample(tmp_path):
    truths = {"a": "مدرسة", "b": "كتاب"}
    pa = _run_file(tmp_path, "ra", "e1", {"a": "مدرسة", "b": "كتلب"}, truths)
    pb = _run_file(tmp_path, "rb", "e2", {"a": "مدرسه خطأ", "b": "كتاب"}, truths)
    budget = build_budget([pa, pb])
    # oracle picks e1 on a (0 edits) and e2 on b (0 edits)
    assert budget["oracle_arbitration_cer_normalized"] == 0.0
    pw = budget["pairwise_agreement"]["e1|e2"]
    assert pw["disagree"] == 2 and pw["e1_wins"] == 1 and pw["e2_wins"] == 1


def test_degenerate_flag():
    assert degenerate_flag("The text in the image is Arabic ...")
    assert degenerate_flag("النص العربي الموجود في الصورة حرفيا كما هو بدون اي شرح او تعليق")
    assert not degenerate_flag("مدرسة")
    assert not degenerate_flag("")
