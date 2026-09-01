import json

from PIL import Image

from moocr.config import Config
from moocr.data.manifest import build_manifest
from moocr.harness.compare import compare_runs
from moocr.harness.evaluate import run_engine_on_split
from moocr.models.base import ENGINES
from moocr.models.fake import FakeRecognizer


def _dataset(tmp_path, truths):
    for key, truth in truths.items():
        Image.new("RGB", (32, 16), "white").save(tmp_path / f"{key}.png")
        (tmp_path / f"{key}.txt").write_text(truth, encoding="utf-8")
    m = build_manifest(tmp_path, seed=7, golden_size=2, dev_size=2)
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(m, ensure_ascii=False), encoding="utf-8")
    return p


def test_end_to_end_with_failure_accounting(tmp_path):
    truths = {f"w_{i:02d}": t for i, t in enumerate(
        ["مدرسة", "كتاب", "قلم", "بيت", "شمس", "قمر"]
    )}
    manifest = _dataset(tmp_path, truths)
    preds = dict(truths)
    preds["w_02"] = "فلم"  # one substitution error
    ENGINES["fake"] = lambda cfg: FakeRecognizer(preds, fail_ids={"w_04"})
    try:
        result = run_engine_on_split("fake", manifest, "heldout", Config())
    finally:
        del ENGINES["fake"]

    assert result["n_samples"] == 2
    per = {s["id"]: s for s in result["per_sample"]}
    heldout_ids = set(per)
    # failure accounting only applies if the failing id landed in this split
    if "w_04" in heldout_ids:
        assert result["n_failures"] == 1
        assert per["w_04"]["failed"] is True
        assert "excluding_failures" in result["scores"]
    assert result["meta"]["norm_version"] == "1.0.0"
    assert "suppressed" in str(result["scores"]["wer"])  # single-word refs
    assert result["bidi_check"]["n_reversed_better"] == 0


def test_compare_runs_paired(tmp_path):
    truths = {f"w_{i:02d}": t for i, t in enumerate(
        ["مدرسة", "كتاب", "قلم", "بيت", "شمس", "قمر"]
    )}
    manifest = _dataset(tmp_path, truths)

    perfect = dict(truths)
    flawed = dict(truths)
    for k in list(flawed)[:3]:
        flawed[k] = flawed[k][:-1] + "ل"

    for name, preds in [("run_a", perfect), ("run_b", flawed)]:
        ENGINES["fake"] = lambda cfg, p=preds: FakeRecognizer(p)
        try:
            r = run_engine_on_split("fake", manifest, "heldout", Config())
        finally:
            del ENGINES["fake"]
        (tmp_path / f"{name}.json").write_text(
            json.dumps(r, ensure_ascii=False), encoding="utf-8"
        )

    out = compare_runs(tmp_path / "run_a.json", tmp_path / "run_b.json")
    assert out["n_paired"] == 2
    assert out["delta_normalized"]["delta_corpus_cer"] <= 0  # A no worse
    fb = out["fix_break_a_to_b"]
    assert fb["fixed"] + fb["broke"] + fb["unchanged"] == 2
