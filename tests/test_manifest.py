import json

from PIL import Image

from moocr.data.manifest import build_manifest, load_split, slice_of


def _make_dataset(tmp_path, n=60):
    for i in range(n):
        Image.new("RGB", (32, 16), "white").save(tmp_path / f"w_{i:03d}.png")
        (tmp_path / f"w_{i:03d}.txt").write_text(f"كلمة{i}", encoding="utf-8")


def test_deterministic_and_disjoint(tmp_path):
    _make_dataset(tmp_path)
    m1 = build_manifest(tmp_path, seed=7, golden_size=10, dev_size=20)
    m2 = build_manifest(tmp_path, seed=7, golden_size=10, dev_size=20)
    assert [f["split"] for f in m1["files"]] == [f["split"] for f in m2["files"]]
    counts = {}
    for f in m1["files"]:
        counts[f["split"]] = counts.get(f["split"], 0) + 1
    assert counts == {"golden": 10, "dev": 20, "heldout": 30}


def test_different_seed_different_assignment(tmp_path):
    _make_dataset(tmp_path)
    m1 = build_manifest(tmp_path, seed=1, golden_size=10, dev_size=20)
    m2 = build_manifest(tmp_path, seed=2, golden_size=10, dev_size=20)
    assert [f["split"] for f in m1["files"]] != [f["split"] for f in m2["files"]]


def test_load_split_roundtrip(tmp_path):
    _make_dataset(tmp_path)
    m = build_manifest(tmp_path, seed=7, golden_size=10, dev_size=20)
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(m, ensure_ascii=False), encoding="utf-8")
    assert len(load_split(p, "golden")) == 10


def test_slices():
    assert slice_of("ابجد")["len_bucket"] == "short(<=4)"
    assert slice_of("ابجدهوز")["len_bucket"] == "medium(5-8)"
    assert slice_of("ابجدهوزحطيكلمن")["len_bucket"] == "long(>=9)"
    assert slice_of("سنة ٢٠٢٣")["has_digit"] is True
    assert slice_of("كلمة")["has_digit"] is False
