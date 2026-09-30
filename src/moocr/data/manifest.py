"""Dataset manifest: content-hashed files with deterministic split assignment.

The manifest is committed to the repo. The held-out split is never trained on
and never tuned against; the golden split is the regression set every change
runs against. Split membership is a pure function of (sorted file list, seed).
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

MANIFEST_VERSION = "1.0.0"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(
    data_dir: Path, seed: int, golden_size: int, dev_size: int
) -> dict[str, object]:
    pairs = []
    unpaired: list[str] = []
    images = sorted(set(data_dir.glob("*.png")) | set(data_dir.glob("*.PNG")))
    for png in images:
        txt = png.with_suffix(".txt")
        if txt.exists():
            pairs.append((png, txt))
        else:
            unpaired.append(png.name)
    if len(pairs) < golden_size + dev_size + 1:
        raise ValueError(
            f"Only {len(pairs)} pairs in {data_dir}; "
            f"need > golden({golden_size}) + dev({dev_size})."
        )

    order = list(range(len(pairs)))
    random.Random(seed).shuffle(order)
    split_of: dict[int, str] = {}
    for rank, i in enumerate(order):
        if rank < golden_size:
            split_of[i] = "golden"
        elif rank < golden_size + dev_size:
            split_of[i] = "dev"
        else:
            split_of[i] = "heldout"

    files = []
    for i, (png, txt) in enumerate(pairs):
        truth = txt.read_text(encoding="utf-8").strip()
        files.append(
            {
                "id": png.stem,
                "image": str(png),
                "sha256_image": _sha256(png),
                "sha256_text": _sha256(txt),
                "truth": truth,
                "split": split_of[i],
            }
        )
    return {
        "manifest_version": MANIFEST_VERSION,
        "n_unpaired_images_skipped": len(unpaired),
        "unpaired_images": unpaired[:50],
        "seed": seed,
        "golden_size": golden_size,
        "dev_size": dev_size,
        "n_files": len(files),
        "files": files,
    }


def load_split(manifest_path: Path, split: str) -> list[dict[str, object]]:
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    rows = [f for f in manifest["files"] if f["split"] == split]
    if not rows:
        raise ValueError(f"No files in split {split!r}")
    return rows


def slice_of(truth: str) -> dict[str, object]:
    """Slice metadata per protocol §4 (per-slice breakdowns)."""
    n = len(truth.replace(" ", ""))
    bucket = "short(<=4)" if n <= 4 else "medium(5-8)" if n <= 8 else "long(>=9)"
    return {
        "len_bucket": bucket,
        "has_digit": any(c.isdigit() or "٠" <= c <= "٩" or "۰" <= c <= "۹" for c in truth),
    }
