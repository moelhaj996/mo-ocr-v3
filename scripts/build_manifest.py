"""Build and commit the dataset manifest with deterministic splits."""

import argparse
import json
from pathlib import Path

from moocr.config import Config
from moocr.data.manifest import build_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    cfg = Config.load(args.config).split
    manifest = build_manifest(
        args.data_dir, cfg.seed, cfg.golden_size, cfg.dev_size
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    counts: dict = {}
    for f in manifest["files"]:
        counts[f["split"]] = counts.get(f["split"], 0) + 1
    print(f"{manifest['n_files']} files -> {counts}")


if __name__ == "__main__":
    main()
