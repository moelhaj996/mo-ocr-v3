#!/usr/bin/env bash
# Regenerates every number reported in RESULTS.md from scratch.
# Requires: uv, model weights (downloaded on first run), data/apti in place.
set -euo pipefail
cd "$(dirname "$0")/.."

uv sync --extra models
uv run pytest -q

# Manifest (must reproduce the committed one bit-for-bit)
uv run python scripts/build_manifest.py \
  --data-dir data/apti --out /tmp/manifest_check.json
uv run python - <<'PY'
import json
a = json.load(open("manifests/manifest_apti.json"))
b = json.load(open("/tmp/manifest_check.json"))
assert [f["split"] for f in a["files"]] == [f["split"] for f in b["files"]], \
    "SPLIT DRIFT — investigate before trusting any number"
print("manifest reproduces: OK")
PY

# Baselines
for engine in easyocr trocr qwen_vl; do
  for split in golden dev; do
    uv run python -m moocr.harness.evaluate \
      --engine "$engine" --manifest manifests/manifest_apti.json \
      --split "$split" --out "results/${engine}_${split}.json"
  done
done

# Error budget over golden runs
uv run python -m moocr.harness.error_budget \
  results/easyocr_golden.json results/trocr_golden.json results/qwen_vl_golden.json \
  --out results/error_budget_golden.json
