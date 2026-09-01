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
for engine in easyocr qwen_vl; do
  for split in golden dev heldout; do
    uv run python -m moocr.harness.evaluate \
      --engine "$engine" --manifest manifests/manifest_apti.json \
      --split "$split" --out "results/${engine}_${split}.json"
  done
done

# Frozen arbitration policy (tau=0.40, dev-tuned) simulated per split
for split in golden dev heldout; do
  uv run python -m moocr.harness.simulate_fusion \
    "results/qwen_vl_${split}.json" "results/easyocr_${split}.json" \
    --min-primary-conf 0.40 --out "results/sim_arb_tau40_${split}.json"
done

# Error budgets
uv run python -m moocr.harness.error_budget \
  results/easyocr_golden.json results/qwen_vl_golden.json \
  --out results/error_budget_golden_2eng.json > /dev/null
uv run python -m moocr.harness.error_budget \
  results/easyocr_heldout.json results/qwen_vl_heldout.json \
  --out results/error_budget_heldout.json > /dev/null

# Dev-only tuning artifacts (policy sweep, corrector calibration)
uv run python scripts/sweep_policy_dev.py
uv run python scripts/calibrate_corrector_dev.py

# Compose RESULTS.md from the artifacts
uv run python scripts/make_results.py
