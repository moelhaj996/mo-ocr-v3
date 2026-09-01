"""Tune arbitration policy on the DEV split only (protocol: never on golden/heldout).

Sweeps the qwen-confidence routing threshold and reports each policy's CER
with a paired bootstrap delta against the easyocr dev baseline.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from moocr.harness.simulate_fusion import simulate  # noqa: E402
from moocr.metrics import paired_bootstrap_delta  # noqa: E402
from moocr.normalization import SCORING_V1, normalize  # noqa: E402

PRIMARY = Path("results/qwen_vl_dev.json")
FALLBACK = Path("results/easyocr_dev.json")

base = json.loads(FALLBACK.read_text(encoding="utf-8"))
base_by_id = {s["id"]: s for s in base["per_sample"]}

rows = []
for tau in [None, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
    run = simulate(PRIMARY, FALLBACK, tau)
    per = run["per_sample"]
    ids = [s["id"] for s in per]
    refs = [normalize(str(s["truth"]), SCORING_V1) for s in per]
    hyps = [normalize(str(s["pred"]), SCORING_V1) for s in per]
    fb_hyps = [normalize(str(base_by_id[i]["pred"]), SCORING_V1) for i in ids]
    delta = paired_bootstrap_delta(hyps, fb_hyps, refs)
    s = run["scores"]["normalized"]
    rows.append(
        {
            "tau": tau,
            "cer_norm": round(s["corpus_cer"], 4),
            "exact": round(s["exact_match"], 4),
            "routed": run["n_routed_to_fallback"],
            "delta_vs_easyocr": round(delta["delta_corpus_cer"], 4),
            "ci_95": [round(x, 4) for x in delta["ci_95"]],
            "p": round(delta["p_two_sided"], 4),
        }
    )
    print(rows[-1])

Path("results/policy_sweep_dev.json").write_text(
    json.dumps(rows, indent=2), encoding="utf-8"
)
