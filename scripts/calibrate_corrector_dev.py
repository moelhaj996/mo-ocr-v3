"""Calibrate CamelBERT corrector margin on DEV predictions (offline).

For each margin, applies the corrector to every stored easyocr dev
prediction and reports fix/break counts and the CER delta with paired CI.
The corrector stays disabled in the pipeline unless a margin is
net-positive with CI excluding zero.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from moocr.config import Config  # noqa: E402
from moocr.metrics import fix_break_counts, paired_bootstrap_delta  # noqa: E402
from moocr.models.camelbert import CamelBERTCorrector, confusion_candidates  # noqa: E402
from moocr.normalization import SCORING_V1, normalize  # noqa: E402

run = json.loads(Path("results/easyocr_dev.json").read_text(encoding="utf-8"))
per = run["per_sample"]
refs = [normalize(str(s["truth"]), SCORING_V1) for s in per]
before = [normalize(str(s["pred"]), SCORING_V1) for s in per]

corr = CamelBERTCorrector(Config())

# Cache PLL per unique word across margins (margin only changes the decision)
t0 = time.time()
pll_cache: dict[str, float] = {}


def pll(w: str) -> float:
    if w not in pll_cache:
        pll_cache[w] = corr._pll(w)
    return pll_cache[w]


decisions = []  # (original, best_candidate, pll_orig, pll_best)
for s in per:
    word = str(s["pred"]).strip()
    if not word or " " in word or len(word) > 15:
        decisions.append((word, word, 0.0, 0.0))
        continue
    cands = confusion_candidates(word, Config().camelbert.max_edit_distance)
    if not cands:
        decisions.append((word, word, 0.0, 0.0))
        continue
    p0 = pll(word)
    best_c, best_p = word, p0
    for c in cands:
        pc = pll(c)
        if pc > best_p:
            best_c, best_p = c, pc
    decisions.append((word, best_c, p0, best_p))
print(f"PLL pass done in {time.time()-t0:.0f}s over {len(pll_cache)} unique words", flush=True)

rows = []
for margin in [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]:
    after = [
        normalize(best if best_p > p0 + margin else orig, SCORING_V1)
        for (orig, best, p0, best_p) in decisions
    ]
    fb = fix_break_counts(before, after, refs)
    delta = paired_bootstrap_delta(after, before, refs)
    rows.append(
        {
            "margin": margin,
            **fb,
            "delta_cer_vs_no_correction": round(delta["delta_corpus_cer"], 4),
            "ci_95": [round(x, 4) for x in delta["ci_95"]],
            "p": round(delta["p_two_sided"], 4),
        }
    )
    print(rows[-1], flush=True)

Path("results/corrector_calibration_dev.json").write_text(
    json.dumps(rows, indent=2), encoding="utf-8"
)
