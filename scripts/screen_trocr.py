"""Quick-screen candidate TrOCR checkpoints on a small golden subset."""

import sys
from pathlib import Path

from moocr.config import Config
from moocr.harness.evaluate import run_engine_on_split

for ckpt in sys.argv[1:]:
    cfg = Config()
    cfg.trocr.checkpoint = ckpt
    try:
        r = run_engine_on_split(
            "trocr", Path("manifests/manifest_apti.json"), "golden", cfg, max_samples=10
        )
        s = r["scores"]["normalized"]
        preds = [p["pred"] for p in r["per_sample"][:3]]
        print(f"SCREEN {ckpt}: CER(norm)={s['corpus_cer']:.4f} exact={s['exact_match']:.0%} sample_preds={preds}")
    except Exception as e:
        print(f"SCREEN {ckpt}: FAILED {type(e).__name__}: {str(e)[:120]}")
