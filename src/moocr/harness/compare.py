"""Paired comparison of two evaluation runs.

Joins per-sample results by id, then reports:
- paired bootstrap CI + p-value on the corpus-CER delta (raw and normalized)
- fix/break/unchanged counts (protocol §4 — both directions, always)
No unpaired means, no one-arm CIs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from moocr.metrics import fix_break_counts, paired_bootstrap_delta
from moocr.normalization import SCORING_V1, normalize


def compare_runs(path_a: Path, path_b: Path) -> dict:
    a = json.loads(path_a.read_text(encoding="utf-8"))
    b = json.loads(path_b.read_text(encoding="utf-8"))
    sa = {s["id"]: s for s in a["per_sample"]}
    sb = {s["id"]: s for s in b["per_sample"]}
    common = sorted(sa.keys() & sb.keys())
    dropped = len(sa.keys() | sb.keys()) - len(common)

    refs = [sa[i]["truth"] for i in common]
    ha = [sa[i]["pred"] for i in common]
    hb = [sb[i]["pred"] for i in common]
    refs_n = [normalize(r, SCORING_V1) for r in refs]
    ha_n = [normalize(h, SCORING_V1) for h in ha]
    hb_n = [normalize(h, SCORING_V1) for h in hb]

    return {
        "a": {"engine": a["meta"]["engine"], "file": str(path_a)},
        "b": {"engine": b["meta"]["engine"], "file": str(path_b)},
        "n_paired": len(common),
        "n_unpaired_dropped": dropped,
        "delta_raw": paired_bootstrap_delta(ha, hb, refs),
        "delta_normalized": paired_bootstrap_delta(ha_n, hb_n, refs_n),
        "fix_break_a_to_b": fix_break_counts(ha_n, hb_n, refs_n),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_a", type=Path)
    parser.add_argument("run_b", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    result = compare_runs(args.run_a, args.run_b)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    d = result["delta_normalized"]
    print(
        f"Δ corpus-CER (A-B, normalized) = {d['delta_corpus_cer']:+.4f} "
        f"CI95 [{d['ci_95'][0]:+.4f}, {d['ci_95'][1]:+.4f}] p={d['p_two_sided']:.4f}"
    )
    fb = result["fix_break_a_to_b"]
    print(f"A→B: fixed={fb['fixed']} broke={fb['broke']} unchanged={fb['unchanged']}")


if __name__ == "__main__":
    main()
