"""Offline arbitration simulation from stored per-sample results.

Valid because every routing signal (degeneration flag, engine confidence)
is available in the stored predictions — no ground truth leaks into routing.
Produces a synthetic run file scoreable/comparable like any engine run.
Policy tuning happens on dev ONLY; held-out is touched once at the end.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from moocr.config import Config
from moocr.harness.error_budget import degenerate_flag
from moocr.harness.evaluate import _score_run


def simulate(
    primary_path: Path,
    fallback_path: Path,
    min_primary_conf: float | None,
) -> dict[str, object]:
    a = json.loads(primary_path.read_text(encoding="utf-8"))
    b = json.loads(fallback_path.read_text(encoding="utf-8"))
    sb = {s["id"]: s for s in b["per_sample"]}
    per_sample: list[dict[str, object]] = []
    n_routed = 0
    for s in a["per_sample"]:
        fb = sb.get(s["id"])
        if fb is None:
            continue
        assert fb["truth"] == s["truth"], f"truth mismatch at {s['id']}"
        use_fallback = degenerate_flag(str(s["pred"])) or bool(s["failed"]) or (
            min_primary_conf is not None
            and s["confidence"] is not None
            and float(s["confidence"]) < min_primary_conf  # type: ignore[arg-type,unused-ignore]
        )
        chosen = fb if use_fallback else s
        n_routed += int(use_fallback)
        per_sample.append(
            {
                **{k: chosen[k] for k in ("id", "truth", "pred", "confidence", "latency_ms", "failed")},
                "len_bucket": chosen["len_bucket"],
                "has_digit": chosen["has_digit"],
                "routed_to_fallback": use_fallback,
            }
        )
    name = f"sim_arb({a['meta']['engine']}->{b['meta']['engine']}"
    name += f",conf<{min_primary_conf})" if min_primary_conf is not None else ")"
    run = _score_run(name, a["meta"]["split"], a["meta"]["manifest"], per_sample, [], Config())
    run["n_routed_to_fallback"] = n_routed
    return run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("primary", type=Path)
    parser.add_argument("fallback", type=Path)
    parser.add_argument("--min-primary-conf", type=float, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run = simulate(args.primary, args.fallback, args.min_primary_conf)
    args.out.write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
    s = run["scores"]["normalized"]  # type: ignore[index]
    print(
        f"{run['meta']['engine']}/{run['meta']['split']}: "  # type: ignore[index]
        f"CER(norm)={s['corpus_cer']:.4f} exact={s['exact_match']:.2%} "
        f"routed={run['n_routed_to_fallback']}/{run['n_samples']}"
    )


if __name__ == "__main__":
    main()
