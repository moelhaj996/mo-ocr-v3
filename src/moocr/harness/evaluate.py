"""Model-agnostic evaluation harness.

Reports, per run (protocol §2, §4, §6):
- CER twice: raw Unicode and SCORING_V1-normalized (gap is a diagnostic)
- WER only when the split contains multi-word references (else suppressed
  as meaningless with a stated reason)
- exact-match raw/normalized
- transparent failure accounting: failed samples are scored CER=1.0 AND
  reported separately; both including/excluding aggregates appear
- bidi check (protocol §3)
- per-slice breakdown (length bucket, digit-bearing)
- character confusion top-30
- per-sample latency; run metadata (engine, checkpoint, norm version, seed,
  git commit) for reproducibility
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

from moocr import __version__
from moocr.config import Config
from moocr.data.manifest import load_split, slice_of
from moocr.logging_util import get_logger, set_doc_id
from moocr.metrics import (
    bidi_check,
    cer,
    confusion_report,
    score_corpus,
    wer,
)
from moocr.models.base import get_engine
from moocr.normalization import NORM_VERSION, SCORING_V1, normalize

log = get_logger(__name__)


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=Path(__file__).parent,
        ).stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def run_engine_on_split(  # noqa: C901
    engine_name: str,
    manifest_path: Path,
    split: str,
    config: Config,
    max_samples: int | None = None,
) -> dict[str, object]:
    rows = load_split(manifest_path, split)
    if max_samples:
        rows = rows[:max_samples]
    engine = get_engine(engine_name, config)

    per_sample: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    from PIL import Image

    for row in rows:
        set_doc_id(str(row["id"]))
        t0 = time.perf_counter()
        try:
            with Image.open(str(row["image"])) as img:
                converted = img.convert("RGB")
                converted._moocr_id = row["id"]  # type: ignore[attr-defined]  # correlation for tests/tracing
                rec = engine.recognize(converted)
            latency = (time.perf_counter() - t0) * 1000
            per_sample.append(
                {
                    "id": row["id"],
                    "truth": row["truth"],
                    "pred": rec.text,
                    "confidence": rec.confidence,
                    "latency_ms": round(latency, 1),
                    "failed": False,
                    **slice_of(str(row["truth"])),
                }
            )
        except Exception as exc:  # accounted, never silently dropped
            latency = (time.perf_counter() - t0) * 1000
            log.warning("sample failed: %s", exc)
            failures.append({"id": row["id"], "error": repr(exc)})
            per_sample.append(
                {
                    "id": row["id"],
                    "truth": row["truth"],
                    "pred": "",
                    "confidence": None,
                    "latency_ms": round(latency, 1),
                    "failed": True,
                    **slice_of(str(row["truth"])),
                }
            )
    set_doc_id("-")
    return _score_run(engine_name, split, str(manifest_path), per_sample, failures, config)


def _score_run(
    engine_name: str,
    split: str,
    manifest_path: str,
    per_sample: list[dict[str, object]],
    failures: list[dict[str, object]],
    config: Config,
) -> dict[str, object]:
    refs_raw = [str(s["truth"]) for s in per_sample]
    hyps_raw = [str(s["pred"]) for s in per_sample]
    refs_norm = [normalize(r, SCORING_V1) for r in refs_raw]
    hyps_norm = [normalize(h, SCORING_V1) for h in hyps_raw]

    ok = [s for s in per_sample if not s["failed"]]
    scores: dict[str, object] = {
        "raw": vars(score_corpus(hyps_raw, refs_raw)),
        "normalized": vars(score_corpus(hyps_norm, refs_norm)),
    }
    if failures and ok:
        scores["excluding_failures"] = {
            "raw": vars(score_corpus([str(s["pred"]) for s in ok], [str(s["truth"]) for s in ok])),
            "normalized": vars(
                score_corpus(
                    [normalize(str(s["pred"]), SCORING_V1) for s in ok],
                    [normalize(str(s["truth"]), SCORING_V1) for s in ok],
                )
            ),
        }

    multiword = sum(len(r.split()) > 1 for r in refs_raw)
    if multiword > 0:
        scores["wer_raw"] = sum(wer(h, r) for h, r in zip(hyps_raw, refs_raw)) / len(refs_raw)
        scores["wer_normalized"] = sum(
            wer(h, r) for h, r in zip(hyps_norm, refs_norm)
        ) / len(refs_norm)
    else:
        scores["wer"] = "suppressed: all references are single words; WER degenerates to exact-match"

    slices: dict[str, object] = {}
    for key in ("len_bucket", "has_digit"):
        for value in sorted({str(s[key]) for s in per_sample}):
            group = [s for s in per_sample if str(s[key]) == value]
            gh = [normalize(str(s["pred"]), SCORING_V1) for s in group]
            gr = [normalize(str(s["truth"]), SCORING_V1) for s in group]
            slices[f"{key}={value}"] = {"n": len(group), **vars(score_corpus(gh, gr))}

    lat = sorted(float(s["latency_ms"]) for s in per_sample)  # type: ignore[arg-type]
    return {
        "meta": {
            "engine": engine_name,
            "engine_config": _engine_cfg(engine_name, config),
            "split": split,
            "manifest": manifest_path,
            "norm_version": NORM_VERSION,
            "scoring_profile": SCORING_V1.name,
            "seed": config.seed,
            "moocr_version": __version__,
            "git_commit": _git_commit(),
            "platform": platform.platform(),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "n_samples": len(per_sample),
        "n_failures": len(failures),
        "failures": failures,
        "scores": scores,
        "bidi_check": bidi_check(hyps_norm, refs_norm),
        "slices": slices,
        "confusions_top30": confusion_report(hyps_norm, refs_norm, 30),
        "latency_ms": {
            "median": lat[len(lat) // 2],
            "p95": lat[int(len(lat) * 0.95)] if len(lat) > 1 else lat[0],
            "mean": round(sum(lat) / len(lat), 1),
        },
        "per_sample": per_sample,
    }


def _engine_cfg(engine_name: str, config: Config) -> dict[str, object]:
    return {
        "easyocr": lambda: config.easyocr.model_dump(),
        "trocr": lambda: config.trocr.model_dump(),
        "qwen_vl": lambda: config.qwen_vl.model_dump(),
    }.get(engine_name, dict)()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", required=True, choices=["golden", "dev", "heldout"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    config = Config.load(args.config)
    result = run_engine_on_split(
        args.engine, args.manifest, args.split, config, args.max_samples
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    s: dict = result["scores"]  # type: ignore[assignment,type-arg]  # CLI display only
    print(
        f"{args.engine}/{args.split}: n={result['n_samples']} "
        f"failures={result['n_failures']} "
        f"CER raw={s['raw']['corpus_cer']:.4f} "
        f"norm={s['normalized']['corpus_cer']:.4f} "
        f"exact(norm)={s['normalized']['exact_match']:.2%}"
    )


if __name__ == "__main__":
    main()
