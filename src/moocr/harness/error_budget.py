"""Error budget across paired engine runs (protocol §4: ablation, not estimates).

Given N evaluation-result JSONs over the SAME split, attributes error and
quantifies the headroom of fusion:

1. orthographic vs recognition error: raw-vs-normalized CER gap per engine
2. cross-engine agreement: on samples where engines disagree, who is right
3. confidence validity: does each engine's confidence actually separate
   correct from incorrect output? (point-biserial correlation + means)
4. oracle arbitration ceiling: corpus CER if a perfect router picked the
   best engine per sample — the upper bound any fusion can reach
5. degeneration detectability (VLM engines): how many catastrophic outputs
   (sample CER > 1) are flaggable without ground truth via length ratio
   and non-Arabic-script content
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np

from moocr.metrics import cer, levenshtein
from moocr.normalization import SCORING_V1, normalize


def _load(path: Path) -> tuple[str, dict[str, dict[str, object]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["meta"]["engine"], {s["id"]: s for s in data["per_sample"]}


def degenerate_flag(pred: str, ref_len_hint: int | None = None) -> bool:
    """Ground-truth-free catastrophe detector for VLM output.

    Flags: output much longer than plausible for the crop, or containing
    Latin/CJK runs. ``ref_len_hint`` (e.g. the fallback engine's text
    length for the same crop) scales the length test for line-level crops;
    without it the word-crop threshold applies.
    """
    # Thresholds set on the dev split (0 false alarms on correct output,
    # 49/49 catastrophes caught); len>30 and >=2 non-Arabic letters also
    # close the refusal-leak class observed in the golden regression set.
    limit = 30 if ref_len_hint is None else max(30, int(2.5 * ref_len_hint))
    if len(pred) > limit:
        return True
    non_arabic = sum(
        1 for c in pred if c.isalpha() and not ("؀" <= c <= "ۿ")
    )
    return non_arabic >= 2


def build_budget(paths: list[Path]) -> dict[str, object]:
    engines: dict[str, dict[str, dict[str, object]]] = {}
    for p in paths:
        name, samples = _load(p)
        engines[name] = samples

    ids = sorted(set.intersection(*(set(s) for s in engines.values())))
    if not ids:
        raise ValueError("No common sample ids across runs")

    def norm_pred(e: str, i: str) -> str:
        return normalize(str(engines[e][i]["pred"]), SCORING_V1)

    def norm_ref(i: str) -> str:
        any_e = next(iter(engines))
        return normalize(str(engines[any_e][i]["truth"]), SCORING_V1)

    refs = {i: norm_ref(i) for i in ids}

    per_engine: dict[str, object] = {}
    sample_cer: dict[str, dict[str, float]] = {e: {} for e in engines}
    for e, samples in engines.items():
        raw_edits = norm_edits = raw_len = norm_len = 0
        confs, corrects = [], []
        for i in ids:
            s = samples[i]
            raw_edits += levenshtein(str(s["truth"]), str(s["pred"]))
            raw_len += max(len(str(s["truth"])), 1)
            c = cer(norm_pred(e, i), refs[i])
            sample_cer[e][i] = c
            norm_edits += levenshtein(refs[i], norm_pred(e, i))
            norm_len += max(len(refs[i]), 1)
            if s.get("confidence") is not None:
                confs.append(float(s["confidence"]))  # type: ignore[arg-type]
                corrects.append(1.0 if c == 0 else 0.0)
        cer_raw, cer_norm = raw_edits / raw_len, norm_edits / norm_len
        conf_stats: dict[str, object] = {"n_with_confidence": len(confs)}
        if len(confs) > 2 and 0 < sum(corrects) < len(corrects):
            ca = np.array(confs)
            co = np.array(corrects)
            conf_stats.update(
                {
                    "point_biserial_r": float(np.corrcoef(ca, co)[0, 1]),
                    "mean_conf_correct": float(ca[co == 1].mean()),
                    "mean_conf_incorrect": float(ca[co == 0].mean()),
                }
            )
        per_engine[e] = {
            "cer_raw": round(cer_raw, 4),
            "cer_normalized": round(cer_norm, 4),
            "orthographic_share_of_error": round((cer_raw - cer_norm) / cer_raw, 4)
            if cer_raw
            else 0.0,
            "exact_match": round(
                sum(1 for i in ids if sample_cer[e][i] == 0) / len(ids), 4
            ),
            "confidence_validity": conf_stats,
        }

    pairwise: dict[str, object] = {}
    for a, b in combinations(engines, 2):
        agree = [i for i in ids if norm_pred(a, i) == norm_pred(b, i)]
        disagree = [i for i in ids if i not in set(agree)]
        a_wins = sum(1 for i in disagree if sample_cer[a][i] < sample_cer[b][i])
        b_wins = sum(1 for i in disagree if sample_cer[b][i] < sample_cer[a][i])
        agree_correct = sum(1 for i in agree if sample_cer[a][i] == 0)
        pairwise[f"{a}|{b}"] = {
            "agree": len(agree),
            "agree_and_correct": agree_correct,
            "disagree": len(disagree),
            f"{a}_wins": a_wins,
            f"{b}_wins": b_wins,
            "tie": len(disagree) - a_wins - b_wins,
        }

    oracle_edits = sum(
        min(
            levenshtein(refs[i], norm_pred(e, i)) for e in engines
        )
        for i in ids
    )
    total_ref = sum(max(len(refs[i]), 1) for i in ids)

    degen: dict[str, object] = {}
    for e, samples in engines.items():
        catastrophic = [i for i in ids if sample_cer[e][i] > 1.0]
        flagged = [i for i in ids if degenerate_flag(str(samples[i]["pred"]))]
        caught = len(set(catastrophic) & set(flagged))
        false_alarm = [
            i for i in flagged if sample_cer[e][i] == 0
        ]
        degen[e] = {
            "catastrophic(cer>1)": len(catastrophic),
            "flagged_by_heuristic": len(flagged),
            "caught": caught,
            "missed": len(catastrophic) - caught,
            "false_alarms_on_correct": len(false_alarm),
        }

    return {
        "n_common_samples": len(ids),
        "per_engine": per_engine,
        "pairwise_agreement": pairwise,
        "oracle_arbitration_cer_normalized": round(oracle_edits / total_ref, 4),
        "degeneration_detectability": degen,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    budget = build_budget(args.runs)
    text = json.dumps(budget, ensure_ascii=False, indent=2)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
