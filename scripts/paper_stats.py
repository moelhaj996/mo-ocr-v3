"""Compute every number cited in paper/paper.tex that is not already in a
run file. Reads only the committed result files; writes results/paper_stats.json.

    PYTHONPATH=src python scripts/paper_stats.py
"""

import json
import re
import statistics
from pathlib import Path

import numpy as np

from moocr.harness.error_budget import degenerate_flag
from moocr.metrics import cer, levenshtein, paired_bootstrap_delta
from moocr.normalization import SCORING_V1, normalize

R = Path("results")
N = lambda t: normalize(str(t), SCORING_V1)  # noqa: E731
TAU = 0.40


def load(name):
    return json.loads((R / f"{name}.json").read_text(encoding="utf-8"))


def flag_original(pred: str) -> bool:
    """The first version of the flag: 40 characters / 3 non-Arabic letters."""
    if len(pred) > 40:
        return True
    return sum(1 for c in pred if c.isalpha() and not ("؀" <= c <= "ۿ")) >= 3


def corpus(hyps, refs):
    return sum(levenshtein(r, h) for h, r in zip(hyps, refs)) / max(sum(len(r) for r in refs), 1)


def route(qw, eo, use_primary):
    """Replay a routing rule; use_primary(sample, cer) -> keep the Qwen text."""
    eo_by = {s["id"]: s for s in eo["per_sample"]}
    hyps, kept = [], 0
    for s in qw["per_sample"]:
        keep = use_primary(s)
        kept += keep
        hyps.append(N((s if keep else eo_by[s["id"]])["pred"]))
    return hyps, kept


def report(hyps, base, refs):
    d = paired_bootstrap_delta(hyps, base, refs)
    return {
        "cer": round(corpus(hyps, refs), 4),
        "exact": round(sum(h == r for h, r in zip(hyps, refs)) / len(refs), 4),
        "delta_vs_easyocr": round(d["delta_corpus_cer"], 4),
        "ci_95": [round(x, 4) for x in d["ci_95"]],
    }


out = {}
qw, eo = load("qwen_vl_heldout"), load("easyocr_heldout")
eo_by = {s["id"]: s for s in eo["per_sample"]}
refs = [N(s["truth"]) for s in qw["per_sample"]]
base = [N(eo_by[s["id"]]["pred"]) for s in qw["per_sample"]]
qcer = {s["id"]: cer(N(s["pred"]), N(s["truth"])) for s in qw["per_sample"]}
conf = lambda s: s["confidence"] is not None and s["confidence"] < TAU  # noqa: E731

variants = {
    "flag_and_confidence": lambda s: not (degenerate_flag(str(s["pred"])) or conf(s)),
    "flag_only": lambda s: not degenerate_flag(str(s["pred"])),
    "confidence_only": lambda s: not conf(s),
    "original_flag_and_confidence": lambda s: not (flag_original(str(s["pred"])) or conf(s)),
    "oracle_degenerate_detector": lambda s: qcer[s["id"]] <= 1,
}
abl = {}
for name, rule in variants.items():
    hyps, kept = route(qw, eo, rule)
    abl[name] = {**report(hyps, base, refs), "kept_qwen": kept, "routed_to_easyocr": len(refs) - kept}
oracle = [min((N(s["pred"]), b), key=lambda h: levenshtein(r, h)) for s, b, r in zip(qw["per_sample"], base, refs)]
abl["oracle_router"] = report(oracle, base, refs)
for tau in (0.35, 0.45):
    hyps, kept = route(qw, eo, lambda s, t=tau: not (degenerate_flag(str(s["pred"])) or (s["confidence"] is not None and s["confidence"] < t)))
    abl[f"posthoc_tau_{tau}"] = {**report(hyps, base, refs), "kept_qwen": kept}
out["heldout_ablation"] = abl

final, _ = route(qw, eo, variants["flag_and_confidence"])
out["heldout_routing"] = {
    "correct_qwen_outputs_discarded": sum(
        1 for s in qw["per_sample"] if qcer[s["id"]] == 0 and not variants["flag_and_confidence"](s)
    ),
    "flagged": sum(degenerate_flag(str(s["pred"])) for s in qw["per_sample"]),
    "degenerate_in_final_output": sum(cer(h, r) > 1 for h, r in zip(final, refs)),
}

# raw (un-normalized) comparison
raw_refs = [str(s["truth"]) for s in qw["per_sample"]]
raw_base = [str(eo_by[s["id"]]["pred"]) for s in qw["per_sample"]]
raw_final = [str((s if variants["flag_and_confidence"](s) else eo_by[s["id"]])["pred"]) for s in qw["per_sample"]]
d = paired_bootstrap_delta(raw_final, raw_base, raw_refs)
out["heldout_raw_delta"] = {"delta": round(d["delta_corpus_cer"], 4), "ci_95": [round(x, 4) for x in d["ci_95"]]}

# spurious spaces
nosp = lambda t: t.replace(" ", "")  # noqa: E731
d = paired_bootstrap_delta([nosp(h) for h in final], [nosp(h) for h in base], refs)
out["spaces"] = {
    "easyocr_outputs_with_space": sum(" " in h for h in base),
    "references_with_space": sum(" " in r for r in refs),
    "easyocr_cer_spaces_removed": round(corpus([nosp(h) for h in base], refs), 4),
    "arbitration_cer_spaces_removed": round(corpus([nosp(h) for h in final], refs), 4),
    "delta": round(d["delta_corpus_cer"], 4),
    "ci_95": [round(x, 4) for x in d["ci_95"]],
}

# outcome classes and kinds of degenerate output
def outcomes(hyps):
    cs = [cer(h, r) for h, r in zip(hyps, refs)]
    return {"exact": sum(c == 0 for c in cs), "wrong": sum(0 < c <= 1 for c in cs), "degenerate": sum(c > 1 for c in cs)}

out["outcomes"] = {
    "easyocr": outcomes(base),
    "qwen_vl": outcomes([N(s["pred"]) for s in qw["per_sample"]]),
    "arbitration": outcomes(final),
}
bad = [str(s["pred"]) for s in qw["per_sample"] if qcer[s["id"]] > 1]
has_cjk = lambda t: bool(re.search(r"[぀-ヿ一-鿿]", t))  # noqa: E731
arabic_refusal = lambda t: any(k in t for k in ("لا أستطيع", "لا يمكنني", "آسف", "عذرا"))  # noqa: E731
echo = lambda t: any(k in t for k in ("حرفيا", "بدون أي شرح", "النص العربي الموجود"))  # noqa: E731
strip_marks = lambda t: re.sub(r"[ً-ْ]", "", t)  # noqa: E731
kinds = {"arabic_refusal": 0, "instruction_echo": 0, "latin_script": 0, "other": 0}
for t in bad:
    u = strip_marks(t)
    if arabic_refusal(u):
        kinds["arabic_refusal"] += 1
    elif echo(u):
        kinds["instruction_echo"] += 1
    elif len(re.findall(r"[A-Za-z]", t)) >= 2:
        kinds["latin_script"] += 1
    else:
        kinds["other"] += 1
out["degenerate_kinds"] = {
    **kinds,
    "total": len(bad),
    "refusals_opening_with_japanese": sum(arabic_refusal(strip_marks(t)) and has_cjk(t) for t in bad),
    "longer_than_30": sum(len(t) > 30 for t in bad),
    "two_or_more_non_arabic_letters": sum(sum(1 for c in t if c.isalpha() and not ("؀" <= c <= "ۿ")) >= 2 for t in bad),
    "median_length": statistics.median(len(t) for t in bad),
}

# share of edits taken by the two dot pairs
def pair_share(run_name, hyps):
    conf_tab = {(c["ref"], c["hyp"]): c["count"] for c in load(run_name)["confusions_top30"]}
    edits = sum(levenshtein(r, h) for h, r in zip(hyps, refs))
    two = conf_tab[("ي", "ب")] + conf_tab[("ت", "ن")]
    return {"total_edits": edits, "yaa_baa_plus_taa_noon": two, "share": round(two / edits, 3)}

out["dot_pairs"] = {"easyocr": pair_share("easyocr_heldout", base), "arbitration": pair_share("sim_arb_tau40_heldout", final)}

# reference lengths
raw_len = [len(str(s["truth"])) for s in qw["per_sample"]]
out["reference_lengths"] = {
    "raw": {str(k): raw_len.count(k) for k in sorted(set(raw_len))},
    "normalized": {str(k): [len(r) for r in refs].count(k) for k in sorted({len(r) for r in refs})},
    "mean_normalized": round(sum(len(r) for r in refs) / len(refs), 2),
}

# latency
lat = {}
for name in ("easyocr_golden", "easyocr_dev", "easyocr_heldout", "qwen_vl_golden", "qwen_vl_dev", "qwen_vl_heldout"):
    ms = [s["latency_ms"] for s in load(name)["per_sample"]]
    lat[name] = {"median_ms": round(statistics.median(ms), 1), "mean_ms": round(sum(ms) / len(ms), 1)}
total_eo = sum(s["latency_ms"] for s in eo["per_sample"])
total_arb = sum(s["latency_ms"] for s in qw["per_sample"]) + sum(
    eo_by[s["id"]]["latency_ms"] for s in qw["per_sample"] if not variants["flag_and_confidence"](s)
)
lat["heldout_total_time_ratio_arbitration_over_easyocr"] = round(total_arb / total_eo, 2)
out["latency"] = lat

# confidence validity per split, and the bidi check
def corr(run):
    xs = np.array([s["confidence"] for s in run["per_sample"]], dtype=float)
    ys = np.array([1.0 if cer(N(s["pred"]), N(s["truth"])) == 0 else 0.0 for s in run["per_sample"]])
    return round(float(np.corrcoef(xs, ys)[0, 1]), 3)

out["confidence_correlation"] = {
    n: corr(load(n)) for n in ("easyocr_golden", "qwen_vl_golden", "easyocr_dev", "qwen_vl_dev", "easyocr_heldout", "qwen_vl_heldout")
}
rev = [(cer(N(s["pred"]), N(s["truth"])), cer(N(s["pred"])[::-1], N(s["truth"]))) for s in qw["per_sample"]]
out["bidi_check_qwen_heldout"] = {
    "reversed_scores_better": sum(b < a for a, b in rev),
    "of_which_degenerate": sum(b < a and a > 1 for a, b in rev),
    "reversal_reaches_cer_at_most_0.2": sum(b < a and b <= 0.2 for a, b in rev),
}

# dev: what the flag does there
qd, ed = load("qwen_vl_dev"), load("easyocr_dev")
dcer = [cer(N(s["pred"]), N(s["truth"])) for s in qd["per_sample"]]
out["dev_flag"] = {
    "degenerate": sum(c > 1 for c in dcer),
    "degenerate_flagged": sum(c > 1 and degenerate_flag(str(s["pred"])) for s, c in zip(qd["per_sample"], dcer)),
    "exact_flagged": sum(c == 0 and degenerate_flag(str(s["pred"])) for s, c in zip(qd["per_sample"], dcer)),
}

(R / "paper_stats.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=1))
