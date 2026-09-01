"""Error metrics for Arabic OCR evaluation.

Definitions (stated per protocol §6):

- Per-sample CER = levenshtein(ref, hyp) / max(len(ref), 1). The max(.,1)
  guard means an empty reference with a non-empty hypothesis scores
  len(hyp), not infinity.
- Corpus CER (primary) = total edits / total reference characters
  (guarded to max(total, 1) only when the whole corpus is empty; empty
  individual references contribute edits but no denominator).
- Macro CER (secondary) = unweighted mean of per-sample CER.
- WER analogous over whitespace tokens. WER is NOT meaningful on
  single-word datasets; callers must suppress it there.

Every metric is computed twice: on raw Unicode and after SCORING
normalization. The raw-vs-normalized gap is itself a diagnostic.
"""

from __future__ import annotations

import difflib
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

import numpy as np

try:
    from rapidfuzz.distance import Levenshtein as _RFLev

    def levenshtein(a: str, b: str) -> int:
        return int(_RFLev.distance(a, b))

except ImportError:  # pragma: no cover - exercised only without rapidfuzz

    def levenshtein(a: str, b: str) -> int:
        if a == b:
            return 0
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
            prev = cur
        return prev[-1]


def cer(hyp: str, ref: str) -> float:
    return levenshtein(ref, hyp) / max(len(ref), 1)


def wer(hyp: str, ref: str) -> float:
    rt, ht = ref.split(), hyp.split()
    return levenshtein_tokens(rt, ht) / max(len(rt), 1)


def levenshtein_tokens(a: Sequence[str], b: Sequence[str]) -> int:
    if list(a) == list(b):
        return 0
    prev = list(range(len(b) + 1))
    for i, ta in enumerate(a, 1):
        cur = [i]
        for j, tb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ta != tb)))
        prev = cur
    return prev[-1]


@dataclass(frozen=True)
class CorpusScore:
    corpus_cer: float
    macro_cer: float
    exact_match: float
    n: int


def score_corpus(hyps: Sequence[str], refs: Sequence[str]) -> CorpusScore:
    assert len(hyps) == len(refs) and refs
    edits = [levenshtein(r, h) for h, r in zip(hyps, refs)]
    per_sample_lens = [max(len(r), 1) for r in refs]
    total_ref = sum(len(r) for r in refs)
    return CorpusScore(
        corpus_cer=sum(edits) / max(total_ref, 1),
        macro_cer=float(np.mean([e / l for e, l in zip(edits, per_sample_lens)])),
        exact_match=sum(h == r for h, r in zip(hyps, refs)) / len(refs),
        n=len(refs),
    )


def bidi_check(hyps: Sequence[str], refs: Sequence[str]) -> dict[str, object]:
    """Protocol §3: count samples where the REVERSED hypothesis scores better.

    A high count means text is being stored in visual order somewhere —
    check that before blaming the recognizer.
    """
    flagged = [
        i
        for i, (h, r) in enumerate(zip(hyps, refs))
        if cer(h[::-1], r) < cer(h, r)
    ]
    return {"n_reversed_better": len(flagged), "indices": flagged[:20]}


def paired_bootstrap_delta(
    hyps_a: Sequence[str],
    hyps_b: Sequence[str],
    refs: Sequence[str],
    n_resamples: int = 10_000,
    seed: int = 20260901,
) -> dict[str, object]:
    """Paired bootstrap CI for corpus-CER(A) - corpus-CER(B).

    Negative delta means system A is better. The same resampled indices are
    used for both arms (paired), unlike a one-arm CI which describes a single
    system rather than the difference.
    """
    assert len(hyps_a) == len(hyps_b) == len(refs) and refs
    edits_a = np.array([levenshtein(r, h) for h, r in zip(hyps_a, refs)], dtype=float)
    edits_b = np.array([levenshtein(r, h) for h, r in zip(hyps_b, refs)], dtype=float)
    lens = np.array([len(r) for r in refs], dtype=float)

    rng = np.random.default_rng(seed)
    n = len(refs)
    idx = rng.integers(0, n, size=(n_resamples, n))
    denom = np.maximum(lens[idx].sum(axis=1), 1.0)
    la = edits_a[idx].sum(axis=1) / denom
    lb = edits_b[idx].sum(axis=1) / denom
    deltas = la - lb
    total = max(lens.sum(), 1.0)
    point = edits_a.sum() / total - edits_b.sum() / total
    frac_le = float(np.mean(deltas <= 0))
    frac_ge = float(np.mean(deltas >= 0))
    return {
        "delta_corpus_cer": float(point),
        "ci_95": [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))],
        "p_two_sided": min(1.0, 2 * min(frac_le, frac_ge)),
        "n_resamples": n_resamples,
        "seed": seed,
    }


def confusion_report(
    hyps: Sequence[str], refs: Sequence[str], top_n: int = 30
) -> list[dict[str, object]]:
    """Top character substitutions/insertions/deletions by frequency (ref->hyp)."""
    counts: Counter[tuple[str, str]] = Counter()
    for h, r in zip(hyps, refs):
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
            None, r, h, autojunk=False
        ).get_opcodes():
            if tag == "replace":
                for x, y in zip(r[i1:i2], h[j1:j2]):
                    counts[(x, y)] += 1
                # unmatched tail of an uneven replace block
                if (i2 - i1) > (j2 - j1):
                    for x in r[i1 + (j2 - j1) : i2]:
                        counts[(x, "∅")] += 1
                elif (j2 - j1) > (i2 - i1):
                    for y in h[j1 + (i2 - i1) : j2]:
                        counts[("∅", y)] += 1
            elif tag == "delete":
                for x in r[i1:i2]:
                    counts[(x, "∅")] += 1
            elif tag == "insert":
                for y in h[j1:j2]:
                    counts[("∅", y)] += 1
    return [
        {"ref": x, "hyp": y, "count": c} for (x, y), c in counts.most_common(top_n)
    ]


def fix_break_counts(
    before: Sequence[str], after: Sequence[str], refs: Sequence[str]
) -> dict[str, int]:
    """Protocol §4: for a correction stage, count fixes AND breaks.

    fixed  = sample CER strictly decreased
    broke  = sample CER strictly increased
    An unconditional corrector is frequently net negative; these two numbers
    are reported for every corrector run, always.
    """
    fixed = broke = unchanged = 0
    for b, a, r in zip(before, after, refs):
        cb, ca = cer(b, r), cer(a, r)
        if ca < cb:
            fixed += 1
        elif ca > cb:
            broke += 1
        else:
            unchanged += 1
    return {"fixed": fixed, "broke": broke, "unchanged": unchanged}
