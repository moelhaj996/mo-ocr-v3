"""CamelBERT post-correction — gated reranker, never a rewriter.

Design constraints from the protocol and Phase A evidence:
- runs ONLY below a recognizer-confidence gate (unconditional correction is
  frequently net negative);
- candidates are generated from the dominant OCR confusion families
  (i'jam dot-groups measured in Phase A), capped at
  ``max_edit_distance`` — the model can choose among close variants but can
  never invent a distant token;
- replacement requires the candidate to beat the original by a margin in
  masked-LM pseudo-log-likelihood, else the original stands;
- every application is scored downstream with fix/break counts.
"""

from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from moocr.config import Config

# i'jam (dot) confusion families + common orthographic neighbours, from the
# measured confusion table (ي↔ب 49, ت↔ن 30, ن↔ل 7, خ↔ح, ق↔ف, ض↔ذ ...).
CONFUSION_FAMILIES: list[str] = [
    "بتثني",
    "جحخ",
    "دذ",
    "رز",
    "سش",
    "صض",
    "طظ",
    "عغ",
    "فق",
    "هة",
    "ىي",
    "اأإآ",
    "ؤو",
    "ئي",
    "لن",
]
_FAMILY_OF: dict[str, str] = {
    ch: fam for fam in CONFUSION_FAMILIES for ch in fam
}


def confusion_candidates(word: str, max_edits: int, cap: int = 200) -> list[str]:
    """All variants of ``word`` with up to ``max_edits`` in-family substitutions."""
    positions = [i for i, ch in enumerate(word) if ch in _FAMILY_OF]
    out: list[str] = []
    for k in range(1, max_edits + 1):
        for combo in combinations(positions, k):
            variants = [word]
            for pos in combo:
                nxt = []
                for v in variants:
                    for repl in _FAMILY_OF[v[pos]]:
                        if repl != v[pos]:
                            nxt.append(v[:pos] + repl + v[pos + 1 :])
                variants = nxt
                if len(out) + len(variants) > cap * 4:
                    break
            out.extend(v for v in variants if v != word)
            if len(out) >= cap:
                return list(dict.fromkeys(out))[:cap]
    return list(dict.fromkeys(out))


class CamelBERTCorrector:
    def __init__(self, config: "Config") -> None:
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        cfg = config.camelbert
        self._tokenizer = AutoTokenizer.from_pretrained(
            cfg.checkpoint, revision=cfg.revision
        )
        self._model = AutoModelForMaskedLM.from_pretrained(
            cfg.checkpoint, revision=cfg.revision
        ).eval()
        self._gate = cfg.confidence_gate
        self._max_edits = cfg.max_edit_distance
        self._torch = torch

    def _pll(self, word: str) -> float:
        """Masked-LM pseudo-log-likelihood, length-normalized."""
        torch = self._torch
        enc = self._tokenizer(word, return_tensors="pt")
        ids = enc.input_ids[0]
        content = [
            i
            for i, t in enumerate(ids.tolist())
            if t not in self._tokenizer.all_special_ids
        ]
        if not content:
            return float("-inf")
        total = 0.0
        with torch.no_grad():
            for pos in content:
                masked = ids.clone().unsqueeze(0)
                masked[0, pos] = self._tokenizer.mask_token_id
                logits = self._model(masked).logits[0, pos]
                total += float(
                    torch.log_softmax(logits, dim=-1)[ids[pos]]
                )
        return total / len(content)

    def correct(
        self, text: str, confidence: float | None, margin: float = 0.5
    ) -> tuple[str, dict[str, object]]:
        meta: dict[str, object] = {"applied": False, "gated_out": False}
        if confidence is not None and confidence >= self._gate:
            meta["gated_out"] = True
            return text, meta
        word = text.strip()
        if not word or " " in word or len(word) > 15:
            return text, meta
        candidates = confusion_candidates(word, self._max_edits)
        if not candidates:
            return text, meta
        base = self._pll(word)
        scored = sorted(
            ((self._pll(c), c) for c in candidates), reverse=True
        )
        best_score, best = scored[0]
        meta.update({"pll_original": base, "pll_best": best_score, "best": best})
        if best_score > base + margin:
            meta["applied"] = True
            return best, meta
        return text, meta
