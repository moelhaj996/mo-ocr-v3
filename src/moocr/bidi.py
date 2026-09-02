"""Ground-truth-free repair of visual-order (reversed) Arabic model output.

Qwen2-VL was observed emitting full-page Arabic in VISUAL order: every
rendered line comes out character-reversed (line order preserved). Protocol
§3: a logically-correct string in visual order scores as a total miss —
repair it, don't blame the recognizer.

Detection uses orthographic asymmetries that reversal destroys:
- "ال/لل" (and وال/بال/فال/كال) are frequent word-INITIALLY only,
- "ة" is word-final; a word STARTING with "ة" is near-impossible,
- the suffixes "ون/ين/ات/ها" are frequent word-FINALLY.

The failure mechanism is systematic per generation, so the decision is made
ONCE per text (summed over lines) and applied to every Arabic-majority
line; a tie or weak evidence keeps the original. Single words with no
signals are never flipped.
"""

from __future__ import annotations

_PUNCT = "،؛:.!؟()[]{}\"'«»-"
_GOOD_PREFIXES = ("ال", "لل", "وال", "بال", "فال", "كال")
_GOOD_SUFFIXES = ("ة", "ون", "ين", "ات", "ها")


def _arabic_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum("؀" <= c <= "ۿ" for c in letters) / len(letters)


def _direction_score(line: str) -> int:
    score = 0
    for token in line.split():
        core = token.strip(_PUNCT)
        if len(core) < 2:
            continue
        if core.startswith(_GOOD_PREFIXES):
            score += 1
        if core.endswith(_GOOD_SUFFIXES):
            score += 1
        if core.startswith("ة"):
            score -= 2
    return score


def repair_visual_order(text: str, margin: int = 1) -> tuple[str, bool]:
    """Return (repaired_text, was_reversed).

    Scores the whole text in both readings (each line reversed
    independently); flips every Arabic-majority line only when the reversed
    reading wins by more than ``margin``.
    """
    lines = text.split("\n")
    arabic = [ln for ln in lines if _arabic_ratio(ln) >= 0.5]
    if not arabic:
        return text, False
    fwd = sum(_direction_score(ln) for ln in arabic)
    rev = sum(_direction_score(ln[::-1]) for ln in arabic)
    if rev <= fwd + margin:
        return text, False
    out = [
        ln[::-1] if _arabic_ratio(ln) >= 0.5 else ln for ln in lines
    ]
    return "\n".join(out), True
