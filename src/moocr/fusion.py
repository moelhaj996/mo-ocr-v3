"""Arbitration between recognizers.

Routing mechanism only; the policy thresholds are set from measured M3
numbers (confidence validity, degeneration detectability), never assumed.

Current policy shape, justified by the golden-50 error budget:
- primary engine produces a candidate;
- if the candidate trips the ground-truth-free degeneration detector, or
  primary confidence is below its validated threshold, the fallback engine's
  output is used instead;
- optional gated corrector runs last.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from moocr.harness.error_budget import degenerate_flag
from moocr.models.base import Recognition, Recognizer

if TYPE_CHECKING:
    from PIL import Image


class ArbitrationEngine(Recognizer):
    name = "arbitration"

    def __init__(
        self,
        primary: Recognizer,
        fallback: Recognizer,
        primary_min_confidence: float | None = None,
        corrector: object | None = None,
    ) -> None:
        self._primary = primary
        self._fallback = fallback
        self._min_conf = primary_min_confidence
        self._corrector = corrector
        self.name = f"arb({primary.name}->{fallback.name})"

    def recognize(self, image: "Image.Image") -> Recognition:
        p = self._primary.recognize(image)
        routed = "primary"
        chosen = p
        if degenerate_flag(p.text) or (
            self._min_conf is not None
            and p.confidence is not None
            and p.confidence < self._min_conf
        ):
            chosen = self._fallback.recognize(image)
            routed = "fallback"
        text = chosen.text
        corr_meta: dict[str, object] = {}
        if self._corrector is not None:
            text, corr_meta = self._corrector.correct(text, chosen.confidence)  # type: ignore[attr-defined]
        return Recognition(
            text=text,
            confidence=chosen.confidence,
            extra={
                "routed": routed,
                "primary_text": p.text,
                "primary_confidence": p.confidence,
                "correction": corr_meta,
            },
        )
