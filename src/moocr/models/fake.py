"""Deterministic fake recognizer for harness tests. No model dependencies."""

from __future__ import annotations

from typing import TYPE_CHECKING

from moocr.models.base import Recognition, Recognizer

if TYPE_CHECKING:
    from PIL import Image


class FakeRecognizer(Recognizer):
    """Returns text from a fixed id->text mapping keyed by image filename."""

    name = "fake"

    def __init__(self, mapping: dict[str, str], fail_ids: set[str] | None = None):
        self._mapping = mapping
        self._fail_ids = fail_ids or set()

    def recognize(self, image: "Image.Image") -> Recognition:
        key = getattr(image, "_moocr_id", None) or (
            image.filename.rsplit("/", 1)[-1].rsplit(".", 1)[0]
            if getattr(image, "filename", "")
            else ""
        )
        if key in self._fail_ids:
            raise RuntimeError(f"synthetic failure for {key}")
        return Recognition(text=self._mapping.get(key, ""), confidence=0.9)
