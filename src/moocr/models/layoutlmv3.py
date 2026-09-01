"""LayoutLMv3 wrapper — document structuring interface.

HONEST SCOPE (see LIMITATIONS): no labeled Arabic field-extraction data
exists in this project, so this component is integrated and smoke-tested
only. No field-F1 is claimed anywhere. It becomes evaluable the day
labeled form data arrives; the interface is final, the numbers are not.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from PIL import Image


class LayoutLMv3Structurer:
    def __init__(
        self,
        checkpoint: str = "microsoft/layoutlmv3-base",
        revision: str | None = None,
    ) -> None:
        from transformers import AutoModel, AutoProcessor

        # apply_ocr=False: our own recognizers supply words/boxes; LayoutLMv3
        # must never silently re-OCR with its default (non-Arabic) Tesseract.
        self._processor = AutoProcessor.from_pretrained(
            checkpoint, revision=revision, apply_ocr=False
        )
        self._model = AutoModel.from_pretrained(
            checkpoint, revision=revision
        ).eval()

    def encode(
        self,
        image: "Image.Image",
        words: list[str],
        boxes: list[list[int]],
    ) -> Any:
        """Return contextual embeddings for (word, box) pairs on the page."""
        import torch

        enc = self._processor(
            image, words, boxes=boxes, return_tensors="pt", truncation=True
        )
        with torch.no_grad():
            return self._model(**enc).last_hidden_state
