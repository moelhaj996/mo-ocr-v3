"""TrOCR engine (VisionEncoderDecoder).

Confidence = exp(mean per-token logprob) of the generated sequence — an
actual model probability, usable for arbitration only after its correlation
with correctness is validated on the dev split (never assumed).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from moocr.models.base import Recognition, Recognizer

if TYPE_CHECKING:
    from PIL import Image

    from moocr.config import Config


def _pick_device(pref: str) -> str:
    import torch

    if pref != "auto":
        return pref
    return "mps" if torch.backends.mps.is_available() else "cpu"


class TrOCREngine(Recognizer):
    name = "trocr"

    def __init__(self, config: "Config") -> None:
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        cfg = config.trocr
        self._device = _pick_device(config.device)
        self._processor = TrOCRProcessor.from_pretrained(
            cfg.checkpoint, revision=cfg.revision
        )
        self._model = (
            VisionEncoderDecoderModel.from_pretrained(
                cfg.checkpoint, revision=cfg.revision
            )
            .to(self._device)
            .eval()
        )
        self._max_new_tokens = cfg.max_new_tokens
        self._num_beams = cfg.num_beams
        self._torch = torch

    def recognize(self, image: "Image.Image") -> Recognition:
        torch = self._torch
        pixel_values = self._processor(
            images=image, return_tensors="pt"
        ).pixel_values.to(self._device)
        with torch.no_grad():
            out = self._model.generate(
                pixel_values,
                max_new_tokens=self._max_new_tokens,
                num_beams=self._num_beams,
                output_scores=True,
                return_dict_in_generate=True,
            )
        text = self._processor.batch_decode(
            out.sequences, skip_special_tokens=True
        )[0].strip()
        conf = None
        if out.sequences_scores is not None:
            # beam search: sequences_scores is length-normalized logprob
            conf = float(torch.exp(out.sequences_scores[0]))
        return Recognition(text=text, confidence=conf)
