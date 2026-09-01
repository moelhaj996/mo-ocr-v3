"""Qwen2-VL engine — vision-language transcription.

Constrained to verbatim transcription by the prompt in config; output is
taken as-is (no chat-artifact stripping beyond whitespace) so failures are
visible in the error analysis rather than papered over.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from moocr.models.base import Recognition, Recognizer
from moocr.models.trocr import _pick_device

if TYPE_CHECKING:
    from PIL import Image

    from moocr.config import Config


class QwenVLEngine(Recognizer):
    name = "qwen_vl"

    def __init__(self, config: "Config") -> None:
        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        cfg = config.qwen_vl
        self._device = _pick_device(config.device)
        dtype = torch.float16 if self._device == "mps" else torch.float32
        self._processor = AutoProcessor.from_pretrained(  # type: ignore[no-untyped-call]
            cfg.checkpoint, revision=cfg.revision,
            min_pixels=64 * 28 * 28, max_pixels=640 * 28 * 28,
        )
        self._model = (
            Qwen2VLForConditionalGeneration.from_pretrained(
                cfg.checkpoint, revision=cfg.revision, torch_dtype=dtype  # type: ignore[arg-type]
            )
            .to(self._device)  # type: ignore[arg-type]
            .eval()
        )
        self._prompt = cfg.prompt
        self._max_new_tokens = cfg.max_new_tokens
        self._torch = torch

    def recognize(self, image: "Image.Image") -> Recognition:
        torch = self._torch
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": self._prompt},
                ],
            }
        ]
        chat_text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[chat_text], images=[image], return_tensors="pt"
        ).to(self._device)
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
            )
        gen = out.sequences[:, inputs.input_ids.shape[1]:]
        text = self._processor.batch_decode(
            gen, skip_special_tokens=True
        )[0].strip()
        conf = None
        if out.scores:
            logprobs = []
            for step, tok in zip(out.scores, gen[0]):
                lp = torch.log_softmax(step[0].float(), dim=-1)[tok]
                logprobs.append(lp)
            if logprobs:
                conf = float(torch.exp(torch.stack(logprobs).mean()))
        return Recognition(text=text, confidence=conf)
