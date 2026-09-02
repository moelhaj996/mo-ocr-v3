"""Page engine: detector-led line segmentation + per-region arbitration.

Full-page VLM reading was observed to emit visual-order text with lost line
structure; word/segment crops are the evaluated domain where arbitration is
measured strong. So: EasyOCR detects and reads regions, each region crop is
re-read by the arbitration engine (Qwen2-VL primary, EasyOCR text as the
no-cost fallback), and lines are joined in RTL reading order.

Page-level accuracy is NOT covered by the quantitative evaluation (no
page-level ground truth exists in this project — see LIMITATIONS).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from moocr.harness.error_budget import degenerate_flag
from moocr.models.base import Recognition, Recognizer
from moocr.models.easyocr_engine import EasyOCREngine, group_regions_rtl

if TYPE_CHECKING:
    from PIL import Image

    from moocr.config import Config


class PageEngine(Recognizer):
    name = "page"

    def __init__(self, config: "Config") -> None:
        from moocr.models.base import get_engine

        self._easyocr = EasyOCREngine(config)
        self._primary = get_engine(config.fusion.primary, config)
        self._min_conf = config.fusion.primary_min_confidence
        # Line-length crops are outside the dev-tuned domain; demand much
        # higher primary confidence there (qualitative default, unevaluated
        # — no page-level ground truth exists; see LIMITATIONS).
        self._min_conf_line = 0.70
        self._line_len = 15
        # Vertical padding is generous (diacritics overhang the box);
        # horizontal is tight so the crop cannot swallow the neighbouring
        # word and duplicate it.
        self._pad_x = 0.04
        self._pad_y = 0.20

    def _crop(
        self, image: "Image.Image", bbox: list[list[float]]
    ) -> "Image.Image":
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        w, h = max(xs) - min(xs), max(ys) - min(ys)
        px, py = w * self._pad_x, h * self._pad_y
        return image.crop(
            (
                max(int(min(xs) - px), 0),
                max(int(min(ys) - py), 0),
                min(int(max(xs) + px), image.width),
                min(int(max(ys) + py), image.height),
            )
        )

    def recognize(self, image: "Image.Image") -> Recognition:
        regions = self._easyocr._reader.readtext(np.array(image))
        if not regions:
            return Recognition(text="", confidence=0.0, extra={"n_regions": 0})
        lines: list[str] = []
        n_primary = 0
        for band in group_regions_rtl(regions):
            parts: list[str] = []
            for region in band:
                bbox = region[0]
                fallback_text = str(region[1])
                rec = self._primary.recognize(
                    self._crop(image, bbox)  # type: ignore[arg-type,unused-ignore]
                )
                text = rec.text.strip()
                bar = (
                    self._min_conf_line
                    if len(fallback_text) > self._line_len
                    else self._min_conf
                )
                use_primary = bool(text) and not degenerate_flag(
                    text, ref_len_hint=len(fallback_text)
                ) and (
                    rec.confidence is None or rec.confidence >= bar
                )
                parts.append(text if use_primary else fallback_text)
                n_primary += int(use_primary)
            lines.append(" ".join(parts))
        return Recognition(
            text="\n".join(lines),
            confidence=None,
            extra={
                "n_regions": len(regions),
                "n_read_by_primary": n_primary,
            },
        )
