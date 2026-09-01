"""EasyOCR engine (continuity baseline).

Differences from the v2 wrapper, by design:
- confidence is EasyOCR's own recognition confidence (mean over regions),
  not an Arabic-script ratio — script membership says nothing about
  correctness.
- multi-region output is joined in RTL reading order (sorted by line band,
  then right edge descending), not in detector order.
- no exception swallowing: failures propagate to the harness, which
  accounts for them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from moocr.models.base import Recognition, Recognizer

if TYPE_CHECKING:
    from PIL import Image

    from moocr.config import Config


def order_regions_rtl(regions: list[tuple[object, ...]]) -> list[tuple[object, ...]]:
    """Sort (bbox, text, conf) tuples into Arabic reading order.

    Group into horizontal bands by vertical center, then within a band sort
    by right edge, rightmost first.
    """
    def geom(region: Any) -> tuple[float, float, float]:
        bbox = region[0]
        ys = [p[1] for p in bbox]
        xs = [p[0] for p in bbox]
        return (min(ys) + max(ys)) / 2, max(xs), max(ys) - min(ys)

    annotated: list[tuple[tuple[float, float, float], tuple[object, ...]]] = [(geom(r), r) for r in regions]
    annotated.sort(key=lambda t: t[0][0])
    bands: list[list[tuple[tuple[float, float, float], tuple[object, ...]]]] = []
    for (cy, _, h), region in annotated:
        if bands and abs(cy - bands[-1][0][0][0]) < max(h, 1) * 0.6:
            bands[-1].append(((cy, geom(region)[1], h), region))
        else:
            bands.append([((cy, geom(region)[1], h), region)])
    ordered: list[tuple[object, ...]] = []
    for band in bands:
        band.sort(key=lambda t: -t[0][1])  # right edge descending
        ordered.extend(r for _, r in band)
    return ordered


class EasyOCREngine(Recognizer):
    name = "easyocr"

    def __init__(self, config: "Config") -> None:
        import easyocr

        self._reader = easyocr.Reader(
            config.easyocr.languages, gpu=config.easyocr.gpu, verbose=False
        )

    def recognize(self, image: "Image.Image") -> Recognition:
        regions = self._reader.readtext(np.array(image))
        if not regions:
            return Recognition(text="", confidence=0.0, extra={"n_regions": 0})
        ordered = order_regions_rtl(regions)
        text = " ".join(str(r[1]) for r in ordered)
        confs = [float(r[2]) for r in ordered if len(r) > 2]  # type: ignore[arg-type]
        return Recognition(
            text=text,
            confidence=float(np.mean(confs)) if confs else None,
            extra={"n_regions": len(regions)},
        )
