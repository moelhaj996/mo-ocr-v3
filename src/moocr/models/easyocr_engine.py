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

    Regions are grouped into lines by vertical-interval overlap against a
    running band interval (median of member intervals), which is robust to:
    - tall boxes from stacked diacritics (overlap ratio uses the SMALLER
      height, so a tall box still overlaps a short neighbour),
    - monotone baseline drift from page skew (the band interval follows its
      members instead of being frozen at the first region),
    - a tall second-line box reaching toward line 1 (its overlap with line 1
      is small relative to line 1's height).
    Within a band, regions sort by right edge, rightmost first (RTL).
    """

    ordered: list[tuple[object, ...]] = []
    for band in group_regions_rtl(regions):
        ordered.extend(band)
    return ordered


def group_regions_rtl(
    regions: list[tuple[object, ...]]
) -> list[list[tuple[object, ...]]]:
    """Group regions into reading-order lines; see order_regions_rtl."""

    def interval(region: Any) -> tuple[float, float, float]:
        bbox = region[0]
        ys = [p[1] for p in bbox]
        xs = [p[0] for p in bbox]
        return min(ys), max(ys), max(xs)

    annotated = [(interval(r), r) for r in regions]
    annotated.sort(key=lambda t: (t[0][0] + t[0][1]) / 2)

    bands: list[dict[str, Any]] = []
    for (y0, y1, xr), region in annotated:
        h = max(y1 - y0, 1.0)
        placed = False
        for band in bands:
            b0 = float(np.median(band["y0s"]))
            b1 = float(np.median(band["y1s"]))
            overlap = min(y1, b1) - max(y0, b0)
            if overlap / max(min(h, max(b1 - b0, 1.0)), 1.0) >= 0.5:
                band["members"].append((xr, region))
                band["y0s"].append(y0)
                band["y1s"].append(y1)
                placed = True
                break
        if not placed:
            bands.append({"members": [(xr, region)], "y0s": [y0], "y1s": [y1]})

    bands.sort(key=lambda b: float(np.median(b["y0s"])))
    out: list[list[tuple[object, ...]]] = []
    for band in bands:
        band["members"].sort(key=lambda m: -m[0])  # right edge descending
        out.append([r for _, r in band["members"]])
    return out


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
