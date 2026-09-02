"""Recognizer interface and lazy registry.

The harness imports ONLY this module; engine modules (torch, easyocr, ...)
are imported lazily by name so evaluation stays model-agnostic and unit
tests never need model dependencies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from PIL import Image

    from moocr.config import Config


@dataclass
class Recognition:
    text: str
    confidence: float | None = None  # engine-native; validated before routing
    latency_ms: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


class Recognizer(ABC):
    name: str = "abstract"

    @abstractmethod
    def recognize(self, image: "Image.Image") -> Recognition:
        """Read one image. Raise on failure; the harness accounts for it."""


def _easyocr(config: "Config") -> Recognizer:
    from moocr.models.easyocr_engine import EasyOCREngine

    return EasyOCREngine(config)


def _trocr(config: "Config") -> Recognizer:
    from moocr.models.trocr import TrOCREngine

    return TrOCREngine(config)


def _qwen_vl(config: "Config") -> Recognizer:
    from moocr.models.qwen_vl import QwenVLEngine

    return QwenVLEngine(config)


def _arbitration(config: "Config") -> Recognizer:
    from moocr.fusion import ArbitrationEngine

    return ArbitrationEngine(
        primary=get_engine(config.fusion.primary, config),
        fallback=get_engine(config.fusion.fallback, config),
        primary_min_confidence=config.fusion.primary_min_confidence,
    )


def _page(config: "Config") -> Recognizer:
    from moocr.models.page import PageEngine

    return PageEngine(config)


ENGINES: dict[str, Callable[["Config"], Recognizer]] = {
    "arbitration": _arbitration,
    "page": _page,
    "easyocr": _easyocr,
    "trocr": _trocr,
    "qwen_vl": _qwen_vl,
}


def get_engine(name: str, config: "Config") -> Recognizer:
    if name not in ENGINES:
        raise KeyError(f"Unknown engine {name!r}; available: {sorted(ENGINES)}")
    return ENGINES[name](config)
