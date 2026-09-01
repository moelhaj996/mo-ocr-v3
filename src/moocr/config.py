"""Typed configuration. Single source of truth; no magic numbers in code.

Loadable from TOML (stdlib tomllib); every field has an explicit default.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import BaseModel, Field


class TrOCRConfig(BaseModel):
    checkpoint: str = "microsoft/trocr-base-printed"  # replaced by M2 probe
    revision: str | None = None  # pinned commit hash once probed
    max_new_tokens: int = 48
    num_beams: int = 4


class QwenVLConfig(BaseModel):
    checkpoint: str = "Qwen/Qwen2-VL-2B-Instruct"
    revision: str | None = None
    max_new_tokens: int = 64
    prompt: str = (
        "انسخ النص العربي الموجود في الصورة حرفياً كما هو، "
        "بدون أي شرح أو تعليق أو علامات تنسيق."
    )


class EasyOCRConfig(BaseModel):
    languages: list[str] = Field(default_factory=lambda: ["ar"])
    gpu: bool = False


class CamelBERTConfig(BaseModel):
    checkpoint: str = "CAMeL-Lab/bert-base-arabic-camelbert-mix"
    revision: str | None = None
    # Gating: correction only runs below this recognizer confidence.
    confidence_gate: float = 0.85
    # A correction may change at most this many characters.
    max_edit_distance: int = 2


class SplitConfig(BaseModel):
    seed: int = 20260901
    golden_size: int = 50
    dev_size: int = 200


class Config(BaseModel):
    device: str = "auto"  # auto -> mps if available else cpu
    seed: int = 20260901
    trocr: TrOCRConfig = Field(default_factory=TrOCRConfig)
    qwen_vl: QwenVLConfig = Field(default_factory=QwenVLConfig)
    easyocr: EasyOCRConfig = Field(default_factory=EasyOCRConfig)
    camelbert: CamelBERTConfig = Field(default_factory=CamelBERTConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        if path is None:
            return cls()
        with open(path, "rb") as fh:
            return cls.model_validate(tomllib.load(fh))
