"""Versioned Arabic text normalization.

Two distinct profiles exist because normalization for *scoring* and
normalization for *output* are different decisions (arabic-nlp-evaluation
protocol §1):

- ``SCORING_V1``: aggressive folding so CER measures recognition, not
  orthography. Applied symmetrically to reference and hypothesis.
- ``OUTPUT_V1``: conservative cleanup for delivered text. Diacritics,
  hamza forms, taa marbuta, and alef maqsura are PRESERVED.

Every reported number must state ``NORM_VERSION`` and the profile used.
MSA-only: these rules are not validated for dialectal orthography.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

NORM_VERSION = "1.1.0"

# U+064B..U+0652 (tanwin, shadda, sukun, harakat) + U+0670 superscript alef.
_DIACRITICS_RE = re.compile("[ً-ْٰ]")
_TATWEEL = "ـ"
# ZWNJ, ZWJ, LRM, RLM, ALM, BOM/ZWNBSP, ZWSP, WORD JOINER, SOFT HYPHEN.
_INVISIBLES_RE = re.compile("[\u200c\u200d\u200e\u200f\u061c\ufeff\u200b\u2060\u00ad]")
_ALEF_RE = re.compile("[أإآٱ]")  # أ إ آ ٱ
_ARABIC_INDIC = {ord("٠") + i: ord("0") + i for i in range(10)}  # ٠..٩
_EXT_ARABIC_INDIC = {ord("۰") + i: ord("0") + i for i in range(10)}  # ۰..۹
_DIGIT_TABLE = {**_ARABIC_INDIC, **_EXT_ARABIC_INDIC}


@dataclass(frozen=True)
class NormalizationProfile:
    """Every rule is an explicit flag; nothing is implicit."""

    name: str
    nfkc: bool  # also folds presentation forms U+FB50..U+FEFF
    strip_invisibles: bool  # ZWJ/ZWNJ + bidi marks
    strip_tatweel: bool
    strip_diacritics: bool
    unify_alef: bool  # أ إ آ ٱ -> ا
    unify_taa_marbuta: bool  # ة -> ه
    unify_alef_maqsura: bool  # ى -> ي
    digits_to_ascii: bool  # ٠١٢ / ۰۱۲ -> 012
    collapse_whitespace: bool


SCORING_V1 = NormalizationProfile(
    name="scoring_v1",
    nfkc=True,
    strip_invisibles=True,
    strip_tatweel=True,
    strip_diacritics=True,
    unify_alef=True,
    unify_taa_marbuta=True,
    unify_alef_maqsura=True,
    digits_to_ascii=True,
    collapse_whitespace=True,
)

OUTPUT_V1 = NormalizationProfile(
    name="output_v1",
    nfkc=True,
    strip_invisibles=True,
    strip_tatweel=True,
    strip_diacritics=False,
    unify_alef=False,
    unify_taa_marbuta=False,
    unify_alef_maqsura=False,
    digits_to_ascii=False,
    collapse_whitespace=True,
)


def normalize(text: str, profile: NormalizationProfile) -> str:
    """Apply ``profile`` to ``text``. Pure and idempotent.

    Invisibles and tatweel are stripped BEFORE NFKC so that a joiner or
    tatweel between a base letter and a combining hamza/madda cannot block
    canonical composition; a final NFC pass guarantees idempotence after
    the substitution rules.
    """
    if profile.strip_invisibles:
        text = _INVISIBLES_RE.sub("", text)
    if profile.strip_tatweel:
        text = text.replace(_TATWEEL, "")
    if profile.nfkc:
        text = unicodedata.normalize("NFKC", text)
    if profile.strip_diacritics:
        text = _DIACRITICS_RE.sub("", text)
    if profile.unify_alef:
        text = _ALEF_RE.sub("ا", text)
    if profile.unify_taa_marbuta:
        text = text.replace("ة", "ه")
    if profile.unify_alef_maqsura:
        text = text.replace("ى", "ي")
    if profile.digits_to_ascii:
        text = text.translate(_DIGIT_TABLE)
    if profile.collapse_whitespace:
        text = " ".join(text.split())
    if profile.nfkc:
        text = unicodedata.normalize("NFC", text)
    return text
