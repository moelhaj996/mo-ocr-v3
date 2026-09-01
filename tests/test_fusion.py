from moocr.fusion import ArbitrationEngine
from moocr.models.base import Recognition, Recognizer


class _Stub(Recognizer):
    def __init__(self, name, text, conf):
        self.name, self._text, self._conf = name, text, conf

    def recognize(self, image):
        return Recognition(text=self._text, confidence=self._conf)


def test_routes_to_fallback_on_degenerate_primary():
    primary = _Stub("p", "The text in the image is Arabic and says something", 0.9)
    fallback = _Stub("f", "مدرسة", 0.5)
    out = ArbitrationEngine(primary, fallback).recognize(None)
    assert out.text == "مدرسة"
    assert out.extra["routed"] == "fallback"


def test_keeps_clean_primary():
    primary = _Stub("p", "مدرسة", 0.9)
    fallback = _Stub("f", "خطأ", 0.5)
    out = ArbitrationEngine(primary, fallback).recognize(None)
    assert out.text == "مدرسة"
    assert out.extra["routed"] == "primary"


def test_confidence_threshold_routing():
    primary = _Stub("p", "مدرسه", 0.2)
    fallback = _Stub("f", "مدرسة", 0.5)
    eng = ArbitrationEngine(primary, fallback, primary_min_confidence=0.4)
    assert eng.recognize(None).extra["routed"] == "fallback"
