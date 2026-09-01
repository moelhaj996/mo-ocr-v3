"""RTL region ordering is pure logic — testable without easyocr installed."""

from moocr.models.easyocr_engine import order_regions_rtl


def _region(x1, y1, x2, y2, text):
    return ([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], text, 0.9)


def test_single_line_rtl_order():
    # detector returns left-to-right; Arabic reads right-to-left
    regions = [_region(0, 0, 40, 20, "LEFT"), _region(60, 0, 100, 20, "RIGHT")]
    assert [r[1] for r in order_regions_rtl(regions)] == ["RIGHT", "LEFT"]


def test_two_lines_top_before_bottom():
    regions = [
        _region(0, 50, 100, 70, "line2"),
        _region(0, 0, 100, 20, "line1"),
    ]
    assert [r[1] for r in order_regions_rtl(regions)] == ["line1", "line2"]
