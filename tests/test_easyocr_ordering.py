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


def test_diacritic_tall_box_stays_in_band():
    # verification finding: tall diacritic box must not split its line
    regions = [
        _region(60, 10, 100, 30, "RIGHT_tall_diacritics"),
        _region(0, 14, 40, 28, "LEFT_short"),
    ]
    regions[0] = ([[60, 2], [100, 2], [100, 30], [60, 30]], "RIGHT_tall_diacritics", 0.9)
    out = [r[1] for r in order_regions_rtl(regions)]
    assert out == ["RIGHT_tall_diacritics", "LEFT_short"]


def test_skew_drift_does_not_split_line():
    # monotone baseline drift: 6 boxes stepping down 4px each, box height 20
    regions = [
        _region(500 - i * 80, i * 4, 560 - i * 80, 20 + i * 4, f"w{i}")
        for i in range(6)
    ]
    out = [r[1] for r in order_regions_rtl(regions)]
    assert out == [f"w{i}" for i in range(6)]


def test_two_distinct_lines_stay_separate():
    regions = [
        _region(0, 0, 100, 20, "line1"),
        _region(0, 40, 100, 60, "line2"),
    ]
    assert [r[1] for r in order_regions_rtl(regions)] == ["line1", "line2"]
