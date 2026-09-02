from moocr.bidi import repair_visual_order


def test_reversed_line_is_repaired():
    original = "المقطع الثاني (في الوفا والشدة):"
    reversed_line = original[::-1]
    fixed, changed = repair_visual_order(reversed_line)
    assert changed
    assert fixed == original


def test_correct_line_untouched():
    line = "وقت الضيق بتعرف مين أصيل ومين دخيلو"
    fixed, changed = repair_visual_order(line)
    assert not changed
    assert fixed == line


def test_single_word_untouched():
    # word-crop outputs (the evaluated domain) must never be flipped
    for w in ["مدرسة", "الدوائر", "كتاب", "ليفونيا", "التأسيسي"]:
        fixed, changed = repair_visual_order(w)
        assert fixed == w and not changed, w


def test_fully_reversed_page_repaired_line_by_line():
    # the observed mechanism: EVERY line reversed, line order kept
    lines = [
        "المقطع الثاني (في الوفا والشدة):",
        "وقت الضيق بتعرف مين أصيل ومين دخيلو",
        "كتير ناس بتخون، لكن خوتي ما بتخون قيلو",
    ]
    text = "\n".join(ln[::-1] for ln in lines)
    fixed, changed = repair_visual_order(text)
    assert changed
    assert fixed == "\n".join(lines)


def test_signal_free_dialect_line_rides_on_page_decision():
    # a line with no orthographic signals is flipped when the page-level
    # evidence says the generation was reversed
    strong = "المقطع الثاني (في الوفا والشدة):"
    weak = "في ساعة العسرة، بلقاهم حواليا سيلو"
    text = strong[::-1] + "\n" + weak[::-1]
    fixed, changed = repair_visual_order(text)
    assert changed
    assert fixed == strong + "\n" + weak


def test_non_arabic_untouched():
    line = "The text in the image is Arabic"
    fixed, changed = repair_visual_order(line)
    assert fixed == line and not changed


def test_empty():
    assert repair_visual_order("") == ("", False)
