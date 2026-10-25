from algorithm import generalise_string, replace_repeated_chars


def test_generalise_string_maps_character_classes():
    assert generalise_string("Aa1 !") == "Uldws"


def test_generalise_string_keeps_prefix_at_specificity_level():
    assert generalise_string("Aa1", specificity_level=2) == "Aad"


def test_generalise_string_empty():
    assert generalise_string("") == ""


def test_generalise_string_negative_level_is_treated_as_zero():
    assert generalise_string("Ab1", specificity_level=-1) == "Uld"


def test_replace_repeated_chars_collapses_runs():
    assert replace_repeated_chars("UUUUddsU") == "U+d+sU"


def test_replace_repeated_chars_short_strings_unchanged():
    assert replace_repeated_chars("") == ""
    assert replace_repeated_chars("U") == "U"


def test_replace_repeated_chars_does_not_collapse_non_runs():
    assert replace_repeated_chars("UlUl") == "UlUl"
