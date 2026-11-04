from f1dash.ui import formatting


def test_gap_switches_to_laps_once_lapped():
    assert formatting.gap(4.1432) == "+4.143"
    assert formatting.gap(95.0, 1) == "+1 lap"
    assert formatting.gap(170.0, 2) == "+2 laps"
    assert formatting.gap(float("nan")) == "—"


def test_position_is_a_whole_place():
    assert formatting.position(5.3) == "P5"
    assert formatting.position(5.6) == "P6"


def test_tyre_and_slope_text():
    assert formatting.tyre("MEDIUM", 12.0) == "M 12"
    assert formatting.tyre("UNKNOWN", float("nan")) == "?"
    assert formatting.slope(float("nan")) == "—"
    assert formatting.slope(0.0861) == "+0.086"
