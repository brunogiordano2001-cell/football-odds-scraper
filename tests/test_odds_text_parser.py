from football_odds_scraper.odds_text_parser import (
    format_parse_success_message,
    format_parse_warning_message,
    parse_odds_paste_text,
)

SAMPLE = """
1X2: 2.15 / 3.27 / 3.98
O/U 2.5: 2.52 / 1.56
TT Home 0.5: 1.29 / 3.56
TT Away 0.5: 1.60 / 2.37
O/U adicional: 1.5:1.485/2.73, 2.0:1.869/2.04, 3.0:4.01/1.26
AH: -0.25 / 1.833 / 2.12
"""


def test_parse_odds_paste_text_full_sample():
    parsed = parse_odds_paste_text(SAMPLE)
    assert parsed.home == 2.15
    assert parsed.draw == 3.27
    assert parsed.away == 3.98
    assert parsed.over == 2.52
    assert parsed.under == 1.56
    assert parsed.goals_line == 2.5
    assert parsed.tt_home_over == 1.29
    assert parsed.tt_home_under == 3.56
    assert parsed.tt_away_over == 1.60
    assert parsed.tt_away_under == 2.37
    assert parsed.ah_line == -0.25
    assert parsed.ah_home == 1.833
    assert parsed.ah_away == 2.12
    assert parsed.ou_curve_text is not None
    assert "1.5:1.485/2.73" in parsed.ou_curve_text
    assert not parsed.missing
    assert "TT Home ✅" in format_parse_success_message(parsed)


def test_parse_odds_paste_text_partial():
    parsed = parse_odds_paste_text("1X2: 2.10 / 3.40 / 3.50\nO/U 2.5: 1.95 / 1.90")
    assert parsed.home == 2.10
    assert parsed.over == 1.95
    assert "TT Home" in parsed.missing
    assert "AH" in parsed.missing
    warning = format_parse_warning_message(parsed)
    assert warning is not None
    assert "TT Home" in warning
