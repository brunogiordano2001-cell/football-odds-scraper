import pytest

from football_odds_scraper.exceptions import ParseError
from football_odds_scraper.models import SelectorConfig
from football_odds_scraper.parser import parse_decimal_odds, parse_match_odds

SAMPLE_HTML = """
<html><body>
  <div class="match-odds">
    <span class="odd-home">2.10</span>
    <span class="odd-draw">3.40</span>
    <span class="odd-away">3.50</span>
    <div class="market-totals">
      <div class="line-2.5">
        <span class="odd-over">1.95</span>
        <span class="odd-under">1.90</span>
      </div>
    </div>
  </div>
</body></html>
"""

CONFIG = SelectorConfig(
    home=".odd-home",
    draw=".odd-draw",
    away=".odd-away",
    over_25=".odd-over",
    under_25=".odd-under",
    market_root=".match-odds",
)


def test_parse_decimal_odds():
    assert parse_decimal_odds("2,10") == 2.10
    assert parse_decimal_odds("  1.85  ") == 1.85


def test_parse_match_odds():
    odds = parse_match_odds(SAMPLE_HTML, "https://test", CONFIG)
    assert odds.home == 2.10
    assert odds.over_25 == 1.95


def test_missing_selector_raises():
    bad = SelectorConfig(
        home=".missing",
        draw=".odd-draw",
        away=".odd-away",
        over_25=".odd-over",
        under_25=".odd-under",
    )
    with pytest.raises(ParseError) as exc:
        parse_match_odds(SAMPLE_HTML, "https://test", bad)
    assert exc.value.field == "home"
