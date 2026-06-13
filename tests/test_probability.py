import pytest

from football_odds_scraper.probability import (
    fair_probabilities,
    implied_probabilities,
    overround,
    remove_overround,
)
from football_odds_scraper.models import MatchOdds


def test_implied_and_overround():
    odds = {"home": 2.0, "draw": 3.5, "away": 4.0}
    implied = implied_probabilities(odds)
    assert abs(sum(implied.values()) - 1.0) > 0  # hay margen
    assert overround(odds) > 0


def test_remove_overround_sums_to_one():
    odds = {"home": 2.0, "draw": 3.5, "away": 4.0}
    fair = remove_overround(odds)
    assert abs(sum(fair.values()) - 1.0) < 1e-9


def test_invalid_odds():
    with pytest.raises(ValueError):
        implied_probabilities({"home": 0.9})


def test_fair_probabilities_from_match_odds():
    match = MatchOdds(
        url="https://example.com",
        home=1.9,
        draw=3.6,
        away=4.2,
        over_25=1.85,
        under_25=2.0,
    )
    fair = fair_probabilities(match)
    assert abs(sum(fair["1x2"].values()) - 1.0) < 1e-9
    assert abs(sum(fair["over_under"].values()) - 1.0) < 1e-9
