import pytest

from football_odds_scraper.pinnacle_client import (
    ODDS_ENDPOINT,
    FIXTURES_ENDPOINT,
    LEAGUES_ENDPOINT,
    SOCCER_SPORT_ID,
    _build_request_url,
    extract_event_odds,
    format_fixture_label,
)


def test_build_request_url():
    url = _build_request_url(
        ODDS_ENDPOINT,
        {
            "sportId": SOCCER_SPORT_ID,
            "leagueIds": 1842,
            "eventIds": 999,
            "oddsFormat": "Decimal",
        },
    )
    assert url == (
        "https://api.pinnacle.com/v2/odds?"
        "sportId=29&leagueIds=1842&eventIds=999&oddsFormat=Decimal"
    )
    assert _build_request_url(LEAGUES_ENDPOINT, {"sportId": SOCCER_SPORT_ID}) == (
        "https://api.pinnacle.com/v2/leagues?sportId=29"
    )
    assert _build_request_url(FIXTURES_ENDPOINT, {"sportId": SOCCER_SPORT_ID, "leagueIds": 1842}) == (
        "https://api.pinnacle.com/v1/fixtures?sportId=29&leagueIds=1842"
    )


def test_extract_event_odds():
    payload = {
        "leagues": [
            {
                "id": 100,
                "events": [
                    {
                        "id": 999,
                        "periods": [
                            {
                                "number": 0,
                                "moneyline": {"home": 2.1, "draw": 3.4, "away": 3.5},
                                "totals": [
                                    {"points": 2.5, "over": 1.95, "under": 1.90},
                                ],
                            }
                        ],
                    }
                ],
            }
        ]
    }
    result = extract_event_odds(payload, 999)
    assert result["home"] == 2.1
    assert result["draw"] == 3.4
    assert result["away"] == 3.5
    assert result["over"] == 1.95
    assert result["under"] == 1.90
    assert result["line"] == 2.5


def test_format_fixture_label():
    label = format_fixture_label(
        {"home": "México", "away": "Sudáfrica", "starts": "2026-06-11T19:00:00Z"}
    )
    assert "México vs Sudáfrica" in label
