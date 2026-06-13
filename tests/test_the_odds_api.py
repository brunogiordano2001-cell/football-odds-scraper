from football_odds_scraper.the_odds_api import parse_event_summary, parse_pinnacle_event


def test_parse_event_summary():
    summary = parse_event_summary(
        {
            "id": "abc123",
            "home_team": "Brazil",
            "away_team": "Argentina",
            "commence_time": "2026-06-15T21:00:00Z",
        }
    )
    assert summary is not None
    assert summary["id"] == "abc123"
    assert summary["home_team"] == "Brazil"


def _sample_event(**overrides):
    event = {
        "home_team": "Arsenal",
        "away_team": "Chelsea",
        "commence_time": "2026-06-01T19:00:00Z",
        "bookmakers": [
            {
                "key": "pinnacle",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Arsenal", "price": 2.10},
                            {"name": "Chelsea", "price": 3.50},
                            {"name": "Draw", "price": 3.40},
                        ],
                    },
                    {
                        "key": "totals",
                        "outcomes": [
                            {"name": "Over", "price": 1.95, "point": 2.5},
                            {"name": "Under", "price": 1.90, "point": 2.5},
                        ],
                    },
                ],
            }
        ],
    }
    event.update(overrides)
    return event


def test_parse_pinnacle_event_ok():
    parsed = parse_pinnacle_event(_sample_event())
    assert parsed is not None
    assert parsed["home"] == 2.10
    assert parsed["draw"] == 3.40
    assert parsed["away"] == 3.50
    assert parsed["over"] == 1.95
    assert parsed["under"] == 1.90


def test_parse_pinnacle_event_missing_totals():
    event = _sample_event()
    event["bookmakers"][0]["markets"] = [event["bookmakers"][0]["markets"][0]]
    assert parse_pinnacle_event(event) is None


def test_parse_pinnacle_event_wrong_totals_line():
    event = _sample_event()
    totals = event["bookmakers"][0]["markets"][1]
    totals["outcomes"][0]["point"] = 3.5
    totals["outcomes"][1]["point"] = 3.5
    assert parse_pinnacle_event(event) is None


def test_parse_pinnacle_event_no_pinnacle_bookmaker():
    event = _sample_event()
    event["bookmakers"][0]["key"] = "bet365"
    assert parse_pinnacle_event(event) is None
