import pytest
from typing import Any

from football_odds_scraper.oddspapi_client import (
    FixtureLiveError,
    OddsPapiError,
    _extract_fixtures_from_odds_by_tournaments,
    _is_world_cup_tournament_candidate,
    _parse_worldcup_fixture_with_odds,
    explain_pinnacle_odds_missing,
    extract_pinnacle_odds,
    fetch_fixture_odds_payload,
    fetch_fixture_pinnacle_odds,
    fetch_worldcup_fixtures_with_odds,
    fixture_has_started,
    fixture_is_finished,
    format_fixture_match_label,
    format_fixture_start_time,
    get_world_cup_tournament_id,
)


def test_is_world_cup_tournament_candidate():
    ok = {
        "tournamentName": "FIFA World Cup",
        "categoryName": "World",
        "categorySlug": "world",
    }
    bad_category = {
        "tournamentName": "World Cup Qualifiers",
        "categoryName": "Europe",
        "categorySlug": "europe",
    }
    assert _is_world_cup_tournament_candidate(ok) is True
    assert _is_world_cup_tournament_candidate(bad_category) is False


def test_get_world_cup_tournament_id_prefers_active_tournament(monkeypatch):
    def fake_get(request_url, params=None, timeout=30):
        class Resp:
            status_code = 200
            ok = True
            text = "[]"

            def __init__(self):
                self.url = str(request_url)

            @staticmethod
            def json():
                return [
                    {
                        "tournamentId": 38785,
                        "tournamentName": "World Cup",
                        "categoryName": "Other",
                        "liveFixtures": 0,
                        "upcomingFixtures": 0,
                    },
                    {
                        "tournamentId": 12345,
                        "tournamentName": "FIFA World Cup",
                        "categoryName": "International",
                        "categorySlug": "international",
                        "liveFixtures": 2,
                        "upcomingFixtures": 10,
                    },
                ]

        return Resp()

    monkeypatch.setattr("football_odds_scraper.oddspapi_client.requests.get", fake_get)
    assert get_world_cup_tournament_id("test-key") == "12345"


def test_extract_fixtures_from_odds_by_tournaments_nested():
    payload = [
        {
            "tournamentId": 12345,
            "fixtures": [
                {"fixtureId": "id1000001abc", "hasOdds": True},
                {"fixtureId": "id1000002abc", "hasOdds": False},
            ],
        }
    ]
    fixtures = _extract_fixtures_from_odds_by_tournaments(payload)
    assert len(fixtures) == 2
    assert fixtures[0]["fixtureId"] == "id1000001abc"


def test_parse_worldcup_fixture_with_odds():
    item = {
        "fixtureId": "id1000001abc",
        "participant1Id": 4752,
        "participant2Id": 4479,
        "startTime": "2026-06-15T21:00:00Z",
        "hasOdds": True,
        "bookmakerOdds": _sample_bookmaker_odds(),
    }
    parsed = _parse_worldcup_fixture_with_odds(item)
    assert parsed is not None
    assert parsed["participant1Id"] == 4752
    assert parsed["odds"]["home"] == 2.10


def test_fetch_worldcup_fixtures_with_odds(monkeypatch):
    calls: list[str] = []
    captured_params: dict[str, Any] = {}

    def fake_get(request_url, params=None, timeout=30):
        calls.append(str(request_url))
        if request_url.endswith("/odds-by-tournaments"):
            captured_params.update(params or {})

        class Resp:
            status_code = 200
            ok = True
            text = "[]"

            def __init__(self):
                self.url = str(request_url)

            @staticmethod
            def json():
                if request_url.endswith("/tournaments"):
                    return [
                        {
                            "tournamentId": 12345,
                            "tournamentName": "FIFA World Cup",
                            "categoryName": "World",
                            "categorySlug": "world",
                            "liveFixtures": 1,
                            "upcomingFixtures": 5,
                        }
                    ]
                if request_url.endswith("/odds-by-tournaments"):
                    return [
                        {
                            "tournamentId": 12345,
                            "fixtures": [
                                {
                                    "fixtureId": "id1000001mex",
                                    "participant1Id": 4479,
                                    "participant2Id": 4752,
                                    "startTime": "2026-06-11T19:00:00Z",
                                    "hasOdds": True,
                                    "bookmakerOdds": _sample_bookmaker_odds(),
                                }
                            ],
                        }
                    ]
                return []

        return Resp()

    monkeypatch.setattr("football_odds_scraper.oddspapi_client.requests.get", fake_get)
    fixtures, tournament_id, _ = fetch_worldcup_fixtures_with_odds("test-key")
    assert tournament_id == "12345"
    assert len(fixtures) == 1
    assert fixtures[0]["participant1Id"] == 4479
    assert fixtures[0]["odds"]["over"] == 1.95
    assert any(url.endswith("/odds-by-tournaments") for url in calls)
    assert captured_params["tournamentIds"] == "12345"
    assert "tournamentId" not in captured_params
    assert captured_params["bookmaker"] == "pinnacle"
    assert "from" not in captured_params
    assert "to" not in captured_params


def test_fetch_worldcup_fixtures_handles_404(monkeypatch):
    def fake_get(request_url, params=None, timeout=30):
        class Resp:
            status_code = 404
            ok = False
            text = '{"message":"No fixtures found"}'

            def __init__(self):
                self.url = str(request_url)

            @staticmethod
            def json():
                return {"message": "No fixtures found"}

        return Resp()

    monkeypatch.setattr("football_odds_scraper.oddspapi_client.requests.get", fake_get)
    monkeypatch.setattr(
        "football_odds_scraper.oddspapi_client.get_world_cup_tournament_id",
        lambda _key: "16",
    )
    fixtures, tournament_id, empty_msg = fetch_worldcup_fixtures_with_odds(
        "test-key", tournament_id="16"
    )
    assert fixtures == []
    assert tournament_id == "16"
    assert empty_msg is not None
    assert "No hay partidos próximos" in empty_msg


def test_fixture_date_window_30_days():
    from datetime import datetime, timezone

    from football_odds_scraper.oddspapi_client import _fixture_date_window

    now = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
    date_from, date_to = _fixture_date_window(now=now)
    assert date_from == "2026-06-01"
    assert date_to == "2026-07-01"


def _sample_bookmaker_odds(**totals_overrides):
    totals_market = {
        "bookmakerMarketId": "line/29/1980/totals",
        "outcomes": {
            "1011": {
                "players": {"0": {"price": 1.90, "bookmakerOutcomeId": "2.5/under"}},
            },
            "1010": {
                "players": {"0": {"price": 1.95, "bookmakerOutcomeId": "2.5/over"}},
            },
        },
    }
    totals_market.update(totals_overrides)
    return {
        "pinnacle": {
            "markets": {
                "1005": {
                    "outcomes": {
                        "1006": {
                            "players": {"0": {"price": 2.10, "bookmakerOutcomeId": "home"}},
                        },
                        "1007": {
                            "players": {"0": {"price": 3.40, "bookmakerOutcomeId": "draw"}},
                        },
                        "1008": {
                            "players": {"0": {"price": 3.50, "bookmakerOutcomeId": "away"}},
                        },
                    }
                },
                "1010": totals_market,
            }
        }
    }


def test_extract_pinnacle_odds_ok():
    extracted = extract_pinnacle_odds(_sample_bookmaker_odds())
    assert extracted is not None
    assert extracted["home"] == 2.10
    assert extracted["draw"] == 3.40
    assert extracted["away"] == 3.50
    assert extracted["over"] == 1.95
    assert extracted["under"] == 1.90
    assert extracted["ah_line"] is None
    assert extracted["ah_home"] is None
    assert extracted["ah_away"] is None
    assert extracted["correct_score_odds"] is None


def test_extract_pinnacle_odds_correct_score():
    odds = _sample_bookmaker_odds()
    odds["pinnacle"]["markets"]["1030"] = {
        "bookmakerMarketId": "line/29/1980/correct-score",
        "limit": 1000,
        "outcomes": {
            "1031": {
                "players": {"0": {"price": 7.5, "bookmakerOutcomeId": "1:0"}},
            },
            "1032": {
                "players": {"0": {"price": 9.0, "bookmakerOutcomeId": "0:0"}},
            },
            "1033": {
                "players": {"0": {"price": 6.5, "bookmakerOutcomeId": "1:1"}},
            },
            "1034": {
                "players": {"0": {"price": 12.0, "bookmakerOutcomeId": "2:1"}},
            },
        },
    }
    extracted = extract_pinnacle_odds(odds)
    assert extracted is not None
    cs = extracted["correct_score_odds"]
    assert cs is not None
    assert cs[(1, 0)] == 7.5
    assert cs[(0, 0)] == 9.0
    assert cs[(1, 1)] == 6.5


def test_ou_curve_filters_basketball_lines():
    odds = _sample_bookmaker_odds()
    odds["pinnacle"]["markets"]["1099"] = {
        "bookmakerMarketId": "line/8581/basketball/totals",
        "outcomes": {
            "2001": {"players": {"0": {"price": 1.90, "bookmakerOutcomeId": "8.0/over"}}},
            "2002": {"players": {"0": {"price": 1.90, "bookmakerOutcomeId": "8.0/under"}}},
            "2003": {"players": {"0": {"price": 2.05, "bookmakerOutcomeId": "8.5/over"}}},
            "2004": {"players": {"0": {"price": 1.80, "bookmakerOutcomeId": "8.5/under"}}},
            "2005": {"players": {"0": {"price": 2.20, "bookmakerOutcomeId": "9.0/over"}}},
            "2006": {"players": {"0": {"price": 1.70, "bookmakerOutcomeId": "9.0/under"}}},
        },
    }
    extracted = extract_pinnacle_odds(odds)
    assert extracted is not None
    curve = extracted.get("ou_curve") or []
    lines = [point for point, _, _ in curve]
    assert 8.0 not in lines
    assert 8.5 not in lines
    assert 9.0 not in lines
    assert all(0.5 <= point <= 5.5 for point in lines)


def test_extract_main_totals_picks_balanced_line():
    odds = _sample_bookmaker_odds()
    odds["pinnacle"]["markets"]["1010"] = {
        "bookmakerMarketId": "line/29/1980/totals",
        "outcomes": {
            "1011": {"players": {"0": {"price": 1.40, "bookmakerOutcomeId": "2.5/over"}}},
            "1010": {"players": {"0": {"price": 3.00, "bookmakerOutcomeId": "2.5/under"}}},
        },
    }
    odds["pinnacle"]["markets"]["1050"] = {
        "bookmakerMarketId": "line/29/1980/totals-alt",
        "outcomes": {
            "1051": {"players": {"0": {"price": 2.10, "bookmakerOutcomeId": "3.5/over"}}},
            "1052": {"players": {"0": {"price": 1.75, "bookmakerOutcomeId": "3.5/under"}}},
        },
    }
    extracted = extract_pinnacle_odds(odds)
    assert extracted is not None
    assert extracted["line"] == 3.5
    assert extracted["over"] == 2.10
    assert extracted["under"] == 1.75


def test_parse_ou_curve_input():
    from football_odds_scraper.oddspapi_client import parse_ou_curve_input

    parsed = parse_ou_curve_input("1.5:1.48/2.73, 2.0:1.87/2.04, 3.0:4.01/1.26")
    assert parsed == [(1.5, 1.48, 2.73), (2.0, 1.87, 2.04), (3.0, 4.01, 1.26)]
    assert parse_ou_curve_input("") is None
    assert parse_ou_curve_input("invalid") is None


def test_parse_ah_curve_input():
    from football_odds_scraper.oddspapi_client import parse_ah_curve_input

    parsed = parse_ah_curve_input("-0.25:1.83/2.12, -0.5:2.14/1.79, +0.25:1.95/1.95")
    assert parsed == [(-0.5, 2.14, 1.79), (-0.25, 1.83, 2.12), (0.25, 1.95, 1.95)]
    assert parse_ah_curve_input("") is None
    assert parse_ah_curve_input("invalid") is None


def test_parse_correct_score_input():
    from football_odds_scraper.oddspapi_client import parse_correct_score_input

    parsed = parse_correct_score_input("1-0:7.5, 0-0:9.0, 1-1:6.5")
    assert parsed == {(1, 0): 7.5, (0, 0): 9.0, (1, 1): 6.5}


def test_format_pinnacle_cs():
    from football_odds_scraper.oddspapi_client import format_pinnacle_cs

    assert format_pinnacle_cs(None) == "📊 Solo 1X2+OU"
    assert "CS incluido" in format_pinnacle_cs(
        {"correct_score_odds": {(1, 0): 7.5, (0, 0): 9.0, (1, 1): 6.5}}
    )
    odds = _sample_bookmaker_odds()
    odds["pinnacle"]["markets"]["1020"] = {
        "bookmakerMarketId": "line/29/1980/handicap",
        "limit": 500,
        "outcomes": {
            "1021": {
                "players": {"0": {"price": 1.92, "bookmakerOutcomeId": "-0.25/home"}},
            },
            "1022": {
                "players": {"0": {"price": 1.98, "bookmakerOutcomeId": "-0.25/away"}},
            },
        },
    }
    odds["pinnacle"]["markets"]["1025"] = {
        "bookmakerMarketId": "line/29/1980/handicap-alt",
        "limit": 100,
        "outcomes": {
            "1026": {
                "players": {"0": {"price": 2.05, "bookmakerOutcomeId": "-0.5/home"}},
            },
            "1027": {
                "players": {"0": {"price": 1.85, "bookmakerOutcomeId": "-0.5/away"}},
            },
        },
    }
    extracted = extract_pinnacle_odds(odds)
    assert extracted is not None
    assert extracted["ah_line"] == -0.25
    assert extracted["ah_home"] == 1.92
    assert extracted["ah_away"] == 1.98
    ah_curve = extracted.get("ah_curve")
    assert ah_curve is not None
    assert len(ah_curve) == 2
    assert ah_curve[0][0] == -0.5
    assert ah_curve[1][0] == -0.25


def test_format_pinnacle_ah():
    from football_odds_scraper.oddspapi_client import format_pinnacle_ah

    assert format_pinnacle_ah(None) == "AH: no disponible"
    assert format_pinnacle_ah({"ah_line": -0.25, "ah_home": 1.92, "ah_away": 1.98}) == (
        "AH: -0.25 (1.92 / 1.98)"
    )


def test_extract_pinnacle_odds_real_api_structure():
    odds = {
        "pinnacle": {
            "markets": {
                "1005": {
                    "outcomes": {
                        "1006": {
                            "players": {"0": {"bookmakerOutcomeId": "home", "price": 2.21}},
                        },
                        "1007": {
                            "players": {"0": {"bookmakerOutcomeId": "draw", "price": 3.10}},
                        },
                        "1008": {
                            "players": {"0": {"bookmakerOutcomeId": "away", "price": 3.80}},
                        },
                    }
                },
                "1010": {
                    "bookmakerMarketId": "...totals...",
                    "outcomes": {
                        "1011": {
                            "players": {"0": {"bookmakerOutcomeId": "2.5/under", "price": 1.709}},
                        },
                        "1010": {
                            "players": {"0": {"bookmakerOutcomeId": "2.5/over", "price": 2.21}},
                        },
                    },
                },
            }
        }
    }
    extracted = extract_pinnacle_odds(odds)
    assert extracted is not None
    assert extracted["home"] == 2.21
    assert extracted["over"] == 2.21
    assert extracted["under"] == 1.709


def test_extract_pinnacle_odds_finds_totals_by_outcome_id():
    odds = _sample_bookmaker_odds(
        outcomes={
            "1011": {
                "players": {"0": {"price": 1.88, "bookmakerOutcomeId": "2.5/under"}},
            },
            "1010": {
                "players": {"0": {"price": 2.05, "bookmakerOutcomeId": "2.5/over"}},
            },
        },
    )
    extracted = extract_pinnacle_odds(odds)
    assert extracted is not None
    assert extracted["over"] == 2.05
    assert extracted["under"] == 1.88


def test_extract_team_totals_falls_back_to_higher_lines():
    odds = _sample_bookmaker_odds()
    odds["pinnacle"]["markets"]["1040"] = {
        "bookmakerMarketId": "line/29/1980/team-totals",
        "outcomes": {
            "1041": {
                "players": {"0": {"price": 1.35, "bookmakerOutcomeId": "home/1.5/over"}},
            },
            "1042": {
                "players": {"0": {"price": 3.20, "bookmakerOutcomeId": "home/1.5/under"}},
            },
            "1043": {
                "players": {"0": {"price": 2.670, "bookmakerOutcomeId": "away/0.5/over"}},
            },
            "1044": {
                "players": {"0": {"price": 1.395, "bookmakerOutcomeId": "away/0.5/under"}},
            },
        },
    }
    extracted = extract_pinnacle_odds(odds)
    assert extracted is not None
    assert extracted["tt_home_over"] == 1.35
    assert extracted["tt_home_under"] == 3.20
    assert extracted["tt_home_line"] == 1.5
    assert extracted["tt_away_over"] == 2.670
    assert extracted["tt_away_under"] == 1.395
    assert extracted["tt_away_line"] == 0.5


def test_extract_team_totals_away_falls_back_to_15():
    odds = _sample_bookmaker_odds()
    odds["pinnacle"]["markets"]["1040"] = {
        "bookmakerMarketId": "line/29/1980/team-totals",
        "outcomes": {
            "1041": {
                "players": {"0": {"price": 1.35, "bookmakerOutcomeId": "home/0.5/over"}},
            },
            "1042": {
                "players": {"0": {"price": 3.20, "bookmakerOutcomeId": "home/0.5/under"}},
            },
            "1043": {
                "players": {"0": {"price": 1.85, "bookmakerOutcomeId": "away/1.5/over"}},
            },
            "1044": {
                "players": {"0": {"price": 2.05, "bookmakerOutcomeId": "away/1.5/under"}},
            },
        },
    }
    extracted = extract_pinnacle_odds(odds)
    assert extracted is not None
    assert extracted["tt_home_line"] == 0.5
    assert extracted["tt_away_over"] == 1.85
    assert extracted["tt_away_under"] == 2.05
    assert extracted["tt_away_line"] == 1.5


def test_extract_pinnacle_odds_team_totals_and_ou_curve():
    odds = _sample_bookmaker_odds()
    odds["pinnacle"]["markets"]["1040"] = {
        "bookmakerMarketId": "line/29/1980/team-totals",
        "outcomes": {
            "1041": {
                "players": {"0": {"price": 1.35, "bookmakerOutcomeId": "home/0.5/over"}},
            },
            "1042": {
                "players": {"0": {"price": 3.20, "bookmakerOutcomeId": "home/0.5/under"}},
            },
            "1043": {
                "players": {"0": {"price": 1.55, "bookmakerOutcomeId": "away/0.5/over"}},
            },
            "1044": {
                "players": {"0": {"price": 2.45, "bookmakerOutcomeId": "away/0.5/under"}},
            },
        },
    }
    odds["pinnacle"]["markets"]["1050"] = {
        "bookmakerMarketId": "line/29/1980/totals",
        "outcomes": {
            "1051": {
                "players": {"0": {"price": 1.485, "bookmakerOutcomeId": "1.5/over"}},
            },
            "1052": {
                "players": {"0": {"price": 2.73, "bookmakerOutcomeId": "1.5/under"}},
            },
            "1053": {
                "players": {"0": {"price": 1.869, "bookmakerOutcomeId": "2.0/over"}},
            },
            "1054": {
                "players": {"0": {"price": 2.04, "bookmakerOutcomeId": "2.0/under"}},
            },
            "1055": {
                "players": {"0": {"price": 2.10, "bookmakerOutcomeId": "3.0/over"}},
            },
            "1056": {
                "players": {"0": {"price": 1.75, "bookmakerOutcomeId": "3.0/under"}},
            },
        },
    }
    odds["pinnacle"]["markets"]["1060"] = {
        "bookmakerMarketId": "line/29/1980/1/totals",
        "outcomes": {
            "1061": {
                "players": {"0": {"price": 1.50, "bookmakerOutcomeId": "0.5/over"}},
            },
            "1062": {
                "players": {"0": {"price": 2.50, "bookmakerOutcomeId": "0.5/under"}},
            },
        },
    }
    extracted = extract_pinnacle_odds(odds)
    assert extracted is not None
    assert extracted["tt_home_over"] == 1.35
    assert extracted["tt_home_under"] == 3.20
    assert extracted["tt_home_line"] == 0.5
    assert extracted["tt_away_over"] == 1.55
    assert extracted["tt_away_under"] == 2.45
    assert extracted["tt_away_line"] == 0.5
    curve = extracted["ou_curve"]
    assert curve is not None
    assert len(curve) >= 3
    assert curve[0][0] == 1.5
    assert curve[0][1] == 1.485
    assert curve[0][2] == 2.73
    lines = [point for point, _, _ in curve]
    assert lines == sorted(lines)
    assert all(point != 0.5 for point in lines)


def test_format_pinnacle_calibration():
    from football_odds_scraper.oddspapi_client import format_pinnacle_calibration

    assert format_pinnacle_calibration(None) == "📊 Solo 1X2+OU"
    odds = {
        "tt_home_over": 1.35,
        "tt_home_under": 3.20,
        "tt_away_over": 1.55,
        "tt_away_under": 2.45,
        "ou_curve": [(1.5, 1.48, 2.73), (2.0, 1.87, 2.04), (2.5, 1.95, 1.90)],
        "correct_score_odds": {(1, 0): 7.5},
        "ah_curve": [(-0.5, 2.05, 1.85), (-0.25, 1.92, 1.98), (0.0, 1.95, 1.95)],
    }
    badge = format_pinnacle_calibration(odds)
    assert "TT" in badge
    assert "O/U×3" in badge
    assert "CS" in badge
    assert "AH×3" in badge


def test_explain_pinnacle_odds_missing_messages():
    assert "no disponible" in explain_pinnacle_odds_missing({}).lower()

    only_1x2 = {
        "pinnacle": {
            "markets": {
                "1005": {
                    "outcomes": {
                        "1006": {
                            "players": {"0": {"price": 2.10, "bookmakerOutcomeId": "home"}},
                        },
                        "1007": {
                            "players": {"0": {"price": 3.40, "bookmakerOutcomeId": "draw"}},
                        },
                        "1008": {
                            "players": {"0": {"price": 3.50, "bookmakerOutcomeId": "away"}},
                        },
                    }
                }
            }
        }
    }
    assert "O/U" in explain_pinnacle_odds_missing(only_1x2)

    missing_1x2 = {
        "pinnacle": {
            "markets": {
                "1010": {
                    "bookmakerMarketId": "totals",
                    "outcomes": {
                        "1010": {
                            "players": {"0": {"price": 1.95, "bookmakerOutcomeId": "2.5/over"}},
                        },
                        "1011": {
                            "players": {"0": {"price": 1.90, "bookmakerOutcomeId": "2.5/under"}},
                        },
                    },
                }
            }
        }
    }
    assert "1X2" in explain_pinnacle_odds_missing(missing_1x2)


def test_parse_pinnacle_odds_ok():
    payload = {
        "fixtureId": "id123",
        "bookmakerOdds": _sample_bookmaker_odds(),
    }
    from football_odds_scraper.oddspapi_client import parse_pinnacle_odds

    parsed = parse_pinnacle_odds(payload)
    assert parsed is not None
    assert parsed["home"] == 2.10
    assert parsed["line"] == 2.5


def test_extract_pinnacle_odds_missing_totals():
    odds = {
        "pinnacle": {
            "markets": {
                "1005": {
                    "outcomes": {
                        "1006": {
                            "players": {"0": {"price": 2.10, "bookmakerOutcomeId": "home"}},
                        },
                        "1007": {
                            "players": {"0": {"price": 3.40, "bookmakerOutcomeId": "draw"}},
                        },
                        "1008": {
                            "players": {"0": {"price": 3.50, "bookmakerOutcomeId": "away"}},
                        },
                    }
                }
            }
        }
    }
    assert extract_pinnacle_odds(odds) is None


def test_parse_worldcup_fixture_equipo_fallback():
    item = {
        "fixtureId": "id1000001abc",
        "participant1Id": 42,
        "participant2Id": 99,
        "startTime": "2026-06-15T21:00:00Z",
        "hasOdds": True,
        "bookmakerOdds": _sample_bookmaker_odds(),
    }
    parsed = _parse_worldcup_fixture_with_odds(item)
    assert parsed is not None
    assert parsed["participant1Id"] == 42


def test_format_fixture_start_time_argentina():
    def fake_get(request_url, params=None, timeout=30):
        class Resp:
            status_code = 200
            ok = True
            text = "[]"

            def __init__(self):
                self.url = str(request_url)

            @staticmethod
            def json():
                if request_url.endswith("/tournaments"):
                    return [
                        {
                            "tournamentId": 12345,
                            "tournamentName": "FIFA World Cup",
                            "categoryName": "World",
                            "liveFixtures": 0,
                            "upcomingFixtures": 1,
                        }
                    ]
                if request_url.endswith("/odds-by-tournaments"):
                    return [
                        {
                            "tournamentId": 12345,
                            "fixtures": [
                                {
                                    "fixtureId": "id1000001mex",
                                    "participant1Id": 4479,
                                    "participant2Id": 4752,
                                    "startTime": "2026-06-11T19:00:00Z",
                                    "hasOdds": True,
                                    "bookmakerOdds": _sample_bookmaker_odds(),
                                },
                                {
                                    "fixtureId": "id1000002esp",
                                    "participant1Id": 100,
                                    "participant2Id": 101,
                                    "startTime": "2026-06-11T20:00:00Z",
                                    "hasOdds": False,
                                    "bookmakerOdds": {},
                                },
                            ],
                        }
                    ]
                return []

        return Resp()

    monkeypatch.setattr("football_odds_scraper.oddspapi_client.requests.get", fake_get)
    fixtures, tournament_id, _ = fetch_worldcup_fixtures_with_odds("test-key")
    assert tournament_id == "12345"
    assert len(fixtures) == 1
    assert fixtures[0]["participant1Id"] is not None


def test_format_fixture_match_label():
    fixture = {
        "participant1Id": 4819,
        "participant2Id": 4781,
        "startTime": "2026-06-12T22:00:00Z",
    }
    label = format_fixture_match_label(fixture)
    assert "🇦🇷 Argentina vs 🇲🇽 Mexico" in label
    assert "12/06 19:00" in label


def test_format_fixture_start_time_argentina():
    assert format_fixture_start_time("2026-06-15T21:00:00Z") == "15/06 18:00"


def test_fixture_has_started():
    from datetime import datetime, timezone

    fixture = {"startTime": "2026-06-15T21:00:00Z"}
    now = datetime(2026, 6, 15, 22, 0, tzinfo=timezone.utc)
    assert fixture_has_started(fixture, now=now) is True

    now_before = datetime(2026, 6, 15, 20, 0, tzinfo=timezone.utc)
    assert fixture_has_started(fixture, now=now_before) is False


def test_fixture_is_finished():
    assert fixture_is_finished({"statusId": 2}) is True
    assert fixture_is_finished({"statusId": 1}) is False
    assert fixture_is_finished({}) is False


def test_handle_response_restricted_access():
    from football_odds_scraper.oddspapi_client import _handle_response

    class Resp:
        status_code = 403
        ok = False
        text = '{"code":"RESTRICTED_ACCESS","message":"Live odds restricted"}'

        @staticmethod
        def json():
            return {"code": "RESTRICTED_ACCESS", "message": "Live odds restricted"}

    with pytest.raises(FixtureLiveError):
        _handle_response(Resp())


def test_fetch_fixture_odds_uses_correct_params(monkeypatch):
    captured: dict[str, Any] = {}

    def fake_get(request_url, params=None, timeout=30):
        captured["params"] = params

        class Resp:
            status_code = 200
            ok = True
            text = '{"bookmakerOdds":{"pinnacle":{"markets":{}}}}'

            def __init__(self):
                self.url = f"{request_url}?fixtureId={params['fixtureId']}"

            @staticmethod
            def json():
                return {"bookmakerOdds": {"pinnacle": {"markets": {}}}}

        return Resp()

    monkeypatch.setattr("football_odds_scraper.oddspapi_client.requests.get", fake_get)
    fetch_fixture_odds_payload("test-key", "id1000001abc")

    assert captured["params"]["bookmaker"] == "pinnacle"
    assert captured["params"]["oddsFormat"] == "decimal"
    assert captured["params"]["fixtureId"] == "id1000001abc"
    assert "bookmakers" not in captured["params"]


def test_fetch_fixture_odds_rejects_invalid_fixture_id():
    with pytest.raises(AssertionError, match="fixtureId inválido"):
        fetch_fixture_odds_payload("test-key", "12345")


def test_fetch_fixture_pinnacle_odds_live(monkeypatch):
    def fake_get(request_url, params=None, timeout=30):
        class Resp:
            status_code = 403
            ok = False
            text = '{"code":"RESTRICTED_ACCESS"}'

            def __init__(self):
                self.url = str(request_url)

            @staticmethod
            def json():
                return {"code": "RESTRICTED_ACCESS"}

        return Resp()

    monkeypatch.setattr("football_odds_scraper.oddspapi_client.requests.get", fake_get)
    with pytest.raises(FixtureLiveError):
        fetch_fixture_pinnacle_odds("test-key", "id1000001abc")
