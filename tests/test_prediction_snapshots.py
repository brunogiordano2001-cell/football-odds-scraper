"""Tests del sistema de snapshots de pronósticos."""

from __future__ import annotations

import pytest

from football_odds_scraper import prediction_snapshots as ps
from football_odds_scraper.oddspapi_client import fixture_is_finished


@pytest.fixture
def snapshots_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(ps, "SNAPSHOTS_DIR", tmp_path)
    return tmp_path


def _sample_row(fixture_id: str = "fx1", score: str = "1-0") -> dict:
    return {
        "fixture_id": fixture_id,
        "fixture": {
            "fixtureId": fixture_id,
            "participant1Id": 4717,
            "participant2Id": 4758,
            "startTime": "2026-06-15T22:00:00Z",
            "statusId": 0,
            "odds": {
                "home": 1.617,
                "draw": 4.04,
                "away": 5.82,
                "over": 1.95,
                "under": 1.90,
            },
        },
        "prediction": {
            "prode": {
                "score": score,
                "epv": 0.87,
                "p_exact": 0.142,
            },
            "fair": {
                "1x2": {"home": 0.612, "draw": 0.221, "away": 0.167},
            },
            "fit": {"lambda_home": 1.82, "mu_away": 0.91},
        },
    }


def test_save_and_load_snapshot(snapshots_dir):
    n = ps.save_snapshot([_sample_row()])
    assert n == 1
    loaded = ps.load_snapshots_today()
    assert len(loaded) == 1
    assert len(loaded[0]["fixtures"]) == 1
    fx = loaded[0]["fixtures"][0]
    assert fx["fixtureId"] == "fx1"
    assert fx["prediction"] == "1-0"
    assert fx["lambda_away"] == pytest.approx(0.91)


def test_save_snapshot_appends(snapshots_dir):
    ps.save_snapshot([_sample_row(score="1-0")])
    ps.save_snapshot([_sample_row(score="2-0")])
    loaded = ps.load_snapshots_today()
    assert len(loaded) == 2
    assert loaded[0]["fixtures"][0]["prediction"] == "1-0"
    assert loaded[1]["fixtures"][0]["prediction"] == "2-0"


def test_save_snapshot_skips_finished():
    row = _sample_row()
    row["fixture"]["statusId"] = 2
    rows = [row]
    active = [r for r in rows if not fixture_is_finished(r["fixture"])]
    assert active == []


def test_fixture_history(snapshots_dir):
    ps.save_snapshot([_sample_row(score="1-0")])
    ps.save_snapshot([_sample_row(score="2-0")])
    history = ps.fixture_history("fx1")
    assert len(history) == 2
    assert history[0]["prediction"] == "1-0"
    assert history[1]["prediction"] == "2-0"


def test_summarize_day_movements(snapshots_dir):
    ps.save_snapshot([_sample_row(score="1-1")])
    ps.save_snapshot([_sample_row(score="1-0")])
    ps.save_snapshot([_sample_row(score="2-0")])
    movements = ps.summarize_day_movements()
    assert len(movements) == 1
    assert "1-1" in movements[0]["timeline"]
    assert "2-0" in movements[0]["timeline"]


def test_max_snapshots_trim(snapshots_dir, monkeypatch):
    monkeypatch.setattr(ps, "MAX_SNAPSHOTS_PER_DAY", 2)
    ps.save_snapshot([_sample_row(score="1-0")])
    ps.save_snapshot([_sample_row(score="1-1")])
    ps.save_snapshot([_sample_row(score="2-0")])
    loaded = ps.load_snapshots_today()
    assert len(loaded) == 2
    assert loaded[0]["fixtures"][0]["prediction"] == "1-1"


def test_snapshot_timestamp_hhmm():
    assert ps.snapshot_timestamp_hhmm("2026-06-15T18:30:00-03:00") == "18:30"
