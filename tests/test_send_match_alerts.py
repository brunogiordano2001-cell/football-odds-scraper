"""Tests del script de alertas Telegram."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from scripts import send_match_alerts as alerts


def _fixture(start_minutes: float, *, fixture_id: str = "fx1", status_id: int = 0) -> dict:
    now = datetime.now(timezone.utc)
    start = now + __import__("datetime").timedelta(minutes=start_minutes)
    return {
        "fixtureId": fixture_id,
        "participant1Id": 4717,
        "participant2Id": 4758,
        "startTime": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "statusId": status_id,
        "hasOdds": True,
        "odds": {
            "home": 1.617,
            "draw": 4.04,
            "away": 5.82,
            "over": 1.95,
            "under": 1.90,
            "line": 2.5,
        },
    }


def test_should_alert_in_window():
    now = datetime.now(timezone.utc)
    fixture = _fixture(20.0)
    assert alerts.should_alert(fixture, now=now) is True


def test_should_alert_too_early():
    now = datetime.now(timezone.utc)
    fixture = _fixture(40.0)
    assert alerts.should_alert(fixture, now=now) is False


def test_should_alert_too_late():
    now = datetime.now(timezone.utc)
    fixture = _fixture(5.0)
    assert alerts.should_alert(fixture, now=now) is False


def test_should_alert_skips_finished():
    now = datetime.now(timezone.utc)
    fixture = _fixture(20.0, status_id=2)
    assert alerts.should_alert(fixture, now=now) is False


def test_load_save_sent(tmp_path, monkeypatch):
    monkeypatch.setattr(alerts, "SENT_FILE", tmp_path / "sent_alerts.json")
    assert alerts.load_sent() == set()
    alerts.save_sent({"a", "b"})
    assert alerts.load_sent() == {"a", "b"}


def test_build_message_contains_teams():
    fixture = {
        "participant1Id": 4717,
        "participant2Id": 4758,
        "startTime": "2026-06-15T22:00:00Z",
    }
    prediction = {
        "odds": {"home": 1.6, "draw": 4.0, "away": 5.8, "over": 1.9, "under": 2.0, "line": 2.5},
        "top3": [
            {"score": "1-0", "epv": 0.87, "p_exact": 0.14, "p_result": 0.61},
            {"score": "2-0", "epv": 0.80, "p_exact": 0.10, "p_result": 0.61},
            {"score": "1-1", "epv": 0.75, "p_exact": 0.12, "p_result": 0.22},
        ],
        "top3_coverage": 0.35,
        "sensitivity": {
            "robustness_label": "🟢 Robusta",
            "robustness_desc": "El score no cambia hasta k=100",
            "score_at_inf": (1, 0),
            "coincide_with_argmax": True,
        },
        "lambda_home": 1.82,
        "lambda_away": 0.91,
    }
    msg = alerts.build_message(fixture, prediction)
    assert "Belgium" in msg or "Bélgica" in msg or "🇧🇪" in msg
    assert "EPV" in msg
    assert "Top 3" in msg or "🏆" in msg


def test_generate_chart_requires_two_points():
    fixture = {"fixtureId": "fx1", "participant1Id": 4717, "participant2Id": 4758}
    assert alerts.generate_chart(fixture, snapshots=[]) is None
