"""Persistencia de snapshots diarios de pronósticos (Mundial 2026)."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from football_odds_scraper.oddspapi_client import AR_TZ, extract_pinnacle_odds
from football_odds_scraper.world_cup_teams import get_team_display

MAX_SNAPSHOTS_PER_DAY = 288
SNAPSHOTS_DIR = Path(__file__).resolve().parent.parent / "scripts" / "snapshots"


def _ensure_snapshots_dir() -> Path:
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    return SNAPSHOTS_DIR


def snapshot_file_for_date(for_date: date | None = None) -> Path:
    day = for_date or datetime.now(AR_TZ).date()
    return _ensure_snapshots_dir() / f"wc2026_{day.strftime('%Y%m%d')}.json"


def load_snapshots_today() -> list[dict[str, Any]]:
    """Lista de snapshots del día actual (orden cronológico)."""
    path = snapshot_file_for_date()
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    snapshots = payload.get("snapshots")
    if not isinstance(snapshots, list):
        return []
    return snapshots


def _load_day_payload(path: Path) -> dict[str, Any]:
    if path.is_file():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return payload
        except (json.JSONDecodeError, OSError):
            pass
    return {"snapshots": []}


def build_snapshot_fixture(row: dict[str, Any]) -> dict[str, Any] | None:
    """Convierte fila de prode (_fixture_prode_row) a entrada JSON del snapshot."""
    fixture = row.get("fixture") or {}
    prediction = row.get("prediction") or {}
    prode = prediction.get("prode") or {}
    fair = prediction.get("fair") or {}
    probs_1x2 = fair.get("1x2") or {}
    fit = prediction.get("fit") or {}

    odds = fixture.get("odds")
    if not odds and isinstance(fixture.get("bookmakerOdds"), dict):
        odds = extract_pinnacle_odds(fixture["bookmakerOdds"])
    if not odds:
        return None

    return {
        "fixtureId": str(row.get("fixture_id") or fixture.get("fixtureId")),
        "home": get_team_display(fixture.get("participant1Id")),
        "away": get_team_display(fixture.get("participant2Id")),
        "startTime": fixture.get("startTime"),
        "prediction": prode.get("score"),
        "epv": float(prode.get("epv", 0.0)),
        "p_exact": float(prode.get("p_exact", 0.0)),
        "p_home": float(probs_1x2.get("home", 0.0)),
        "p_draw": float(probs_1x2.get("draw", 0.0)),
        "p_away": float(probs_1x2.get("away", 0.0)),
        "lambda_home": float(fit.get("lambda_home", 0.0)),
        "lambda_away": float(fit.get("mu_away", 0.0)),
        "odds_home": float(odds.get("home", 0.0)),
        "odds_draw": float(odds.get("draw", 0.0)),
        "odds_away": float(odds.get("away", 0.0)),
    }


def save_snapshot(fixtures_with_predictions: list[dict[str, Any]]) -> int:
    """
    Agrega un snapshot al JSON del día (append, no reemplaza).
    Retorna cantidad de fixtures guardados.
    """
    entries: list[dict[str, Any]] = []
    for row in fixtures_with_predictions:
        entry = build_snapshot_fixture(row)
        if entry is not None:
            entries.append(entry)

    if not entries:
        return 0

    path = snapshot_file_for_date()
    payload = _load_day_payload(path)
    snapshots: list[dict[str, Any]] = payload.setdefault("snapshots", [])

    snapshots.append(
        {
            "timestamp": datetime.now(AR_TZ).isoformat(timespec="seconds"),
            "fixtures": entries,
        }
    )

    if len(snapshots) > MAX_SNAPSHOTS_PER_DAY:
        snapshots[:] = snapshots[-MAX_SNAPSHOTS_PER_DAY:]

    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return len(entries)


def snapshot_timestamp_hhmm(timestamp: str) -> str:
    try:
        dt = datetime.fromisoformat(timestamp)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(AR_TZ).strftime("%H:%M")
    except ValueError:
        return timestamp[:5] if len(timestamp) >= 5 else timestamp


def fixture_history(
    fixture_id: str,
    snapshots: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Serie temporal de un partido a través de los snapshots del día."""
    history: list[dict[str, Any]] = []
    source = snapshots if snapshots is not None else load_snapshots_today()
    fid = str(fixture_id)
    for snap in source:
        ts = str(snap.get("timestamp", ""))
        fixtures = snap.get("fixtures") or []
        if not isinstance(fixtures, list):
            continue
        for fx in fixtures:
            if str(fx.get("fixtureId")) == fid:
                history.append({"timestamp": ts, **fx})
                break
    return history


def summarize_day_movements(
    snapshots: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Partidos cuyo score predicho cambió al menos una vez hoy."""
    source = snapshots if snapshots is not None else load_snapshots_today()
    by_fixture: dict[str, list[dict[str, Any]]] = {}

    for snap in source:
        ts = str(snap.get("timestamp", ""))
        for fx in snap.get("fixtures") or []:
            if not isinstance(fx, dict):
                continue
            fid = str(fx.get("fixtureId", ""))
            if not fid:
                continue
            by_fixture.setdefault(fid, []).append({"timestamp": ts, **fx})

    movements: list[dict[str, Any]] = []
    for fid, entries in by_fixture.items():
        predictions = [str(e.get("prediction", "")) for e in entries]
        if len(set(predictions)) < 2:
            continue

        timeline_parts: list[str] = []
        prev_pred: str | None = None
        for entry in entries:
            pred = str(entry.get("prediction", ""))
            if pred != prev_pred:
                timeline_parts.append(f"{snapshot_timestamp_hhmm(entry['timestamp'])}: {pred}")
                prev_pred = pred

        p_home_first = float(entries[0].get("p_home", 0.0))
        p_home_last = float(entries[-1].get("p_home", 0.0))
        delta_pp = (p_home_last - p_home_first) * 100.0
        sign = "↑" if delta_pp >= 0 else "↓"

        movements.append(
            {
                "fixture_id": fid,
                "label": f"{entries[0].get('home', '?')} vs {entries[0].get('away', '?')}",
                "timeline": " → ".join(timeline_parts),
                "p_home_first_pct": p_home_first * 100.0,
                "p_home_last_pct": p_home_last * 100.0,
                "p_home_delta_pp": delta_pp,
                "p_home_delta_label": f"{sign}{abs(delta_pp):.0f}%",
            }
        )

    movements.sort(key=lambda item: abs(item["p_home_delta_pp"]), reverse=True)
    return movements
