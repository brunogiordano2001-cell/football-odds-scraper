from __future__ import annotations

import os
from datetime import datetime
from typing import Any
from urllib.parse import urlencode

import requests
from requests.auth import HTTPBasicAuth

BASE_URL = "https://api.pinnacle.com"
SOCCER_SPORT_ID = 29
FULL_MATCH_PERIOD = 0
DEFAULT_OU_LINE = 2.5

# Endpoints oficiales (Lines API — https://pinnacleapi.github.io)
LEAGUES_ENDPOINT = "/v2/leagues"
FIXTURES_ENDPOINT = "/v1/fixtures"
ODDS_ENDPOINT = "/v2/odds"


class PinnacleAPIError(Exception):
    """Error de la API de Pinnacle."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def get_pinnacle_credentials() -> tuple[str, str]:
    """
    Lee credenciales desde st.secrets o variables de entorno.

    Orden: PINNACLE_USER / PINNACLE_PASSWORD en secrets → env vars.
    """
    try:
        import streamlit as st

        secrets = getattr(st, "secrets", {})
        user = secrets.get("PINNACLE_USER") or os.getenv("PINNACLE_USER", "")
        password = secrets.get("PINNACLE_PASSWORD") or os.getenv("PINNACLE_PASSWORD", "")
        if user and password:
            return str(user), str(password)
    except Exception:
        pass

    user = os.getenv("PINNACLE_USER", "")
    password = os.getenv("PINNACLE_PASSWORD", "")
    if not user or not password:
        raise PinnacleAPIError(
            "Credenciales Pinnacle no configuradas. "
            "Define PINNACLE_USER y PINNACLE_PASSWORD en st.secrets o variables de entorno."
        )
    return user, password


def _build_request_url(endpoint: str, params: dict[str, Any] | None = None) -> str:
    url = f"{BASE_URL}{endpoint}"
    if params:
        url = f"{url}?{urlencode(params, doseq=True)}"
    return url


def _log_request_url(url: str) -> None:
    print(f"[Pinnacle API] GET {url}")
    try:
        import streamlit as st

        st.write(f"Pinnacle request: `{url}`")
    except Exception:
        pass


def _request(
    endpoint: str,
    *,
    user: str,
    password: str,
    params: dict[str, Any] | None = None,
    timeout: int = 30,
) -> Any:
    url = _build_request_url(endpoint, params)
    _log_request_url(url)
    response = requests.get(
        url,
        auth=HTTPBasicAuth(user, password),
        headers={"Accept": "application/json"},
        timeout=timeout,
    )

    if response.status_code == 401:
        raise PinnacleAPIError("401 — Credenciales Pinnacle inválidas.", status_code=401)
    if response.status_code == 403:
        raise PinnacleAPIError(
            "403 — Cuenta sin acceso a la API de Pinnacle.",
            status_code=403,
        )
    if response.status_code == 429:
        raise PinnacleAPIError(
            "429 — Límite de peticiones excedido (1 req / 2 min por endpoint).",
            status_code=429,
        )
    if not response.ok:
        raise PinnacleAPIError(
            f"Error HTTP {response.status_code}: {response.text[:300]}",
            status_code=response.status_code,
        )

    if not response.text.strip():
        return {}
    return response.json()


def fetch_soccer_leagues(user: str, password: str) -> list[dict[str, Any]]:
    """Ligas de fútbol activas con ofertas (hasOfferings=true)."""
    data = _request(
        LEAGUES_ENDPOINT,
        user=user,
        password=password,
        params={"sportId": SOCCER_SPORT_ID},
    )
    leagues = data.get("leagues", data) if isinstance(data, dict) else data
    if not isinstance(leagues, list):
        return []

    active = [
        lg
        for lg in leagues
        if lg.get("hasOfferings") is True or lg.get("hasOfferings") == "true"
    ]
    active.sort(key=lambda x: str(x.get("name", "")))
    return active


def fetch_league_fixtures(
    user: str,
    password: str,
    league_id: int,
) -> list[dict[str, Any]]:
    """Partidos disponibles de una liga."""
    data = _request(
        FIXTURES_ENDPOINT,
        user=user,
        password=password,
        params={"sportId": SOCCER_SPORT_ID, "leagueIds": league_id},
    )

    events: list[dict[str, Any]] = []
    leagues = data.get("league", [])
    if not isinstance(leagues, list):
        leagues = [leagues] if leagues else []

    for league in leagues:
        for ev in league.get("events", []):
            events.append(
                {
                    "id": ev.get("id"),
                    "home": ev.get("home", ""),
                    "away": ev.get("away", ""),
                    "starts": ev.get("starts", ""),
                    "status": ev.get("status", ""),
                    "league_id": league.get("id", league_id),
                    "league_name": league.get("name", ""),
                }
            )

    events.sort(key=lambda e: str(e.get("starts", "")))
    return events


def fetch_league_odds(
    user: str,
    password: str,
    league_id: int,
    *,
    event_id: int | None = None,
) -> dict[str, Any]:
    """Odds decimales de eventos de una liga (opcionalmente filtrados por eventId)."""
    params: dict[str, Any] = {
        "sportId": SOCCER_SPORT_ID,
        "leagueIds": league_id,
        "oddsFormat": "Decimal",
    }
    if event_id is not None:
        params["eventIds"] = event_id
    return _request(
        ODDS_ENDPOINT,
        user=user,
        password=password,
        params=params,
    )


def format_fixture_label(fixture: dict[str, Any]) -> str:
    home = fixture.get("home", "?")
    away = fixture.get("away", "?")
    starts = fixture.get("starts", "")
    date_str = ""
    if starts:
        try:
            dt = datetime.fromisoformat(starts.replace("Z", "+00:00"))
            date_str = dt.strftime("%d/%m/%Y %H:%M UTC")
        except ValueError:
            date_str = starts
    return f"{home} vs {away} — {date_str}"


def extract_event_odds(
    odds_payload: dict[str, Any],
    event_id: int,
    *,
    ou_line: float = DEFAULT_OU_LINE,
) -> dict[str, float]:
    """
    Extrae 1X2 (moneyline) y Over/Under para un evento.

    Returns
    -------
    dict con keys: home, draw, away, over, under, line
    """
    event_id = int(event_id)
    leagues = odds_payload.get("leagues", [])

    for league in leagues:
        for event in league.get("events", []):
            if int(event.get("id", -1)) != event_id:
                continue

            period = _find_full_match_period(event.get("periods", []))
            if period is None:
                raise PinnacleAPIError(
                    f"Sin periodo de partido completo para evento {event_id}."
                )

            moneyline = period.get("moneyline") or {}
            home = _to_float(moneyline.get("home"))
            away = _to_float(moneyline.get("away"))
            draw = _to_float(moneyline.get("draw"))

            if not all(v and v > 1.0 for v in (home, draw, away)):
                raise PinnacleAPIError(
                    f"Moneyline incompleto para evento {event_id}."
                )

            over, under, line = _extract_totals(period.get("totals", []), ou_line)

            result: dict[str, float] = {
                "home": home,
                "draw": draw,
                "away": away,
                "line": line,
            }
            if over and under:
                result["over"] = over
                result["under"] = under
            return result

    raise PinnacleAPIError(f"No se encontraron odds para el evento {event_id}.")


def _find_full_match_period(periods: list[dict[str, Any]]) -> dict[str, Any] | None:
    for p in periods:
        if int(p.get("number", -1)) == FULL_MATCH_PERIOD:
            return p
    return periods[0] if periods else None


def _extract_totals(
    totals: list[dict[str, Any]],
    target_line: float,
) -> tuple[float | None, float | None, float]:
    if not totals:
        return None, None, target_line

    best = None
    best_diff = float("inf")
    for t in totals:
        pts = _to_float(t.get("points"))
        if pts is None:
            continue
        diff = abs(pts - target_line)
        if diff < best_diff:
            best_diff = diff
            best = t

    if best is None:
        return None, None, target_line

    line = float(best.get("points", target_line))
    over = _to_float(best.get("over"))
    under = _to_float(best.get("under"))
    return over, under, line


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
