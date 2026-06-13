from __future__ import annotations

import os
import re
import unicodedata
from typing import Any

import pandas as pd
import requests

BASE_URL = "https://api.the-odds-api.com/v4"
WORLDCUP_SPORT_KEY = "soccer_fifa_world_cup"
DEFAULT_TOTALS_LINE = 2.5


class OddsAPIError(Exception):
    pass


def get_odds_api_key() -> str:
    """Lee ODDS_API_KEY desde st.secrets o variable de entorno."""
    try:
        import streamlit as st

        secrets = getattr(st, "secrets", {})
        key = secrets.get("ODDS_API_KEY") or os.getenv("ODDS_API_KEY", "")
        if key:
            return str(key).strip()
    except Exception:
        pass

    return os.getenv("ODDS_API_KEY", "").strip()


def _normalize_name(name: str) -> str:
    """Normaliza para emparejar equipos entre APIs."""
    text = str(name).strip()
    if "  " in text:
        text = text.split("  ", 1)[-1].strip()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


_NAME_ALIASES: dict[str, str] = {
    "south korea": "corea del sur",
    "korea republic": "corea del sur",
    "usa": "estados unidos",
    "united states": "estados unidos",
    "ivory coast": "costa de marfil",
    "cote divoire": "costa de marfil",
    "czech republic": "chequia",
    "czechia": "chequia",
    "republic of ireland": "irlanda",
    "dr congo": "rd congo",
    "congo dr": "rd congo",
    "democratic republic of congo": "rd congo",
    "saudi arabia": "arabia saudita",
    "iran": "iran",
    "qatar": "qatar",
    "netherlands": "paises bajos",
    "germany": "alemania",
    "spain": "espana",
    "france": "francia",
    "england": "inglaterra",
    "mexico": "mexico",
    "brazil": "brasil",
    "argentina": "argentina",
    "portugal": "portugal",
    "belgium": "belgica",
    "croatia": "croacia",
    "morocco": "marruecos",
    "senegal": "senegal",
    "japan": "japon",
    "australia": "australia",
    "canada": "canada",
    "switzerland": "suiza",
    "scotland": "escocia",
    "uruguay": "uruguay",
    "colombia": "colombia",
    "ecuador": "ecuador",
    "paraguay": "paraguay",
    "tunisia": "tunez",
    "algeria": "argelia",
    "austria": "austria",
    "norway": "noruega",
    "egypt": "egipto",
    "ghana": "ghana",
    "panama": "panama",
    "haiti": "haiti",
    "jordan": "jordania",
    "new zealand": "nueva zelanda",
    "uzbekistan": "uzbekistan",
    "curacao": "curazao",
    "cape verde": "cabo verde",
    "iraq": "irak",
    "romania": "rumania",
    "ukraine": "ucrania",
    "turkey": "turquia",
    "bosnia and herzegovina": "bosnia y herzegovina",
}


def _canonical(name: str) -> str:
    n = _normalize_name(name)
    return _NAME_ALIASES.get(n, n)


def _parse_usage_headers(response: requests.Response) -> dict[str, int | None]:
    def _to_int(value: str | None) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    return {
        "remaining": _to_int(response.headers.get("x-requests-remaining")),
        "used": _to_int(response.headers.get("x-requests-used")),
    }


def _ensure_api_key(api_key: str) -> str:
    key = (api_key or "").strip()
    if not key:
        raise OddsAPIError(
            "API Key no configurada. Define ODDS_API_KEY en st.secrets o variables de entorno."
        )
    return key


def _handle_response(response: requests.Response) -> Any:
    if response.status_code == 401:
        raise OddsAPIError("API Key inválida (401).")
    if response.status_code == 429:
        raise OddsAPIError("Límite de requests alcanzado (429).")
    if not response.ok:
        raise OddsAPIError(f"Error HTTP {response.status_code}: {response.text[:300]}")
    return response.json()


def _odds_url(sport_key: str) -> str:
    return f"{BASE_URL}/sports/{sport_key}/odds"


def _request_odds(
    sport_key: str,
    api_key: str,
    *,
    markets: str = "h2h,totals",
    bookmakers: str | None = "pinnacle",
    event_ids: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int | None]]:
    params: dict[str, str] = {
        "apiKey": _ensure_api_key(api_key),
        "regions": "eu",
        "markets": markets,
        "oddsFormat": "decimal",
    }
    if bookmakers:
        params["bookmakers"] = bookmakers
    if event_ids:
        params["eventIds"] = event_ids

    response = requests.get(_odds_url(sport_key), params=params, timeout=30)
    usage = _parse_usage_headers(response)
    payload = _handle_response(response)

    if not isinstance(payload, list):
        raise OddsAPIError("Respuesta inesperada de The Odds API.")
    return payload, usage


def fetch_sports(api_key: str) -> tuple[list[dict[str, Any]], dict[str, int | None]]:
    """Lista deportes disponibles (GET /v4/sports)."""
    response = requests.get(
        f"{BASE_URL}/sports",
        params={"apiKey": _ensure_api_key(api_key)},
        timeout=30,
    )
    usage = _parse_usage_headers(response)
    payload = _handle_response(response)
    if not isinstance(payload, list):
        raise OddsAPIError("Respuesta inesperada al listar deportes.")
    return payload, usage


def find_world_cup_sport_key(api_key: str) -> str:
    """Busca en /sports la key del Mundial (contiene 'world_cup' en key o title)."""
    sports, _ = fetch_sports(api_key)
    for sport in sports:
        key = str(sport.get("key", ""))
        title = str(sport.get("title", ""))
        if "world_cup" in key.lower() or "world cup" in title.lower():
            print(
                f"[The Odds API] Deporte Mundial encontrado: "
                f"key={key!r}, title={title!r}"
            )
            return key
    raise OddsAPIError(
        "No se encontró un deporte Mundial activo. "
        "Verifica que el torneo esté disponible en The Odds API."
    )


def resolve_worldcup_sport_key(api_key: str) -> str:
    """
    Resuelve la sport key del Mundial.

    Usa soccer_fifa_world_cup por defecto; si no responde, busca en /sports.
    """
    key = _ensure_api_key(api_key)
    probe = requests.get(
        _odds_url(WORLDCUP_SPORT_KEY),
        params={
            "apiKey": key,
            "regions": "eu",
            "markets": "h2h",
            "oddsFormat": "decimal",
        },
        timeout=30,
    )
    if probe.status_code in (200, 204):
        return WORLDCUP_SPORT_KEY
    if probe.status_code == 404:
        return find_world_cup_sport_key(api_key)
    _handle_response(probe)
    return WORLDCUP_SPORT_KEY


def parse_event_summary(event: dict[str, Any]) -> dict[str, Any] | None:
    """Extrae id, equipos y hora de un evento (sin validar odds)."""
    event_id = str(event.get("id", "")).strip()
    home_team = str(event.get("home_team", "")).strip()
    away_team = str(event.get("away_team", "")).strip()
    commence_time = str(event.get("commence_time", "")).strip()
    if not event_id or not home_team or not away_team:
        return None
    return {
        "id": event_id,
        "home_team": home_team,
        "away_team": away_team,
        "commence_time": commence_time,
    }


def fetch_worldcup_fixture_list(
    api_key: str,
    *,
    sport_key: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int | None], str]:
    """
    Lista partidos del Mundial (solo metadata, markets=h2h para minimizar costo).

    Returns: (eventos, usage, sport_key_usada)
    """
    resolved_key = sport_key or resolve_worldcup_sport_key(api_key)
    events, usage = _request_odds(
        resolved_key,
        api_key,
        markets="h2h",
        bookmakers=None,
    )
    summaries = [s for s in (parse_event_summary(ev) for ev in events) if s is not None]
    summaries.sort(key=lambda ev: ev.get("commence_time", ""))
    return summaries, usage, resolved_key


def parse_pinnacle_event(event: dict[str, Any]) -> dict[str, Any] | None:
    """
    Extrae odds Pinnacle (1X2 + O/U 2.5) de un evento de The Odds API.

    Devuelve None si faltan mercados requeridos.
    """
    home_team = str(event.get("home_team", "")).strip()
    away_team = str(event.get("away_team", "")).strip()
    commence_time = str(event.get("commence_time", "")).strip()
    event_id = str(event.get("id", "")).strip()
    if not home_team or not away_team:
        return None

    pinnacle = None
    for bookmaker in event.get("bookmakers", []):
        if bookmaker.get("key") == "pinnacle":
            pinnacle = bookmaker
            break
    if pinnacle is None:
        return None

    h2h: dict[str, float] = {}
    over: float | None = None
    under: float | None = None

    for market in pinnacle.get("markets", []):
        key = market.get("key")
        if key == "h2h":
            for out in market.get("outcomes", []):
                name = str(out.get("name", ""))
                price = float(out.get("price", 0))
                if name == home_team:
                    h2h["home"] = price
                elif name == away_team:
                    h2h["away"] = price
                elif name.lower() == "draw":
                    h2h["draw"] = price
        elif key == "totals":
            for out in market.get("outcomes", []):
                point = float(out.get("point", 0))
                if abs(point - DEFAULT_TOTALS_LINE) >= 0.01:
                    continue
                label = str(out.get("name", "")).lower()
                price = float(out.get("price", 0))
                if label == "over":
                    over = price
                elif label == "under":
                    under = price

    if len(h2h) != 3 or over is None or under is None:
        return None
    if not all(v > 1.0 for v in (*h2h.values(), over, under)):
        return None

    return {
        "id": event_id,
        "home_team": home_team,
        "away_team": away_team,
        "commence_time": commence_time,
        "home": h2h["home"],
        "draw": h2h["draw"],
        "away": h2h["away"],
        "over": over,
        "under": under,
        "line": DEFAULT_TOTALS_LINE,
    }


def fetch_worldcup_event_pinnacle_odds(
    api_key: str,
    sport_key: str,
    event_id: str,
) -> tuple[dict[str, Any] | None, dict[str, int | None]]:
    """Odds Pinnacle completas (1X2 + O/U 2.5) para un partido del Mundial."""
    events, usage = _request_odds(
        sport_key,
        api_key,
        markets="h2h,totals",
        bookmakers="pinnacle",
        event_ids=event_id,
    )
    if not events:
        return None, usage
    parsed = parse_pinnacle_event(events[0])
    return parsed, usage


def fetch_worldcup_odds(api_key: str) -> list[dict[str, Any]]:
    """Descarga odds FIFA World Cup (retrocompatibilidad con apply_odds_api_to_dataframe)."""
    events, _ = _request_odds(WORLDCUP_SPORT_KEY, api_key, bookmakers=None)
    return events


def _parse_event_odds(event: dict[str, Any]) -> dict[str, float] | None:
    home_team = event.get("home_team", "")
    away_team = event.get("away_team", "")
    h2h: dict[str, float] = {}
    totals: dict[str, Any] = {}

    for bookmaker in event.get("bookmakers", []):
        for market in bookmaker.get("markets", []):
            key = market.get("key")
            if key == "h2h":
                for out in market.get("outcomes", []):
                    name = out.get("name", "")
                    price = float(out.get("price", 0))
                    if name == home_team:
                        h2h["home"] = price
                    elif name == away_team:
                        h2h["away"] = price
                    elif name.lower() == "draw":
                        h2h["draw"] = price
            elif key == "totals":
                for out in market.get("outcomes", []):
                    point = float(out.get("point", 0))
                    if abs(point - DEFAULT_TOTALS_LINE) < 0.01:
                        label = out.get("name", "").lower()
                        price = float(out.get("price", 0))
                        if label == "over":
                            totals["over"] = price
                        elif label == "under":
                            totals["under"] = price
                        totals["line"] = point

        if len(h2h) == 3 and "over" in totals and "under" in totals:
            break

    if len(h2h) < 3:
        return None

    return {
        "home_team": home_team,
        "away_team": away_team,
        "home": h2h["home"],
        "draw": h2h.get("draw", h2h["home"]),
        "away": h2h["away"],
        "over": totals.get("over", float("nan")),
        "under": totals.get("under", float("nan")),
        "line": totals.get("line", DEFAULT_TOTALS_LINE),
    }


def apply_odds_api_to_dataframe(
    df: pd.DataFrame,
    api_key: str,
    *,
    strip_team_fn: Any,
) -> tuple[pd.DataFrame, int, int]:
    """Actualiza cuotas del DataFrame emparejando equipos con The-Odds-API."""
    events = fetch_worldcup_odds(api_key)
    parsed = [_parse_event_odds(ev) for ev in events]
    parsed = [p for p in parsed if p is not None]

    lookup: dict[tuple[str, str], dict[str, float]] = {}
    for p in parsed:
        key = (_canonical(p["home_team"]), _canonical(p["away_team"]))
        lookup[key] = p
        lookup.setdefault(
            (_canonical(p["away_team"]), _canonical(p["home_team"])),
            p,
        )

    out = df.copy()
    matched = 0

    for idx, row in out.iterrows():
        home = _canonical(strip_team_fn(str(row.get("Equipo Local", ""))))
        away = _canonical(strip_team_fn(str(row.get("Equipo Visitante", ""))))
        odds = lookup.get((home, away))
        if not odds:
            continue

        out.at[idx, "Cuota 1"] = odds["home"]
        out.at[idx, "Cuota X"] = odds["draw"]
        out.at[idx, "Cuota 2"] = odds["away"]
        if odds.get("over") and odds["over"] == odds["over"]:
            out.at[idx, "Cuota Over"] = odds["over"]
        if odds.get("under") and odds["under"] == odds["under"]:
            out.at[idx, "Cuota Under"] = odds["under"]
        out.at[idx, "Línea O/U"] = odds.get("line", DEFAULT_TOTALS_LINE)
        matched += 1

    return out, matched, len(parsed)
