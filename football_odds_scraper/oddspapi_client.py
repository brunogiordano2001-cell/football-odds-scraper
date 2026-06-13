from __future__ import annotations

import os
from collections.abc import Callable
from datetime import date, datetime, timedelta, timezone
from typing import Any

import requests

from football_odds_scraper.world_cup_teams import WORLD_CUP_TEAMS, get_team_display

BASE_URL = "https://api.oddspapi.io/v4"
SOCCER_SPORT_ID = 10
BOOKMAKER_PINNACLE = "pinnacle"
FOOTBALL_OU_LINE_MIN = 0.5
FOOTBALL_OU_LINE_MAX = 5.5
FIXTURE_WINDOW_DAYS = 30
ODDS_BY_TOURNAMENTS_EMPTY_MSG = (
    "No hay partidos próximos disponibles en este momento. "
    "El fixture se actualizará cuando Pinnacle publique "
    "las odds de los próximos partidos."
)
ODDS_FORMAT_DECIMAL = "decimal"
DEFAULT_TOTALS_LINE = 2.5
AR_TZ = timezone(timedelta(hours=-3))

_api_request_callback: Callable[[], None] | None = None


def set_api_request_callback(callback: Callable[[], None] | None) -> None:
    """Registra callback invocado en cada GET real a api.oddspapi.io."""
    global _api_request_callback
    _api_request_callback = callback


def _api_get(url: str, **kwargs: Any) -> requests.Response:
    response = requests.get(url, **kwargs)
    if url.startswith(BASE_URL) and _api_request_callback is not None:
        _api_request_callback()
    return response


class OddsPapiError(Exception):
    pass


class FixtureLiveError(Exception):
    """El fixture ya está en curso y las odds en vivo no están disponibles."""


def get_oddspapi_key() -> str:
    """Lee ODDSPAPI_KEY desde st.secrets o variable de entorno."""
    try:
        import streamlit as st

        secrets = getattr(st, "secrets", {})
        key = secrets.get("ODDSPAPI_KEY") or os.getenv("ODDSPAPI_KEY", "")
        if key:
            return str(key).strip()
    except Exception:
        pass

    return os.getenv("ODDSPAPI_KEY", "").strip()


def _ensure_api_key(api_key: str) -> str:
    key = (api_key or "").strip()
    if not key:
        raise OddsPapiError(
            "API Key no configurada. Define ODDSPAPI_KEY en st.secrets o variables de entorno."
        )
    return key


def _parse_error_payload(response: requests.Response) -> dict[str, Any]:
    try:
        data = response.json()
        return data if isinstance(data, dict) else {}
    except ValueError:
        return {}


def _log_raw_response(response: requests.Response, *, label: str = "RAW RESPONSE") -> None:
    print(f"=== {label} ===")
    print("Status:", response.status_code)
    print("URL llamada:", response.url)
    print("Body:", response.text[:2000])
    print("=== FIN RAW ===")


def _handle_response(response: requests.Response) -> Any:
    if response.status_code == 401:
        raise OddsPapiError("API Key inválida (401).")
    if response.status_code == 429:
        raise OddsPapiError("Límite de requests alcanzado (429).")
    if response.status_code == 403:
        err = _parse_error_payload(response)
        if err.get("code") == "RESTRICTED_ACCESS":
            raise FixtureLiveError(
                "Partido en curso — odds en vivo no disponibles en el plan gratuito."
            )
        raise OddsPapiError(f"Error HTTP 403: {response.text[:300]}")
    if not response.ok:
        raise OddsPapiError(f"Error HTTP {response.status_code}: {response.text[:300]}")
    if not response.text.strip():
        return []
    return response.json()


def _fixture_date_window(
    *,
    now: datetime | None = None,
) -> tuple[str, str]:
    """Ventana UTC para endpoints que acepten from/to (YYYY-MM-DD)."""
    current = now or datetime.now(timezone.utc)
    date_from = current.strftime("%Y-%m-%d")
    date_to = (current + timedelta(days=FIXTURE_WINDOW_DAYS)).strftime("%Y-%m-%d")
    return date_from, date_to


def _is_world_cup_tournament(tournament_name: Any) -> bool:
    return "world cup" in str(tournament_name or "").lower()


def _validate_fixture_id(fixture_id: str) -> str:
    fixture_id = str(fixture_id).strip()
    assert fixture_id.startswith("id"), f"fixtureId inválido: {fixture_id}"
    return fixture_id


def _normalize_list_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("tournaments", "fixtures", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def _tournament_has_world_category(tournament: dict[str, Any]) -> bool:
    category_name = str(tournament.get("categoryName", "")).lower()
    category_slug = str(tournament.get("categorySlug", "")).lower()
    return (
        "world" in category_name
        or "world" in category_slug
        or "international" in category_slug
    )


def _is_world_cup_tournament_candidate(tournament: dict[str, Any]) -> bool:
    name = str(tournament.get("tournamentName", "")).lower()
    if "world cup" not in name:
        return False
    return _tournament_has_world_category(tournament)


def _tournament_priority(tournament: dict[str, Any]) -> tuple[int, int, int, int]:
    live = int(tournament.get("liveFixtures") or 0)
    upcoming = int(tournament.get("upcomingFixtures") or 0)
    has_fixtures = 1 if (live > 0 or upcoming > 0) else 0
    return (has_fixtures, live + upcoming, live, upcoming)


def _log_all_tournaments(tournaments: list[dict[str, Any]], *, reason: str) -> None:
    import json

    print(f"=== TORNEOS COMPLETOS ({reason}) ===")
    print(json.dumps(tournaments, indent=2, default=str))
    print("=== FIN TORNEOS COMPLETOS ===")


def get_world_cup_tournament_id(api_key: str) -> str:
    """Resuelve el tournamentId del Mundial vía GET /tournaments."""
    response = _api_get(
        f"{BASE_URL}/tournaments",
        params={
            "apiKey": _ensure_api_key(api_key),
            "sportId": SOCCER_SPORT_ID,
        },
        timeout=30,
    )
    payload = _handle_response(response)
    tournaments = _normalize_list_payload(payload)

    print("=== TORNEOS World/Cup ===")
    for tournament in tournaments:
        name = str(tournament.get("tournamentName", ""))
        lower_name = name.lower()
        if "world" in lower_name or "cup" in lower_name:
            print(
                f"  id={tournament.get('tournamentId')} | name={name} | "
                f"category={tournament.get('categoryName')} | "
                f"slug={tournament.get('categorySlug')} | "
                f"live={tournament.get('liveFixtures')} | "
                f"upcoming={tournament.get('upcomingFixtures')}"
            )
    print("=== FIN TORNEOS ===")

    candidates = [t for t in tournaments if _is_world_cup_tournament_candidate(t)]
    if not candidates:
        candidates = [
            t
            for t in tournaments
            if "world cup" in str(t.get("tournamentName", "")).lower()
        ]

    if not candidates:
        _log_all_tournaments(tournaments, reason="sin candidatos World Cup")
        raise OddsPapiError("No se encontró el torneo del Mundial en OddsPapi.")

    best = max(candidates, key=_tournament_priority)
    raw_id = best.get("tournamentId")
    tournament_id = str(raw_id).strip() if raw_id is not None else ""
    if not tournament_id or tournament_id.lower() == "none":
        _log_all_tournaments(tournaments, reason="tournamentId inválido en candidato")
        raise OddsPapiError("Torneo del Mundial sin tournamentId válido.")

    print(
        f"=== Mundial tournamentId seleccionado: {tournament_id} "
        f"({best.get('tournamentName')}) ==="
    )
    return tournament_id


def _extract_fixtures_from_odds_by_tournaments(payload: Any) -> list[dict[str, Any]]:
    fixtures: list[dict[str, Any]] = []
    for item in _normalize_list_payload(payload):
        nested = item.get("fixtures")
        if isinstance(nested, list):
            fixtures.extend(f for f in nested if isinstance(f, dict))
        elif item.get("fixtureId") is not None:
            fixtures.append(item)
    return fixtures


def fetch_worldcup_fixtures_with_odds(
    api_key: str,
    *,
    tournament_id: str | None = None,
) -> tuple[list[dict[str, Any]], str, str | None]:
    """
    Carga fixtures del Mundial + odds Pinnacle (GET /odds-by-tournaments).
    Nombres de equipos vía WORLD_CUP_TEAMS (sin llamadas extra a la API).

    Retorna (fixtures, tournament_id, mensaje_informativo).
    mensaje_informativo se setea cuando la API responde 404 sin fixtures.
    """
    tid = tournament_id or get_world_cup_tournament_id(api_key)
    if not tid:
        raise OddsPapiError(
            "get_world_cup_tournament_id() retornó vacío — revisar log de torneos completos."
        )

    # /odds-by-tournaments: tournamentIds + bookmaker bastan; sin filtro from/to.
    params = {
        "apiKey": _ensure_api_key(api_key),
        "tournamentIds": str(tid),
        "bookmaker": BOOKMAKER_PINNACLE,
        "oddsFormat": ODDS_FORMAT_DECIMAL,
    }
    print("=== odds-by-tournaments params ===", params)

    response = _api_get(
        f"{BASE_URL}/odds-by-tournaments",
        params=params,
        timeout=30,
    )
    print("=== URL odds-by-tournaments ===", response.url)

    if response.status_code == 404:
        err = _parse_error_payload(response)
        detail = err.get("message") or response.text[:200]
        print(f"=== odds-by-tournaments 404 (sin fixtures): {detail} ===")
        return [], tid, ODDS_BY_TOURNAMENTS_EMPTY_MSG

    _log_raw_response(response, label="RAW ODDS-BY-TOURNAMENTS RESPONSE")
    payload = _handle_response(response)

    raw_fixtures = _extract_fixtures_from_odds_by_tournaments(payload)
    with_odds_flag = [
        item
        for item in raw_fixtures
        if item.get("hasOdds") is True or str(item.get("hasOdds")).lower() == "true"
    ]

    parsed = [
        fixture
        for item in with_odds_flag
        if (fixture := _parse_worldcup_fixture_with_odds(item)) is not None
    ]
    parsed.sort(key=lambda f: str(f.get("startTime", "")))

    print(
        f"=== odds-by-tournaments: tournamentId={tid} | "
        f"hasOdds=True: {len(with_odds_flag)} | parseados: {len(parsed)} ==="
    )
    return parsed, tid, None


def _parse_worldcup_fixture_with_odds(item: dict[str, Any]) -> dict[str, Any] | None:
    fixture_id = item.get("fixtureId")
    if fixture_id is None:
        return None

    fixture_id_str = str(fixture_id)
    bookmaker_odds = item.get("bookmakerOdds")
    if not isinstance(bookmaker_odds, dict):
        bookmaker_odds = {}

    parsed_odds = extract_pinnacle_odds(bookmaker_odds) if bookmaker_odds else None

    return {
        "fixtureId": fixture_id_str,
        "participant1Id": item.get("participant1Id"),
        "participant2Id": item.get("participant2Id"),
        "startTime": item.get("startTime"),
        "hasOdds": item.get("hasOdds"),
        "tournamentId": item.get("tournamentId"),
        "tournamentName": item.get("tournamentName"),
        "bookmakerOdds": bookmaker_odds,
        "odds": parsed_odds,
    }


def format_fixture_match_label(
    fixture: dict[str, Any],
    *,
    live: bool = False,
) -> str:
    home = get_team_display(fixture.get("participant1Id"))
    away = get_team_display(fixture.get("participant2Id"))
    schedule = format_fixture_start_time(fixture.get("startTime"))
    live_tag = " 🔴 EN VIVO" if live else ""
    return f"{home} vs {away} — {schedule}{live_tag}"


def fetch_worldcup_fixtures(
    api_key: str,
    *,
    today: date | None = None,
) -> list[dict[str, Any]]:
    """Retrocompatibilidad: devuelve fixtures con odds del flujo optimizado."""
    fixtures, _, _ = fetch_worldcup_fixtures_with_odds(api_key)
    return fixtures


def _to_valid_price(value: Any) -> float | None:
    try:
        price = float(value)
        return price if price > 1.0 else None
    except (TypeError, ValueError):
        return None


def _player_price(outcome_data: dict[str, Any]) -> float | None:
    try:
        return _to_valid_price(outcome_data["players"]["0"]["price"])
    except (KeyError, TypeError):
        return None


def _outcome_label(outcome_data: dict[str, Any]) -> str:
    player = outcome_data.get("players", {}).get("0", {})
    if not isinstance(player, dict):
        return ""
    for key in ("outcomeName", "bookmakerOutcomeId", "name"):
        value = player.get(key) or outcome_data.get(key)
        if value:
            return str(value)
    return ""


def _extract_1x2(
    markets: dict[str, Any],
) -> tuple[float | None, float | None, float | None]:
    """1X2: busca outcomes con bookmakerOutcomeId home/draw/away."""
    home: float | None = None
    draw: float | None = None
    away: float | None = None

    for market_data in markets.values():
        if not isinstance(market_data, dict):
            continue
        outcomes = market_data.get("outcomes", {})
        if not isinstance(outcomes, dict):
            continue
        for outcome_data in outcomes.values():
            if not isinstance(outcome_data, dict):
                continue
            label = _bookmaker_outcome_label(outcome_data).lower()
            price = _player_price(outcome_data)
            if price is None:
                continue
            if label == "home":
                home = price
            elif label == "draw":
                draw = price
            elif label == "away":
                away = price
        if home is not None and draw is not None and away is not None:
            break

    return home, draw, away


def _extract_totals_2_5(
    markets: dict[str, Any],
) -> tuple[float | None, float | None]:
    """O/U 2.5: busca outcomes con bookmakerOutcomeId 2.5/over y 2.5/under."""
    over: float | None = None
    under: float | None = None

    for market_data in markets.values():
        if not isinstance(market_data, dict):
            continue
        outcomes = market_data.get("outcomes", {})
        if not isinstance(outcomes, dict):
            continue
        for outcome_data in outcomes.values():
            if not isinstance(outcome_data, dict):
                continue
            label = _bookmaker_outcome_label(outcome_data)
            price = _player_price(outcome_data)
            if price is None:
                continue
            if label == "2.5/over":
                over = price
            elif label == "2.5/under":
                under = price
        if over is not None and under is not None:
            break

    return over, under


def _totals_line_balance(over: float, under: float) -> float:
    margin = 1.0 / over + 1.0 / under
    p_over = 1.0 / over / margin
    return abs(p_over - 0.5)


def _filter_football_ou_curve(
    curve: list[tuple[float, float, float]] | None,
) -> list[tuple[float, float, float]] | None:
    """Excluye líneas O/U fuera del rango realista de fútbol (p. ej. basketball 8+)."""
    if not curve:
        return None
    filtered = [
        (p, o, u)
        for p, o, u in curve
        if FOOTBALL_OU_LINE_MIN <= float(p) <= FOOTBALL_OU_LINE_MAX
    ]
    return filtered if filtered else None


def _main_totals_from_ou_curve(
    ou_curve: list[tuple[float, float, float]],
) -> tuple[float, float, float]:
    """Línea principal = la más cercana a over/under 50/50 en la curva filtrada."""
    line, over, under = min(
        ou_curve,
        key=lambda item: _totals_line_balance(item[1], item[2]),
    )
    return line, over, under


def _resolve_main_totals_and_curve(
    markets: dict[str, Any],
) -> tuple[float | None, float | None, float | None, list[tuple[float, float, float]] | None]:
    """Main O/U + curva filtrada (0.5–5.5). Fallback a 2.5 si la curva queda vacía."""
    ou_curve = _filter_football_ou_curve(_extract_ou_curve(markets))

    if ou_curve:
        line, over, under = _main_totals_from_ou_curve(ou_curve)
        return line, over, under, ou_curve

    over, under = _extract_totals_2_5(markets)
    if over is not None and under is not None:
        return 2.5, over, under, None
    return None, None, None, None


def _extract_main_totals(
    markets: dict[str, Any],
) -> tuple[float | None, float | None, float | None]:
    """Compat: delega en curva filtrada o fallback 2.5."""
    line, over, under, _ = _resolve_main_totals_and_curve(markets)
    return line, over, under


def _parse_totals_outcome_label(label: str) -> tuple[float, str] | None:
    """Parsea '2.5/over' → (2.5, 'over')."""
    text = label.strip().lower()
    if "/" not in text:
        return None
    line_part, side = text.split("/", 1)
    if side not in ("over", "under"):
        return None
    try:
        return float(line_part), side
    except ValueError:
        return None


def _is_full_time_totals_market(market_id: str) -> bool:
    mid = market_id.lower()
    if "totals" not in mid:
        return False
    if "/1/" in mid or "/2/" in mid:
        return False
    return True


def _extract_team_totals(
    markets: dict[str, Any],
) -> tuple[float | None, float | None, float | None, float | None, list[str]]:
    """Team totals 0.5 exactos: home/0.5/over|under y away/0.5/over|under."""
    tt_home_over: float | None = None
    tt_home_under: float | None = None
    tt_away_over: float | None = None
    tt_away_under: float | None = None
    tt_outcome_ids_found: list[str] = []

    for market_data in markets.values():
        if not isinstance(market_data, dict):
            continue
        outcomes = market_data.get("outcomes", {})
        if not isinstance(outcomes, dict):
            continue
        for outcome_data in outcomes.values():
            if not isinstance(outcome_data, dict):
                continue
            label = _bookmaker_outcome_label(outcome_data).lower()
            price = _player_price(outcome_data)
            if price is None:
                continue
            if label.startswith(("home/", "away/")) and (
                label.endswith("/over") or label.endswith("/under")
            ):
                tt_outcome_ids_found.append(label)
            if label == "home/0.5/over":
                tt_home_over = price
            elif label == "home/0.5/under":
                tt_home_under = price
            elif label == "away/0.5/over":
                tt_away_over = price
            elif label == "away/0.5/under":
                tt_away_under = price

    return (
        tt_home_over,
        tt_home_under,
        tt_away_over,
        tt_away_under,
        sorted(set(tt_outcome_ids_found)),
    )


def _extract_ou_curve(
    markets: dict[str, Any],
) -> list[tuple[float, float, float]] | None:
    """Curva O/U full-time: [(punto, over, under), ...] ordenada por punto."""
    by_line: dict[float, dict[str, float]] = {}

    for market_data in markets.values():
        if not isinstance(market_data, dict):
            continue
        market_id = str(market_data.get("bookmakerMarketId", ""))
        if not _is_full_time_totals_market(market_id):
            continue

        outcomes = market_data.get("outcomes", {})
        if not isinstance(outcomes, dict):
            continue

        for outcome_data in outcomes.values():
            if not isinstance(outcome_data, dict):
                continue
            label = _bookmaker_outcome_label(outcome_data)
            parsed = _parse_totals_outcome_label(label)
            price = _player_price(outcome_data)
            if parsed is None or price is None:
                continue
            line, side = parsed
            if not (FOOTBALL_OU_LINE_MIN <= line <= FOOTBALL_OU_LINE_MAX):
                continue
            bucket = by_line.setdefault(line, {})
            bucket[side] = price

    curve: list[tuple[float, float, float]] = []
    for line in sorted(by_line):
        sides = by_line[line]
        over = sides.get("over")
        under = sides.get("under")
        if over is not None and under is not None:
            curve.append((line, over, under))

    return curve if curve else None


def format_pinnacle_calibration(odds: dict[str, Any] | None) -> str:
    """Resumen de inputs de calibración disponibles."""
    if not odds:
        return "📊 Solo 1X2+OU"
    parts: list[str] = []
    if all(
        odds.get(k) is not None
        for k in ("tt_home_over", "tt_home_under", "tt_away_over", "tt_away_under")
    ):
        parts.append("TT")
    ou_curve = odds.get("ou_curve")
    if isinstance(ou_curve, list) and len(ou_curve) >= 3:
        parts.append(f"O/U×{len(ou_curve)}")
    if odds.get("correct_score_odds"):
        parts.append("CS")
    if odds.get("ah_line") is not None:
        parts.append("AH")
    if parts:
        return f"📊 {' + '.join(parts)}"
    return "📊 Solo 1X2+OU"


def _parse_handicap_line(label: str) -> float | None:
    """Parsea la línea desde bookmakerOutcomeId, ej. '-0.25/home' → -0.25."""
    text = label.strip()
    if "/" in text:
        text = text.split("/", 1)[0]
    text = text.replace("+", "")
    try:
        return float(text)
    except ValueError:
        return None


def _market_limit(market_data: dict[str, Any]) -> float:
    limit = market_data.get("limit")
    try:
        return float(limit) if limit is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def _extract_asian_handicap(
    markets: dict[str, Any],
) -> tuple[float | None, float | None, float | None]:
    """AH principal: market handicap con mayor limit y exactamente 2 outcomes."""
    best_limit = -1.0
    best: tuple[float, float, float] | None = None

    for market_data in markets.values():
        if not isinstance(market_data, dict):
            continue
        market_id = str(market_data.get("bookmakerMarketId", ""))
        if "handicap" not in market_id.lower():
            continue

        outcomes = market_data.get("outcomes", {})
        if not isinstance(outcomes, dict) or len(outcomes) != 2:
            continue

        ah_line: float | None = None
        ah_home: float | None = None
        ah_away: float | None = None

        for outcome_data in outcomes.values():
            if not isinstance(outcome_data, dict):
                continue
            label = _bookmaker_outcome_label(outcome_data).lower()
            price = _player_price(outcome_data)
            if price is None:
                continue
            if label.endswith("/home"):
                ah_home = price
                parsed = _parse_handicap_line(label)
                if parsed is not None:
                    ah_line = parsed
            elif label.endswith("/away"):
                ah_away = price
                if ah_line is None:
                    parsed = _parse_handicap_line(label)
                    if parsed is not None:
                        ah_line = parsed

        if ah_line is None or ah_home is None or ah_away is None:
            continue

        limit_val = _market_limit(market_data)
        if limit_val >= best_limit:
            best_limit = limit_val
            best = (ah_line, ah_home, ah_away)

    if best is None:
        return None, None, None
    return best


def format_pinnacle_ah(odds: dict[str, Any] | None) -> str:
    """Texto compacto de AH para UI, ej. 'AH: -0.25 (1.92 / 1.98)'."""
    if not odds:
        return "AH: no disponible"
    line = odds.get("ah_line")
    home = odds.get("ah_home")
    away = odds.get("ah_away")
    if line is None or home is None or away is None:
        return "AH: no disponible"
    line_f = float(line)
    if line_f > 0:
        line_str = f"+{line_f:g}"
    else:
        line_str = f"{line_f:g}"
    return f"AH: {line_str} ({float(home):.2f} / {(float(away)):.2f})"


def _parse_correct_score_label(label: str) -> tuple[int, int] | None:
    """Parsea '1:0', '1-0', etc. → (home, away)."""
    text = label.strip().replace("-", ":")
    if ":" not in text:
        return None
    parts = text.split(":", 1)
    try:
        h, a = int(parts[0]), int(parts[1])
        if h >= 0 and a >= 0:
            return h, a
    except (TypeError, ValueError):
        return None
    return None


def _is_correct_score_market(market_data: dict[str, Any]) -> bool:
    market_id = str(market_data.get("bookmakerMarketId", "")).lower()
    if "correct" in market_id or "score" in market_id:
        return True
    raw_market_id = market_data.get("marketId")
    if raw_market_id is not None and str(raw_market_id) in {"41", "199"}:
        return True
    return False


def _extract_correct_score(
    markets: dict[str, Any],
) -> dict[tuple[int, int], float] | None:
    """Correct Score: outcomes con bookmakerOutcomeId tipo '1:0', '2:1', etc."""
    best_limit = -1.0
    best: dict[tuple[int, int], float] | None = None

    for market_data in markets.values():
        if not isinstance(market_data, dict):
            continue
        if not _is_correct_score_market(market_data):
            continue

        outcomes = market_data.get("outcomes", {})
        if not isinstance(outcomes, dict) or len(outcomes) < 3:
            continue

        cs_odds: dict[tuple[int, int], float] = {}
        for outcome_data in outcomes.values():
            if not isinstance(outcome_data, dict):
                continue
            label = _bookmaker_outcome_label(outcome_data)
            price = _player_price(outcome_data)
            if price is None:
                continue
            score = _parse_correct_score_label(label)
            if score is not None:
                cs_odds[score] = price

        if len(cs_odds) < 3:
            continue

        limit_val = _market_limit(market_data)
        if limit_val >= best_limit:
            best_limit = limit_val
            best = cs_odds

    return best


def parse_correct_score_input(text: str) -> dict[tuple[int, int], float] | None:
    """Parsea '1-0:7.5, 0-0:9.0, 1-1:6.5' → dict de cuotas CS."""
    if not text or not str(text).strip():
        return None
    cs_odds: dict[tuple[int, int], float] = {}
    for part in str(text).split(","):
        chunk = part.strip()
        if not chunk or ":" not in chunk:
            continue
        score_part, _, odd_part = chunk.partition(":")
        score = _parse_correct_score_label(score_part.strip())
        if score is None:
            continue
        try:
            odd = float(odd_part.strip())
        except (TypeError, ValueError):
            continue
        if odd > 1.0:
            cs_odds[score] = odd
    return cs_odds if len(cs_odds) >= 3 else None


def parse_ou_curve_input(text: str) -> list[tuple[float, float, float]] | None:
    """Parsea '1.5:1.48/2.73, 2.0:1.87/2.04' → [(línea, over, under), ...]."""
    if not text or not str(text).strip():
        return None
    curve: list[tuple[float, float, float]] = []
    for part in str(text).split(","):
        chunk = part.strip()
        if not chunk or ":" not in chunk:
            continue
        line_part, _, prices = chunk.partition(":")
        if "/" not in prices:
            continue
        over_part, _, under_part = prices.partition("/")
        try:
            line = float(line_part.strip())
            over = float(over_part.strip())
            under = float(under_part.strip())
        except (TypeError, ValueError):
            continue
        if line > 0 and over > 1.0 and under > 1.0:
            if FOOTBALL_OU_LINE_MIN <= line <= FOOTBALL_OU_LINE_MAX:
                curve.append((line, over, under))
    if not curve:
        return None
    curve.sort(key=lambda item: item[0])
    return curve


def format_pinnacle_cs(odds: dict[str, Any] | None) -> str:
    if odds and odds.get("correct_score_odds"):
        n = len(odds["correct_score_odds"])
        return f"📊 CS incluido ({n} scores)"
    return "📊 Solo 1X2+OU"


def _bookmaker_outcome_label(outcome_data: dict[str, Any]) -> str:
    player = outcome_data.get("players", {}).get("0", {})
    if isinstance(player, dict):
        value = player.get("bookmakerOutcomeId")
        if value:
            return str(value)
    return _outcome_label(outcome_data)


def extract_pinnacle_odds(bookmaker_odds: dict[str, Any]) -> dict[str, float | None] | None:
    """
    Extrae 1X2, O/U 2.5 y AH principal de bookmakerOdds["pinnacle"].

    Retorna dict con keys: home, draw, away, over, under, ah_line, ah_home, ah_away,
    correct_score_odds, tt_home_over, tt_home_under, tt_away_over, tt_away_under,
    ou_curve (opcionales).
    Retorna None si falta alguna de las 5 odds 1X2/O/U.
    AH es opcional (None si no está disponible).
    """
    pinnacle = bookmaker_odds.get(BOOKMAKER_PINNACLE, {})
    if not isinstance(pinnacle, dict):
        return None
    markets = pinnacle.get("markets", {})
    if not isinstance(markets, dict):
        return None

    home, draw, away = _extract_1x2(markets)
    main_line, over, under, ou_curve = _resolve_main_totals_and_curve(markets)

    if None in (home, draw, away, over, under, main_line):
        return None

    ah_line, ah_home, ah_away = _extract_asian_handicap(markets)
    correct_score_odds = _extract_correct_score(markets)
    tt_home_over, tt_home_under, tt_away_over, tt_away_under, tt_outcome_ids = (
        _extract_team_totals(markets)
    )

    print("=== TT DEBUG ===")
    print(f"  tt_home_over outlet: {tt_home_over}")
    print(f"  tt_home_under outlet: {tt_home_under}")
    print(f"  tt_away_over outlet: {tt_away_over}")
    print(f"  tt_away_under outlet: {tt_away_under}")
    print(f"  bookmakerOutcomeIds encontrados para TT: {tt_outcome_ids}")

    return {
        "home": home,
        "draw": draw,
        "away": away,
        "over": over,
        "under": under,
        "line": main_line,
        "ah_line": ah_line,
        "ah_home": ah_home,
        "ah_away": ah_away,
        "correct_score_odds": correct_score_odds,
        "tt_home_over": tt_home_over,
        "tt_home_under": tt_home_under,
        "tt_away_over": tt_away_over,
        "tt_away_under": tt_away_under,
        "ou_curve": ou_curve,
    }


def explain_pinnacle_odds_missing(bookmaker_odds: dict[str, Any]) -> str:
    """Mensaje específico según qué mercado falta en Pinnacle."""
    if BOOKMAKER_PINNACLE not in bookmaker_odds:
        return "⚠️ Pinnacle no disponible para este partido"

    pinnacle = bookmaker_odds.get(BOOKMAKER_PINNACLE, {})
    if not isinstance(pinnacle, dict):
        return "⚠️ Pinnacle no disponible para este partido"

    markets = pinnacle.get("markets", {})
    if not isinstance(markets, dict) or not markets:
        return "⚠️ Pinnacle no disponible para este partido"

    home, draw, away = _extract_1x2(markets)
    if None in (home, draw, away):
        return "⚠️ Pinnacle no tiene 1X2 para este partido"

    over, under = _extract_main_totals(markets)[1:]
    if over is None or under is None:
        return "⚠️ Pinnacle no tiene O/U (totals) para este partido"


def parse_pinnacle_odds(
    odds_payload: dict[str, Any],
    *,
    fixture: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Extrae 1X2 + O/U 2.5 de Pinnacle desde la respuesta de /odds."""
    bookmaker_odds = odds_payload.get("bookmakerOdds")
    if not isinstance(bookmaker_odds, dict):
        return None
    if BOOKMAKER_PINNACLE not in bookmaker_odds:
        return None

    extracted = extract_pinnacle_odds(bookmaker_odds)
    if extracted is None:
        return None

    meta = fixture or {}
    return {
        "fixtureId": str(odds_payload.get("fixtureId") or meta.get("fixtureId", "")),
        "home_team": str(meta.get("participant1Name", "")),
        "away_team": str(meta.get("participant2Name", "")),
        "startTime": meta.get("startTime"),
        **extracted,
        "line": DEFAULT_TOTALS_LINE,
    }


def fetch_fixture_odds_payload(
    api_key: str,
    fixture_id: str,
) -> dict[str, Any]:
    """Respuesta cruda de /odds para un fixture."""
    fixture_id = _validate_fixture_id(fixture_id)
    response = _api_get(
        f"{BASE_URL}/odds",
        params={
            "apiKey": _ensure_api_key(api_key),
            "fixtureId": fixture_id,
            "bookmaker": BOOKMAKER_PINNACLE,
            "oddsFormat": ODDS_FORMAT_DECIMAL,
        },
        timeout=30,
    )
    _log_raw_response(response)
    payload = _handle_response(response)
    if not isinstance(payload, dict):
        return {}
    return payload


def get_pinnacle_markets_debug(bookmaker_odds: dict[str, Any]) -> dict[str, Any]:
    """Markets crudos de Pinnacle para debug."""
    pinnacle = bookmaker_odds.get(BOOKMAKER_PINNACLE, {})
    markets = pinnacle.get("markets", {})
    return markets if isinstance(markets, dict) else {}


def fetch_and_parse_fixture_odds(
    api_key: str,
    fixture_id: str,
    *,
    fixture: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Descarga odds, loguea response crudo y parsea. Retorna (odds, error)."""
    payload = fetch_fixture_odds_payload(api_key, fixture_id)

    if not payload:
        return None, "⚠️ Pinnacle no disponible para este partido"

    if "bookmakerOdds" not in payload:
        payload = {**payload, "fixtureId": fixture_id}

    bookmaker_odds = payload.get("bookmakerOdds", {})
    odds = parse_pinnacle_odds(payload, fixture=fixture)
    if odds is not None:
        return odds, None

    if isinstance(bookmaker_odds, dict):
        return None, explain_pinnacle_odds_missing(bookmaker_odds)
    return None, "⚠️ Pinnacle no disponible para este partido"


def fetch_fixture_pinnacle_odds(
    api_key: str,
    fixture_id: str,
    *,
    fixture: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Odds Pinnacle completas para un fixture del Mundial."""
    odds, _ = fetch_and_parse_fixture_odds(api_key, fixture_id, fixture=fixture)
    return odds


def parse_fixture_start_datetime(start_time: Any) -> datetime | None:
    """Convierte startTime de OddsPapi a datetime UTC."""
    if start_time is None:
        return None
    if isinstance(start_time, datetime):
        return start_time if start_time.tzinfo else start_time.replace(tzinfo=timezone.utc)
    if isinstance(start_time, (int, float)):
        try:
            ts = float(start_time)
            if ts > 1_000_000_000_000:
                ts /= 1000.0
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        except (OSError, OverflowError, ValueError):
            return None
    text = str(start_time).strip()
    if not text:
        return None
    if text.isdigit():
        try:
            ts = float(text)
            if ts > 1_000_000_000_000:
                ts /= 1000.0
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        except (OSError, OverflowError, ValueError):
            return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def fixture_has_started(
    fixture: dict[str, Any],
    *,
    now: datetime | None = None,
) -> bool:
    """True si el partido ya empezó (startTime < ahora UTC)."""
    start = parse_fixture_start_datetime(fixture.get("startTime"))
    if start is None:
        return False
    current = now or datetime.now(timezone.utc)
    return start < current


def format_fixture_start_time(
    start_time: Any,
    *,
    tz: timezone | None = None,
) -> str:
    """Formatea startTime como DD/MM HH:MM (por defecto hora Argentina UTC-3)."""
    dt = parse_fixture_start_datetime(start_time)
    if dt is None:
        if start_time is None:
            return "—"
        return str(start_time)
    display_tz = tz if tz is not None else AR_TZ
    return dt.astimezone(display_tz).strftime("%d/%m %H:%M")
