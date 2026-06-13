from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import requests

ONEFOOTBALL_FIXTURES_URL = (
    "https://onefootball.com/es/competicion/campeonato-del-mundo-12/partidos"
)
_DEFAULT_TZ = ZoneInfo("America/Mexico_City")
_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)

# Mapeo nombre OneFootball → grupo (FIFA 2026)
_TEAM_TO_GROUP: dict[str, str] = {
    "México": "A",
    "Sudáfrica": "A",
    "Corea del Sur": "A",
    "República Checa": "A",
    "Chequia": "A",
    "Canadá": "B",
    "Catar": "B",
    "Suiza": "B",
    "Rumania": "B",
    "Brasil": "C",
    "Marruecos": "C",
    "Haití": "C",
    "Escocia": "C",
    "Estados Unidos": "D",
    "Paraguay": "D",
    "Australia": "D",
    "Turquía": "D",
    "Alemania": "E",
    "Curazao": "E",
    "Costa de Marfil": "E",
    "Ecuador": "E",
    "Países Bajos": "F",
    "Japón": "F",
    "Ucrania": "F",
    "Túnez": "F",
    "Bélgica": "G",
    "Egipto": "G",
    "Irán": "G",
    "Nueva Zelanda": "G",
    "España": "H",
    "Cabo Verde": "H",
    "Arabia Saudita": "H",
    "Uruguay": "H",
    "Francia": "I",
    "Senegal": "I",
    "Irak": "I",
    "Noruega": "I",
    "Argentina": "J",
    "Argelia": "J",
    "Austria": "J",
    "Jordania": "J",
    "Portugal": "K",
    "RD Congo": "K",
    "Congo DR": "K",
    "Uzbekistán": "K",
    "Colombia": "K",
    "Inglaterra": "L",
    "Croacia": "L",
    "Ghana": "L",
    "Panamá": "L",
}


def _extract_next_data(html: str) -> dict[str, Any]:
    match = re.search(
        r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>',
        html,
        re.DOTALL,
    )
    if not match:
        raise ValueError("No se encontró __NEXT_DATA__ en la página de OneFootball.")
    return json.loads(match.group(1))


def _parse_kickoff(iso_utc: str, tz: ZoneInfo = _DEFAULT_TZ) -> tuple[str, str]:
    dt_utc = datetime.fromisoformat(iso_utc.replace("Z", "+00:00"))
    local = dt_utc.astimezone(tz)
    return local.strftime("%d/%m/%Y"), local.strftime("%H:%M")


def _infer_group(home: str, away: str) -> str:
    gh = _TEAM_TO_GROUP.get(home, "")
    ga = _TEAM_TO_GROUP.get(away, "")
    if gh and ga:
        return gh if gh == ga else f"{gh}/{ga}"
    return gh or ga or ""


def _iter_match_cards(page_props: dict[str, Any]) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for container in page_props.get("containers", []):
        content = (
            container.get("type", {})
            .get("fullWidth", {})
            .get("component", {})
            .get("contentType", {})
        )
        appender = content.get("matchCardsListsAppender")
        if not appender:
            continue
        for lst in appender.get("lists", []):
            jornada = ""
            header = lst.get("sectionHeader") or {}
            if isinstance(header, dict):
                jornada = str(header.get("subtitle") or "")
            for card in lst.get("matchCards", []):
                home = (card.get("homeTeam") or {}).get("name", "").strip()
                away = (card.get("awayTeam") or {}).get("name", "").strip()
                kickoff = card.get("kickoff", "")
                if not home or not away or not kickoff:
                    continue
                matches.append(
                    {
                        "home": home,
                        "away": away,
                        "kickoff": kickoff,
                        "jornada": jornada,
                        "link": card.get("link", ""),
                        "period": card.get("period"),
                    }
                )
    return matches


def fetch_onefootball_worldcup_fixture(
    *,
    url: str = ONEFOOTBALL_FIXTURES_URL,
    timeout: int = 30,
    tz: ZoneInfo = _DEFAULT_TZ,
) -> pd.DataFrame:
    """
    Descarga el fixture en vivo desde OneFootball (JSON embebido __NEXT_DATA__).

    Usa ``requests`` — no requiere Playwright para esta página.
    """
    response = requests.get(
        url,
        headers={"User-Agent": _USER_AGENT, "Accept-Language": "es-ES,es;q=0.9"},
        timeout=timeout,
    )
    response.raise_for_status()

    data = _extract_next_data(response.text)
    page_props = data.get("props", {}).get("pageProps", {})
    raw_matches = _iter_match_cards(page_props)

    if not raw_matches:
        raise ValueError(
            "OneFootball no devolvió partidos. ¿Cambió la estructura de la página?"
        )

    rows: list[dict[str, Any]] = []
    for i, m in enumerate(raw_matches, start=1):
        fecha, hora = _parse_kickoff(m["kickoff"], tz=tz)
        home, away = m["home"], m["away"]
        rows.append(
            {
                "match_id": f"OF-{i:03d}",
                "Grupo": _infer_group(home, away),
                "Jornada": m.get("jornada", ""),
                "Fecha": fecha,
                "Hora": hora,
                "Equipo Local": home,
                "Equipo Visitante": away,
                "odds_url": f"https://onefootball.com{m['link']}" if m.get("link") else "",
            }
        )

    df = pd.DataFrame(rows)
    df["_sort_dt"] = pd.to_datetime(
        df["Fecha"] + " " + df["Hora"],
        format="%d/%m/%Y %H:%M",
        errors="coerce",
    )
    df = df.sort_values("_sort_dt").reset_index(drop=True)
    df["match_id"] = [f"OF-{i+1:03d}" for i in range(len(df))]
    return df.drop(columns="_sort_dt")
