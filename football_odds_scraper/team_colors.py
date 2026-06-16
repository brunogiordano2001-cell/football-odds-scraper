"""Colores primarios por equipo del Mundial 2026 (para gráficos de evolución)."""

from __future__ import annotations

from football_odds_scraper.world_cup_teams import WORLD_CUP_TEAMS, get_team_display

TEAM_COLORS: dict[str, str] = {
    "Korea Republic": "#CD2E3A",
    "Czechia": "#11457E",
    "Canada": "#FF0000",
    "Bosnia and Herzegovina": "#002395",
    "USA": "#3C3B6E",
    "Paraguay": "#D52B1E",
    "Qatar": "#8A1538",
    "Switzerland": "#FF0000",
    "Brazil": "#009C3B",
    "Morocco": "#C1272D",
    "Haiti": "#00209F",
    "Scotland": "#0065BD",
    "Australia": "#FFCD00",
    "Turkiye": "#E30A17",
    "Germany": "#000000",
    "Curacao": "#002B7F",
    "Netherlands": "#FF6600",
    "Japan": "#BC002D",
    "Ivory Coast": "#F77F00",
    "Ecuador": "#FFD100",
    "Sweden": "#006AA7",
    "Tunisia": "#E70013",
    "Spain": "#AA151B",
    "Cape Verde": "#003893",
    "Belgium": "#EF3340",
    "Egypt": "#CE1126",
    "Saudi Arabia": "#006C35",
    "Uruguay": "#0038A8",
    "IR Iran": "#239F40",
    "New Zealand": "#000000",
    "France": "#002395",
    "Senegal": "#00853F",
    "Iraq": "#007A3D",
    "Norway": "#BA0C2F",
    "Argentina": "#74ACDF",
    "Algeria": "#006233",
    "Austria": "#ED2939",
    "Jordan": "#007A3D",
    "Portugal": "#006600",
    "Congo DR": "#007FFF",
    "England": "#FFFFFF",
    "Croatia": "#171796",
    "Ghana": "#006B3F",
    "Panama": "#005293",
    "Uzbekistan": "#1EB53A",
    "Colombia": "#FCD116",
    "Mexico": "#006847",
    "South Africa": "#007A4D",
}

_DEFAULT_HOME_COLOR = "#2563EB"
_DEFAULT_AWAY_COLOR = "#DC2626"
_DRAW_COLOR = "#94A3B8"


def get_team_color(participant_id: int | None, *, fallback: str = "#64748B") -> str:
    if participant_id is None:
        return fallback
    team = WORLD_CUP_TEAMS.get(int(participant_id))
    if not team:
        return fallback
    return TEAM_COLORS.get(team["name"], fallback)


def get_team_color_by_display(display: str, *, fallback: str = "#64748B") -> str:
    """Busca color por nombre en string tipo '🇧🇪 Belgium'."""
    for name, color in TEAM_COLORS.items():
        if name in display:
            return color
    return fallback


def team_display_short(participant_id: int | None) -> str:
    """Nombre corto con bandera para leyendas."""
    text = get_team_display(participant_id)
    parts = text.split(" ", 1)
    if len(parts) == 2:
        return parts[1]
    return text
