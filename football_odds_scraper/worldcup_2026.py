from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from football_odds_scraper.onefootball_fixture import fetch_onefootball_worldcup_fixture
from football_odds_scraper.score_predictor import ScorePredictor

# Mapeo explícito nombre (español) → emoji Unicode
FLAG_MAP: dict[str, str] = {
    "México": "🇲🇽",
    "Sudáfrica": "🇿🇦",
    "Corea del Sur": "🇰🇷",
    "República Checa": "🇨🇿",
    "Chequia": "🇨🇿",
    "Canadá": "🇨🇦",
    "Catar": "🇶🇦",
    "Suiza": "🇨🇭",
    "Rumania": "🇷🇴",
    "Brasil": "🇧🇷",
    "Marruecos": "🇲🇦",
    "Haití": "🇭🇹",
    "Escocia": "🏴󠁧󠁢󠁳󠁣󠁴󠁿",
    "Estados Unidos": "🇺🇸",
    "Paraguay": "🇵🇾",
    "Australia": "🇦🇺",
    "Turquía": "🇹🇷",
    "Alemania": "🇩🇪",
    "Curazao": "🇨🇼",
    "Costa de Marfil": "🇨🇮",
    "Ecuador": "🇪🇨",
    "Países Bajos": "🇳🇱",
    "Japón": "🇯🇵",
    "Ucrania": "🇺🇦",
    "Túnez": "🇹🇳",
    "Bélgica": "🇧🇪",
    "Egipto": "🇪🇬",
    "Irán": "🇮🇷",
    "Nueva Zelanda": "🇳🇿",
    "España": "🇪🇸",
    "Cabo Verde": "🇨🇻",
    "Arabia Saudita": "🇸🇦",
    "Uruguay": "🇺🇾",
    "Francia": "🇫🇷",
    "Senegal": "🇸🇳",
    "Irak": "🇮🇶",
    "Noruega": "🇳🇴",
    "Argentina": "🇦🇷",
    "Argelia": "🇩🇿",
    "Austria": "🇦🇹",
    "Jordania": "🇯🇴",
    "Portugal": "🇵🇹",
    "RD Congo": "🇨🇩",
    "Congo DR": "🇨🇩",
    "Uzbekistán": "🇺🇿",
    "Colombia": "🇨🇴",
    "Inglaterra": "🏴󠁧󠁢󠁥󠁮󠁧󠁿",
    "Croacia": "🇭🇷",
    "Ghana": "🇬🇭",
    "Panamá": "🇵🇦",
    "Bosnia y Herzegovina": "🇧🇦",
    "Bosnia-Herzegovina": "🇧🇦",
}

# Alias retrocompatible
TEAM_EMOJI = FLAG_MAP

FIXTURE_COLUMNS = [
    "match_id",
    "Grupo",
    "Jornada",
    "Fecha",
    "Hora",
    "Equipo Local",
    "Equipo Visitante",
    "Cuota 1",
    "Cuota X",
    "Cuota 2",
    "Línea O/U",
    "Cuota Over",
    "Cuota Under",
    "Predicción Top 1",
    "Predicción Top 2",
    "Predicción Top 3",
    "odds_url",
]

DISPLAY_COLUMNS = [
    "match_id",
    "Grupo",
    "Jornada",
    "Fecha",
    "Hora",
    "Equipo Local",
    "Equipo Visitante",
    *[
        "Cuota 1",
        "Cuota X",
        "Cuota 2",
        "Línea O/U",
        "Cuota Over",
        "Cuota Under",
    ],
    "Predicción Top 1",
    "Predicción Top 2",
    "Predicción Top 3",
]

ODDS_COLUMNS = ["Cuota 1", "Cuota X", "Cuota 2", "Línea O/U", "Cuota Over", "Cuota Under"]
PREDICTION_COLUMNS = ["Predicción Top 1", "Predicción Top 2", "Predicción Top 3"]


def apply_flag(team_name: str) -> str:
    """Aplica FLAG_MAP al nombre del equipo."""
    return team_display_name(team_name)


def team_display_name(team_name: str) -> str:
    """Nombre con emoji de bandera para la UI."""
    name = strip_team_display(str(team_name))
    if not name:
        return "🏳️"
    emoji = FLAG_MAP.get(name)
    if emoji is None:
        # búsqueda parcial para nombres compuestos
        for key, flag in FLAG_MAP.items():
            if key in name or name in key:
                emoji = flag
                break
    emoji = emoji or "🏳️"
    return f"{emoji}  {name}"


def strip_team_display(label: str) -> str:
    """Quita emoji y devuelve nombre canónico."""
    if not isinstance(label, str):
        return str(label)
    cleaned = label.strip()
    if "  " in cleaned:
        cleaned = cleaned.split("  ", 1)[-1].strip()
    cleaned = cleaned.lstrip("🏳️⚽️ ").strip()
    if cleaned in FLAG_MAP:
        return cleaned
    for name in sorted(FLAG_MAP.keys(), key=len, reverse=True):
        if name in cleaned:
            return name
    return cleaned


def _init_odds_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    defaults: dict[str, Any] = {
        "Cuota 1": np.nan,
        "Cuota X": np.nan,
        "Cuota 2": np.nan,
        "Línea O/U": 2.5,
        "Cuota Over": np.nan,
        "Cuota Under": np.nan,
        "Predicción Top 1": "",
        "Predicción Top 2": "",
        "Predicción Top 3": "",
    }
    for col, val in defaults.items():
        if col not in out.columns:
            out[col] = val
    for col in PREDICTION_COLUMNS:
        out[col] = out[col].astype(object)
    if "Jornada" not in out.columns:
        out["Jornada"] = ""
    if "odds_url" not in out.columns:
        out["odds_url"] = ""
    return out


def enrich_fixture_display(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica FLAG_MAP en columnas de equipos (texto con emoji)."""
    out = _init_odds_columns(df)
    out["Equipo Local"] = out["Equipo Local"].map(lambda x: apply_flag(str(x)))
    out["Equipo Visitante"] = out["Equipo Visitante"].map(lambda x: apply_flag(str(x)))
    return out


def build_worldcup_2026_fixture() -> pd.DataFrame:
    """Descarga fixture en vivo desde OneFootball y lo formatea para Streamlit."""
    raw = fetch_onefootball_worldcup_fixture()
    return enrich_fixture_display(raw)


def get_fixture_csv_schema_help() -> str:
    return (
        "El fixture se obtiene en vivo desde "
        "[OneFootball](https://onefootball.com/es/competicion/campeonato-del-mundo-12/partidos). "
        "Pulsa **Actualizar Fixture desde OneFootball** para refrescar."
    )


def _valid_odd(value: Any) -> bool:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    try:
        return float(value) > 1.0
    except (TypeError, ValueError):
        return False


def row_has_complete_odds(row: pd.Series) -> bool:
    return all(_valid_odd(row.get(c)) for c in ["Cuota 1", "Cuota X", "Cuota 2", "Cuota Over", "Cuota Under"])


def format_top_prediction(score: str, percentage: float) -> str:
    return f"{score} ({percentage:.1f}%)"


def predict_row_top_scores(
    row: pd.Series,
    *,
    goals_line: float | None = None,
) -> tuple[str, str, str]:
    line = float(row.get("Línea O/U", 2.5)) if goals_line is None else goals_line
    if np.isnan(line):
        line = 2.5

    odds_1x2 = {
        "home": float(row["Cuota 1"]),
        "draw": float(row["Cuota X"]),
        "away": float(row["Cuota 2"]),
    }
    odds_ou = {"over": float(row["Cuota Over"]), "under": float(row["Cuota Under"])}

    predictor = ScorePredictor.from_odds(odds_1x2, odds_ou, goals_line=line)
    top = predictor.top_exact_scores(3)
    if top.empty:
        return "", "", ""

    parts = [
        format_top_prediction(str(top.iloc[i]["score"]), float(top.iloc[i]["percentage"]))
        for i in range(min(3, len(top)))
    ]
    while len(parts) < 3:
        parts.append("")
    return parts[0], parts[1], parts[2]


def run_batch_predictions(df: pd.DataFrame) -> pd.DataFrame:
    out = _init_odds_columns(df.copy())
    for idx, row in out.iterrows():
        if not row_has_complete_odds(row):
            continue
        try:
            t1, t2, t3 = predict_row_top_scores(row)
            out.at[idx, "Predicción Top 1"] = t1
            out.at[idx, "Predicción Top 2"] = t2
            out.at[idx, "Predicción Top 3"] = t3
        except (ValueError, FloatingPointError):
            continue
    # No re-formatear predicciones; solo refrescar banderas en equipos
    out["Equipo Local"] = out["Equipo Local"].map(lambda x: apply_flag(str(x)))
    out["Equipo Visitante"] = out["Equipo Visitante"].map(lambda x: apply_flag(str(x)))
    return out


def merge_editor_into_fixture(base: pd.DataFrame, edited: pd.DataFrame) -> pd.DataFrame:
    """Fusiona cuotas/predicciones editadas; conserva equipos y fechas del fixture base."""
    if edited.empty or "match_id" not in edited.columns:
        return base

    out = base.set_index("match_id")
    upd = edited.set_index("match_id")
    for col in ODDS_COLUMNS + PREDICTION_COLUMNS:
        if col in upd.columns:
            out[col] = upd[col].combine_first(out[col])
    return enrich_fixture_display(out.reset_index())
