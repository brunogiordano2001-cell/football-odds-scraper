import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from football_odds_scraper.worldcup_2026 import (
    FLAG_MAP,
    apply_flag,
    build_worldcup_2026_fixture,
    enrich_fixture_display,
    merge_editor_into_fixture,
    row_has_complete_odds,
    run_batch_predictions,
    strip_team_display,
    team_display_name,
)


def test_flag_map_compound_names():
    assert FLAG_MAP["Bosnia y Herzegovina"] == "🇧🇦"
    assert "🇧🇦" in apply_flag("Bosnia y Herzegovina")
    assert "🇲🇽" in team_display_name("México")
    assert strip_team_display("🇲🇽  México") == "México"


@patch("football_odds_scraper.worldcup_2026.fetch_onefootball_worldcup_fixture")
def test_build_fixture_from_onefootball(mock_fetch):
    mock_fetch.return_value = pd.DataFrame(
        {
            "match_id": ["OF-001"],
            "Grupo": ["A"],
            "Jornada": ["Fase de grupos: 1 Jornada"],
            "Fecha": ["11/06/2026"],
            "Hora": ["13:00"],
            "Equipo Local": ["México"],
            "Equipo Visitante": ["Sudáfrica"],
            "odds_url": [""],
        }
    )
    df = build_worldcup_2026_fixture()
    assert "🇲🇽" in df.iloc[0]["Equipo Local"]
    assert row_has_complete_odds(df.iloc[0]) is False


def test_merge_editor_only_odds():
    df = enrich_fixture_display(
        pd.DataFrame(
            {
                "match_id": ["OF-001"],
                "Grupo": ["A"],
                "Jornada": [""],
                "Fecha": ["11/06/2026"],
                "Hora": ["13:00"],
                "Equipo Local": ["🇲🇽  México"],
                "Equipo Visitante": ["🇿🇦  Sudáfrica"],
                "Cuota 1": [np.nan],
                "Cuota X": [np.nan],
                "Cuota 2": [np.nan],
                "Línea O/U": [2.5],
                "Cuota Over": [np.nan],
                "Cuota Under": [np.nan],
                "Predicción Top 1": [""],
                "Predicción Top 2": [""],
                "Predicción Top 3": [""],
                "odds_url": [""],
            }
        )
    )
    edited = df.copy()
    edited["Cuota 1"] = 2.5
    merged = merge_editor_into_fixture(df, edited)
    assert float(merged.iloc[0]["Cuota 1"]) == 2.5
    assert "🇲🇽" in merged.iloc[0]["Equipo Local"]


def test_batch_predictions():
    df = enrich_fixture_display(
        pd.DataFrame(
            {
                "match_id": ["OF-001"],
                "Grupo": ["A"],
                "Jornada": [""],
                "Fecha": ["11/06/2026"],
                "Hora": ["13:00"],
                "Equipo Local": ["🇲🇽  México"],
                "Equipo Visitante": ["🇿🇦  Sudáfrica"],
                "Cuota 1": [2.1],
                "Cuota X": [3.4],
                "Cuota 2": [3.5],
                "Línea O/U": [2.5],
                "Cuota Over": [1.9],
                "Cuota Under": [1.9],
                "Predicción Top 1": [""],
                "Predicción Top 2": [""],
                "Predicción Top 3": [""],
                "odds_url": [""],
            }
        )
    )
    out = run_batch_predictions(df)
    assert len(str(out.iloc[0]["Predicción Top 1"])) > 0


@pytest.mark.integration
def test_live_onefootball_fixture():
    from football_odds_scraper.onefootball_fixture import fetch_onefootball_worldcup_fixture

    df = fetch_onefootball_worldcup_fixture()
    assert len(df) >= 60
    dts = pd.to_datetime(df["Fecha"] + " " + df["Hora"], format="%d/%m/%Y %H:%M")
    assert dts.is_monotonic_increasing
