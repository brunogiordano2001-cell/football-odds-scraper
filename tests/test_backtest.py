from pathlib import Path

import pytest

from football_odds_scraper.backtest import (
    load_football_data_csv,
    outcome_from_goals,
    run_backtest,
)

FIXTURE = Path(__file__).parent / "fixtures" / "sample_football_data.csv"


def test_load_fixture():
    df = load_football_data_csv(FIXTURE)
    assert len(df) == 4
    assert "B365H" in df.columns
    assert "B365>2.5" in df.columns


def test_run_backtest_on_sample():
    df = load_football_data_csv(FIXTURE)
    report = run_backtest(df, source_label="sample")
    assert report.evaluated == 4
    assert report.skipped == 0
    assert 0 <= report.exact_hit_rate <= 1
    assert 0 <= report.direction_hit_rate <= 1
    assert len(report.rows) == 4


def test_skips_missing_odds():
    import pandas as pd

    df = pd.DataFrame(
        {
            "FTHG": [1],
            "FTAG": [0],
            "B365H": [None],
            "B365D": [3.5],
            "B365A": [4.0],
            "B365>2.5": [1.9],
            "B365<2.5": [1.9],
        }
    )
    report = run_backtest(df)
    assert report.evaluated == 0
    assert report.skipped == 1


def test_outcome_from_goals():
    assert outcome_from_goals(2, 1) == "home"
    assert outcome_from_goals(0, 0) == "draw"
    assert outcome_from_goals(0, 2) == "away"
