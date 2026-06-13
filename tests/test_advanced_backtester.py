from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from football_odds_scraper.advanced_backtester import (
    discover_csv_files,
    load_season_folder,
    parse_season_id,
    run_advanced_backtest,
    temporal_train_test_split,
    train_global_model,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "seasons"


def _write_season(path: Path, season_suffix: str, n_rows: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_rows):
        lam, mu = 1.1 + rng.random(), 0.9 + rng.random()
        x, y = int(rng.poisson(lam)), int(rng.poisson(mu))
        rows.append(
            {
                "FTHG": x,
                "FTAG": y,
                "B365H": 2.0 + rng.random(),
                "B365D": 3.2 + rng.random(),
                "B365A": 3.5 + rng.random(),
                "B365>2.5": 1.8 + rng.random() * 0.3,
                "B365<2.5": 1.8 + rng.random() * 0.3,
                "Date": "01/01/20",
                "HomeTeam": "A",
                "AwayTeam": "B",
            }
        )
    pd.DataFrame(rows).to_csv(path / f"E0_{season_suffix}.csv", index=False)


@pytest.fixture(scope="module")
def season_folder(tmp_path_factory):
    base = tmp_path_factory.mktemp("seasons")
    for i, suffix in enumerate(["1920", "2021", "2122", "2223", "2324"]):
        _write_season(base, suffix, n_rows=35, seed=100 + i)
    return base


def test_parse_season_id():
    assert parse_season_id(Path("E0_2122.csv")) == "2122"
    assert parse_season_id(Path("SP1_2324.csv")) == "2324"


def test_discover_and_load(season_folder):
    files = discover_csv_files(season_folder)
    assert len(files) == 5
    df = load_season_folder(season_folder, show_progress=False)
    assert len(df) == 175
    assert "_season_id" in df.columns


def test_temporal_split(season_folder):
    df = load_season_folder(season_folder, show_progress=False)
    split = temporal_train_test_split(df, n_train_seasons=4)
    assert split.test_seasons == ["2324"]
    assert len(split.train_seasons) == 4
    assert len(split.test) == 35
    assert len(split.train) == 140


def test_train_global_model(season_folder):
    df = load_season_folder(season_folder, show_progress=False)
    split = temporal_train_test_split(df)
    params = train_global_model(split.train, max_goals=5)
    assert -0.35 <= params.rho <= 0.05
    assert 0.0 <= params.pi <= 0.45


def test_run_advanced_backtest_end_to_end(season_folder):
    report = run_advanced_backtest(
        season_folder,
        n_train_seasons=4,
        show_progress=False,
    )
    assert report.evaluated > 0
    assert report.train_matches > report.test_matches
    assert 0 <= report.exact_hit_rate <= 1
    assert report.mean_log_loss > 0
    assert report.mean_brier_1x2 >= 0
