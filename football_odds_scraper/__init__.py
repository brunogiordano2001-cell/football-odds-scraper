"""Scraper asíncrono de cuotas de fútbol (1X2 y Más/Menos 2.5)."""

from football_odds_scraper.models import MatchOdds, SelectorConfig
from football_odds_scraper.probability import (
    fair_probabilities,
    implied_probabilities,
    overround,
    remove_overround,
)
from football_odds_scraper.scraper import OddsScraper, ScraperError, scrape_match_odds
from football_odds_scraper.backtest import BacktestReport, load_football_data_csv, run_backtest
from football_odds_scraper.score_predictor import (
    GlobalModelParams,
    PoissonFit,
    ScorePredictor,
    BASE_SCORE_RATES,
    calibrate_lambdas,
    calibrate_from_correct_score,
    find_optimal_rho,
)

__all__ = [
    "MatchOdds",
    "SelectorConfig",
    "OddsScraper",
    "ScraperError",
    "scrape_match_odds",
    "implied_probabilities",
    "fair_probabilities",
    "remove_overround",
    "overround",
    "ScorePredictor",
    "PoissonFit",
    "GlobalModelParams",
    "calibrate_lambdas",
    "calibrate_from_correct_score",
    "find_optimal_rho",
    "BASE_SCORE_RATES",
    "run_backtest",
    "load_football_data_csv",
    "BacktestReport",
]

__version__ = "0.1.0"
