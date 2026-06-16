#!/usr/bin/env python3
"""Backtest del modelo: argmax vs value ratio, rho fijo vs optimizado."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import requests

from football_odds_scraper.backtest import (
    _is_valid_odd,
    _parse_goals,
    load_football_data_csv,
    outcome_from_goals,
    predicted_outcome_from_matrix,
)
from football_odds_scraper.score_predictor import (
    _DEFAULT_RHO,
    ScorePredictor,
    _model_p_ah_home,
    _poisson_matrix,
    _poisson_total_over_prob,
    calibrate_lambdas,
    find_optimal_rho,
    prode_points_awarded,
    prode_epv,
)
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
TRAIN_FRACTION = 0.70
RANDOM_SEED = 42

DATA_URLS = [
    "https://www.football-data.co.uk/mmz4281/2425/E0.csv",
    "https://www.football-data.co.uk/mmz4281/2324/E0.csv",
    "https://www.football-data.co.uk/mmz4281/2425/SP1.csv",
    "https://www.football-data.co.uk/mmz4281/2324/SP1.csv",
]

PINNACLE_ODDS_COLS = {
    "home": "PSH",
    "draw": "PSD",
    "away": "PSA",
    "over": "P>2.5",
    "under": "P<2.5",
}
GOALS_LINE = 2.5
MAX_GOALS = 5

PredictionMethod = Literal["argmax", "value_ratio"]


@dataclass
class ModeMetrics:
    label: str
    n_matches: int = 0
    exact_hits: int = 0
    result_hits: int = 0
    top3_hits: int = 0
    pred_1_1: int = 0

    @property
    def exact_acc(self) -> float:
        return self.exact_hits / self.n_matches if self.n_matches else 0.0

    @property
    def result_acc(self) -> float:
        return self.result_hits / self.n_matches if self.n_matches else 0.0

    @property
    def top3_acc(self) -> float:
        return self.top3_hits / self.n_matches if self.n_matches else 0.0

    @property
    def pct_1_1(self) -> float:
        return self.pred_1_1 / self.n_matches if self.n_matches else 0.0


@dataclass
class ProdeMetrics(ModeMetrics):
    epv_sum: float = 0.0
    total_points: int = 0
    lambda_home_sum: float = 0.0
    lambda_away_sum: float = 0.0
    lambda_home_sq_sum: float = 0.0
    lambda_away_sq_sum: float = 0.0

    @property
    def avg_epv(self) -> float:
        return self.epv_sum / self.n_matches if self.n_matches else 0.0

    @property
    def avg_points(self) -> float:
        return self.total_points / self.n_matches if self.n_matches else 0.0

    @property
    def lambda_home_mean(self) -> float:
        return self.lambda_home_sum / self.n_matches if self.n_matches else 0.0

    @property
    def lambda_away_mean(self) -> float:
        return self.lambda_away_sum / self.n_matches if self.n_matches else 0.0

    @property
    def lambda_home_std(self) -> float:
        if self.n_matches < 2:
            return 0.0
        mean = self.lambda_home_mean
        var = self.lambda_home_sq_sum / self.n_matches - mean * mean
        return float(np.sqrt(max(var, 0.0)))

    @property
    def lambda_away_std(self) -> float:
        if self.n_matches < 2:
            return 0.0
        mean = self.lambda_away_mean
        var = self.lambda_away_sq_sum / self.n_matches - mean * mean
        return float(np.sqrt(max(var, 0.0)))


@dataclass
class Top3CoverageMetrics:
    """Modo H: top 3 EPV puro vs top 3 con diversidad de resultado."""

    label: str
    n_matches: int = 0
    top3_pure_hits: int = 0
    top3_diverse_hits: int = 0

    @property
    def top3_pure_acc(self) -> float:
        return self.top3_pure_hits / self.n_matches if self.n_matches else 0.0

    @property
    def top3_diverse_acc(self) -> float:
        return self.top3_diverse_hits / self.n_matches if self.n_matches else 0.0


def _url_to_filename(url: str) -> str:
    parts = url.rstrip("/").split("/")
    return f"{parts[-1].replace('.csv', '')}_{parts[-2]}.csv"


def download_csv(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 0:
        return dest
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    dest.write_bytes(response.content)
    return dest


def load_datasets() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for url in DATA_URLS:
        path = download_csv(url, DATA_DIR / _url_to_filename(url))
        frames.append(load_football_data_csv(path))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def filter_valid_matches(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in df.iterrows():
        if not all(
            _is_valid_odd(row.get(PINNACLE_ODDS_COLS[k]))
            for k in ("home", "draw", "away", "over", "under")
        ):
            continue
        fthg = _parse_goals(row.get("FTHG"))
        ftag = _parse_goals(row.get("FTAG"))
        if fthg is None or ftag is None:
            continue
        rows.append(row.to_dict())
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).reset_index(drop=True)


def train_test_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    shuffled = df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    split = int(len(shuffled) * TRAIN_FRACTION)
    return shuffled.iloc[:split].copy(), shuffled.iloc[split:].copy()


def _fair_to_odds(prob: float, overround: float = 1.05) -> float:
    return overround / max(float(prob), 1e-6)


_SYNTHETIC_OU_LINES = [1.5, 2.0, 2.5, 3.0, 3.5]
_SYNTHETIC_AH_LINES = [-1.5, -1.0, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1.0, 1.5]


def _synthetic_ou_curve(
    lambda_home: float,
    lambda_away: float,
    *,
    overround: float = 1.05,
) -> list[tuple[float, float, float]]:
    lambda_total = float(lambda_home) + float(lambda_away)
    curve: list[tuple[float, float, float]] = []
    for line in _SYNTHETIC_OU_LINES:
        p_over = _poisson_total_over_prob(lambda_total, line)
        p_under = max(1.0 - p_over, 1e-6)
        curve.append((line, overround / p_over, overround / p_under))
    return curve


def _synthetic_ah_curve(
    lambda_home: float,
    lambda_away: float,
    *,
    overround: float = 1.05,
) -> list[tuple[float, float, float]]:
    matrix = _poisson_matrix(float(lambda_home), float(lambda_away))
    curve: list[tuple[float, float, float]] = []
    for line in _SYNTHETIC_AH_LINES:
        p_home = _model_p_ah_home(matrix, line)
        p_away = max(1.0 - p_home, 1e-6)
        curve.append((line, overround / p_home, overround / p_away))
    return curve


def _synthetic_cs_odds(
    odds_1x2: dict[str, float],
    odds_ou: dict[str, float],
    *,
    matrix_size: int = 8,
    overround: float = 1.10,
) -> dict[tuple[int, int], float]:
    """Proxy CS Poisson+OR para backtest cuando no hay mercado CS histórico."""
    lh, la = calibrate_lambdas(
        odds_1x2["home"],
        odds_1x2["draw"],
        odds_1x2["away"],
        odds_ou["over"],
        odds_ou["under"],
        input_is_odds=True,
        goals_line=GOALS_LINE,
    )
    cs: dict[tuple[int, int], float] = {}
    for h in range(matrix_size):
        for a in range(matrix_size):
            prob = stats.poisson.pmf(h, lh) * stats.poisson.pmf(a, la)
            if prob > 1e-5:
                cs[(h, a)] = overround / prob
    return cs


def evaluate_mode(
    df: pd.DataFrame,
    *,
    method: PredictionMethod,
    rho: float,
) -> ModeMetrics:
    label = f"{method}, rho={rho:.3f}"
    metrics = ModeMetrics(label=label)

    for _, row in df.iterrows():
        fthg = int(row["FTHG"])
        ftag = int(row["FTAG"])
        odds_1x2 = {
            k: float(row[PINNACLE_ODDS_COLS[k]])
            for k in ("home", "draw", "away")
        }
        odds_ou = {
            k: float(row[PINNACLE_ODDS_COLS[k]])
            for k in ("over", "under")
        }
        try:
            predictor = ScorePredictor.from_odds(
                odds_1x2,
                odds_ou,
                rho=rho,
                max_goals=MAX_GOALS,
                goals_line=GOALS_LINE,
            )
            predictor.fit()
            pred_h, pred_a, _ = predictor.most_likely_exact_score(
                method=method,
                rho=rho,
            )
            pred_outcome = predicted_outcome_from_matrix(predictor)
            top3 = predictor.top_exact_scores(3)
            actual_score = f"{fthg}-{ftag}"
            top3_hit = any(
                f"{int(r['home_goals'])}-{int(r['away_goals'])}" == actual_score
                for _, r in top3.iterrows()
            )
        except (ValueError, KeyError):
            continue

        actual_outcome = outcome_from_goals(fthg, ftag)
        metrics.n_matches += 1
        if pred_h == fthg and pred_a == ftag:
            metrics.exact_hits += 1
        if pred_outcome == actual_outcome:
            metrics.result_hits += 1
        if top3_hit:
            metrics.top3_hits += 1
        if pred_h == 1 and pred_a == 1:
            metrics.pred_1_1 += 1

    return metrics


def evaluate_mode_cs(
    df: pd.DataFrame,
    *,
    rho: float = _DEFAULT_RHO,
    use_synthetic_cs: bool = True,
) -> ModeMetrics:
    """Modo E: Correct Score directo; fallback implícito a Poisson si no hay CS."""
    metrics = ModeMetrics(label="E: Correct Score market")

    for _, row in df.iterrows():
        fthg = int(row["FTHG"])
        ftag = int(row["FTAG"])
        odds_1x2 = {
            k: float(row[PINNACLE_ODDS_COLS[k]])
            for k in ("home", "draw", "away")
        }
        odds_ou = {
            k: float(row[PINNACLE_ODDS_COLS[k]])
            for k in ("over", "under")
        }

        cs_odds = row.get("correct_score_odds")
        if isinstance(cs_odds, dict) and cs_odds:
            cs_parsed = {
                (int(k[0]), int(k[1])): float(v)
                for k, v in cs_odds.items()
                if float(v) > 1.0
            }
        elif use_synthetic_cs:
            cs_parsed = _synthetic_cs_odds(odds_1x2, odds_ou)
        else:
            continue

        if len(cs_parsed) < 3:
            continue

        try:
            predictor = ScorePredictor.from_odds(
                odds_1x2,
                odds_ou,
                rho=rho,
                max_goals=MAX_GOALS,
                goals_line=GOALS_LINE,
                correct_score_odds=cs_parsed,
            )
            predictor.fit()
            pred_h, pred_a, _ = predictor.most_likely_exact_score(method="argmax", rho=rho)
            pred_outcome = predicted_outcome_from_matrix(predictor)
            top3 = predictor.top_exact_scores(3)
            actual_score = f"{fthg}-{ftag}"
            top3_hit = any(
                f"{int(r['home_goals'])}-{int(r['away_goals'])}" == actual_score
                for _, r in top3.iterrows()
            )
        except (ValueError, KeyError):
            continue

        actual_outcome = outcome_from_goals(fthg, ftag)
        metrics.n_matches += 1
        if pred_h == fthg and pred_a == ftag:
            metrics.exact_hits += 1
        if pred_outcome == actual_outcome:
            metrics.result_hits += 1
        if top3_hit:
            metrics.top3_hits += 1
        if pred_h == 1 and pred_a == 1:
            metrics.pred_1_1 += 1

    return metrics


def evaluate_mode_prode(
    df: pd.DataFrame,
    *,
    rho: float = _DEFAULT_RHO,
    method: Literal["argmax", "prode"] = "prode",
) -> ProdeMetrics:
    """Modo G (prode) o baseline argmax con métricas EPV/puntos prode."""
    label = "G: prode EPV" if method == "prode" else "A: argmax (prode metrics)"
    metrics = ProdeMetrics(label=label)

    for _, row in df.iterrows():
        fthg = int(row["FTHG"])
        ftag = int(row["FTAG"])
        odds_1x2 = {
            k: float(row[PINNACLE_ODDS_COLS[k]])
            for k in ("home", "draw", "away")
        }
        odds_ou = {
            k: float(row[PINNACLE_ODDS_COLS[k]])
            for k in ("over", "under")
        }
        try:
            predictor = ScorePredictor.from_odds(
                odds_1x2,
                odds_ou,
                rho=rho,
                max_goals=MAX_GOALS,
                goals_line=GOALS_LINE,
            )
            predictor.fit()
            fit = predictor._fit
            if fit is not None:
                metrics.lambda_home_sum += fit.lambda_home
                metrics.lambda_away_sum += fit.mu_away
                metrics.lambda_home_sq_sum += fit.lambda_home ** 2
                metrics.lambda_away_sq_sum += fit.mu_away ** 2
            matrix = predictor.score_matrix()
            if method == "prode":
                prode = predictor.predict_for_prode()
                pred_h, pred_a = prode["score"]
                epv = float(prode["epv"])
            else:
                pred_h, pred_a, _ = predictor.most_likely_exact_score(
                    method="argmax",
                    rho=rho,
                )
                epv = prode_epv(matrix, pred_h, pred_a)
            pred_outcome = predicted_outcome_from_matrix(predictor)
            top3 = predictor.top_exact_scores(3)
            actual_score = f"{fthg}-{ftag}"
            top3_hit = any(
                f"{int(r['home_goals'])}-{int(r['away_goals'])}" == actual_score
                for _, r in top3.iterrows()
            )
        except (ValueError, KeyError):
            continue

        actual_outcome = outcome_from_goals(fthg, ftag)
        metrics.n_matches += 1
        metrics.epv_sum += epv
        metrics.total_points += prode_points_awarded(pred_h, pred_a, fthg, ftag)
        if pred_h == fthg and pred_a == ftag:
            metrics.exact_hits += 1
        if pred_outcome == actual_outcome:
            metrics.result_hits += 1
        if top3_hit:
            metrics.top3_hits += 1
        if pred_h == 1 and pred_a == 1:
            metrics.pred_1_1 += 1

    return metrics


def evaluate_mode_j(
    df: pd.DataFrame,
    *,
    rho: float = _DEFAULT_RHO,
) -> ProdeMetrics:
    """Modo J: prode EPV con curva AH sintética desde λ de mercado."""
    metrics = ProdeMetrics(label="J: prode EPV + curva AH")

    for _, row in df.iterrows():
        fthg = int(row["FTHG"])
        ftag = int(row["FTAG"])
        odds_1x2 = {
            k: float(row[PINNACLE_ODDS_COLS[k]])
            for k in ("home", "draw", "away")
        }
        odds_ou = {
            k: float(row[PINNACLE_ODDS_COLS[k]])
            for k in ("over", "under")
        }
        try:
            ref_lh, ref_la = calibrate_lambdas(
                odds_1x2["home"],
                odds_1x2["draw"],
                odds_1x2["away"],
                odds_ou["over"],
                odds_ou["under"],
                input_is_odds=True,
                goals_line=GOALS_LINE,
            )
            ou_curve = _synthetic_ou_curve(ref_lh, ref_la)
            ah_curve = _synthetic_ah_curve(ref_lh, ref_la)
            predictor = ScorePredictor.from_odds(
                odds_1x2,
                odds_ou,
                rho=rho,
                max_goals=MAX_GOALS,
                goals_line=GOALS_LINE,
                ou_curve=ou_curve,
                ah_curve=ah_curve,
            )
            predictor.fit()
            fit = predictor._fit
            if fit is not None:
                metrics.lambda_home_sum += fit.lambda_home
                metrics.lambda_away_sum += fit.mu_away
                metrics.lambda_home_sq_sum += fit.lambda_home ** 2
                metrics.lambda_away_sq_sum += fit.mu_away ** 2
            prode = predictor.predict_for_prode()
            pred_h, pred_a = prode["score"]
            epv = float(prode["epv"])
            pred_outcome = predicted_outcome_from_matrix(predictor)
            top3 = predictor.top_exact_scores(3)
            actual_score = f"{fthg}-{ftag}"
            top3_hit = any(
                f"{int(r['home_goals'])}-{int(r['away_goals'])}" == actual_score
                for _, r in top3.iterrows()
            )
        except (ValueError, KeyError):
            continue

        actual_outcome = outcome_from_goals(fthg, ftag)
        metrics.n_matches += 1
        metrics.epv_sum += epv
        metrics.total_points += prode_points_awarded(pred_h, pred_a, fthg, ftag)
        if pred_h == fthg and pred_a == ftag:
            metrics.exact_hits += 1
        if pred_outcome == actual_outcome:
            metrics.result_hits += 1
        if top3_hit:
            metrics.top3_hits += 1
        if pred_h == 1 and pred_a == 1:
            metrics.pred_1_1 += 1

    return metrics


def _top3_hit(
    top3: list[dict[str, object]],
    actual_home: int,
    actual_away: int,
) -> bool:
    actual_score = f"{actual_home}-{actual_away}"
    return any(str(entry["score"]) == actual_score for entry in top3)


def evaluate_mode_h(
    df: pd.DataFrame,
    *,
    rho: float = _DEFAULT_RHO,
) -> Top3CoverageMetrics:
    """Modo H: compara acierto en top 3 EPV puro vs top 3 con diversidad."""
    metrics = Top3CoverageMetrics(label="H: top 3 diversificado vs puro EPV")

    for _, row in df.iterrows():
        fthg = int(row["FTHG"])
        ftag = int(row["FTAG"])
        odds_1x2 = {
            k: float(row[PINNACLE_ODDS_COLS[k]])
            for k in ("home", "draw", "away")
        }
        odds_ou = {
            k: float(row[PINNACLE_ODDS_COLS[k]])
            for k in ("over", "under")
        }
        try:
            predictor = ScorePredictor.from_odds(
                odds_1x2,
                odds_ou,
                rho=rho,
                max_goals=MAX_GOALS,
                goals_line=GOALS_LINE,
            )
            predictor.fit()
            prode = predictor.predict_for_prode()
            top3_pure = prode["top3_pure"]
            top3_diverse = prode["top3"]
        except (ValueError, KeyError):
            continue

        metrics.n_matches += 1
        if _top3_hit(top3_pure, fthg, ftag):
            metrics.top3_pure_hits += 1
        if _top3_hit(top3_diverse, fthg, ftag):
            metrics.top3_diverse_hits += 1

    return metrics


def print_top3_coverage_comparison(mode_h: Top3CoverageMetrics) -> None:
    width = 80
    print()
    print("=" * width)
    print(
        f"MODO H — Top-3 Coverage Accuracy — {mode_h.n_matches} partidos"
    )
    print("=" * width)
    print(
        f"\n{'Métrica':<36}{'Top-3 puro EPV':>18}{'Top-3 diversificado':>20}"
    )
    print("-" * width)
    print(
        f"{'Top-3 Coverage Accuracy':<36}"
        f"{mode_h.top3_pure_acc * 100:>17.1f}%"
        f"{mode_h.top3_diverse_acc * 100:>19.1f}%"
    )
    delta = (mode_h.top3_diverse_acc - mode_h.top3_pure_acc) * 100
    sign = "+" if delta >= 0 else ""
    print(
        f"\nΔ Top-3 Coverage (H diversificado − puro EPV): {sign}{delta:.1f} pp"
    )
    print("=" * width)


def print_report(
    *,
    optimal_rho: float,
    train_n: int,
    test_n: int,
    modes: dict[str, ModeMetrics],
) -> None:
    width = 72
    print("=" * width)
    print(
        f"BACKTEST 4 MODOS — train {train_n} / test {test_n} "
        "(PL + LaLiga 23/24 + 24/25)"
    )
    print(f"ρ óptimo (train, argmax): {optimal_rho:.4f}")
    print("=" * width)
    print(
        f"\n{'Modo':<28}{'Exact':>10}{'1X2':>10}{'% 1-1':>10}{'Top-3':>10}"
    )
    print("-" * width)
    for key in ("A", "B", "C", "D"):
        m = modes[key]
        print(
            f"{m.label:<28}"
            f"{m.exact_acc * 100:>9.1f}%"
            f"{m.result_acc * 100:>9.1f}%"
            f"{m.pct_1_1 * 100:>9.1f}%"
            f"{m.top3_acc * 100:>9.1f}%"
        )
    print()
    print("Objetivos: Exact Score > 14.2%  |  predicción 1-1 < 20%")
    best = max(modes.values(), key=lambda m: m.exact_acc)
    print(f"Mejor Exact Score: {best.label} ({best.exact_acc * 100:.1f}%)")
    lowest_11 = min(modes.values(), key=lambda m: m.pct_1_1)
    print(f"Menor % 1-1: {lowest_11.label} ({lowest_11.pct_1_1 * 100:.1f}%)")
    print("=" * width)


def print_prode_comparison(mode_a: ProdeMetrics, mode_g: ProdeMetrics, *, n_total: int) -> None:
    width = 80
    print()
    print("=" * width)
    print(f"MODO G — Prode EPV vs Modo A (argmax) — {n_total} partidos")
    print("=" * width)
    print(
        f"\n{'Modo':<28}{'EPV avg':>10}{'Pts avg':>10}{'Exact':>10}"
        f"{'1X2':>10}{'% 1-1':>10}{'Pts tot':>10}"
    )
    print("-" * width)
    for m in (mode_a, mode_g):
        pts_avg = m.avg_points if isinstance(m, ProdeMetrics) else float("nan")
        epv_avg = m.avg_epv if isinstance(m, ProdeMetrics) else float("nan")
        total_pts = m.total_points if isinstance(m, ProdeMetrics) else "—"
        print(
            f"{m.label:<28}"
            f"{epv_avg:>10.3f}"
            f"{pts_avg:>10.3f}"
            f"{m.exact_acc * 100:>9.1f}%"
            f"{m.result_acc * 100:>9.1f}%"
            f"{m.pct_1_1 * 100:>9.1f}%"
            f"{total_pts:>10}"
        )
    print()
    if isinstance(mode_a, ProdeMetrics):
        print(
            f"Δ EPV esperado (G − A): {mode_g.avg_epv - mode_a.avg_epv:+.3f} pts/partido"
        )
        print(
            f"Δ puntos simulados totales (G − A): "
            f"{mode_g.total_points - mode_a.total_points:+d} pts"
        )
    else:
        print(f"EPV promedio G: {mode_g.avg_epv:.3f} pts/partido")
    print("=" * width)


def print_ah_curve_comparison(mode_g: ProdeMetrics, mode_j: ProdeMetrics, *, n_total: int) -> None:
    width = 90
    print()
    print("=" * width)
    print(f"MODO J — Curva AH vs Modo G (prode EPV) — {n_total} partidos")
    print("=" * width)
    print(
        f"\n{'Modo':<28}{'Exact':>10}{'Pts avg':>10}{'λ_home':>16}{'λ_away':>16}"
    )
    print("-" * width)
    for m in (mode_g, mode_j):
        print(
            f"{m.label:<28}"
            f"{m.exact_acc * 100:>9.1f}%"
            f"{m.avg_points:>10.3f}"
            f"{m.lambda_home_mean:>8.3f}±{m.lambda_home_std:.3f}"
            f"{m.lambda_away_mean:>8.3f}±{m.lambda_away_std:.3f}"
        )
    delta_exact = (mode_j.exact_acc - mode_g.exact_acc) * 100
    delta_pts = mode_j.avg_points - mode_g.avg_points
    sign = "+" if delta_exact >= 0 else ""
    print(
        f"\nΔ Exact Score (J − G): {sign}{delta_exact:.1f} pp | "
        f"Δ Pts prode avg: {delta_pts:+.3f}"
    )
    print("=" * width)


def print_cs_comparison(mode_a: ModeMetrics, mode_e: ModeMetrics) -> None:
    width = 72
    print()
    print("=" * width)
    print("MODO E — Correct Score vs Modo A (subset con CS disponible)")
    print("=" * width)
    print(f"Partidos evaluados: {mode_e.n_matches}")
    print(
        f"\n{'Modo':<28}{'Exact':>10}{'1X2':>10}{'% 1-1':>10}{'Top-3':>10}"
    )
    print("-" * width)
    for m in (mode_a, mode_e):
        print(
            f"{m.label:<28}"
            f"{m.exact_acc * 100:>9.1f}%"
            f"{m.result_acc * 100:>9.1f}%"
            f"{m.pct_1_1 * 100:>9.1f}%"
            f"{m.top3_acc * 100:>9.1f}%"
        )
    delta = (mode_e.exact_acc - mode_a.exact_acc) * 100
    sign = "+" if delta >= 0 else ""
    print(f"\nΔ Exact Score (E − A): {sign}{delta:.1f}%")
    print("=" * width)


def main() -> int:
    print("Cargando datos Pinnacle (cache en scripts/data/)...", file=sys.stderr)
    try:
        raw = load_datasets()
    except requests.RequestException as exc:
        print(f"Error descargando datos: {exc}", file=sys.stderr)
        return 1

    valid = filter_valid_matches(raw)
    if valid.empty:
        print("No hay partidos válidos.", file=sys.stderr)
        return 1

    train_df, test_df = train_test_split(valid)
    print(
        f"Train: {len(train_df)} | Test: {len(test_df)} — "
        "optimizando ρ y evaluando 4 modos...",
        file=sys.stderr,
    )

    optimal_rho = find_optimal_rho(
        train_df,
        odds_columns=PINNACLE_ODDS_COLS,
        max_goals=MAX_GOALS,
        goals_line=GOALS_LINE,
        method="argmax",
    )

    modes = {
        "A": evaluate_mode(test_df, method="argmax", rho=_DEFAULT_RHO),
        "B": evaluate_mode(test_df, method="argmax", rho=optimal_rho),
        "C": evaluate_mode(test_df, method="value_ratio", rho=_DEFAULT_RHO),
        "D": evaluate_mode(test_df, method="value_ratio", rho=optimal_rho),
    }
    modes["A"].label = "A: argmax, ρ=-0.13"
    modes["B"].label = "B: argmax, ρ óptimo"
    modes["C"].label = "C: value ratio, ρ=-0.13"
    modes["D"].label = "D: value ratio, ρ óptimo"

    print_report(
        optimal_rho=optimal_rho,
        train_n=len(train_df),
        test_n=len(test_df),
        modes=modes,
    )

    mode_e = evaluate_mode_cs(test_df, use_synthetic_cs=True)
    mode_e.label = "E: Correct Score market"
    mode_a_cs = evaluate_mode(test_df, method="argmax", rho=_DEFAULT_RHO)
    mode_a_cs.label = "A: argmax, ρ=-0.13"
    print_cs_comparison(mode_a_cs, mode_e)
    print(
        "\nNota: CSV histórico sin CS Pinnacle — Modo E usa proxy Poisson+OR.",
        file=sys.stderr,
    )

    print(
        f"\nEvaluando Modo G (prode EPV) vs A sobre {len(valid)} partidos...",
        file=sys.stderr,
    )
    mode_a_prode = evaluate_mode_prode(valid, method="argmax", rho=_DEFAULT_RHO)
    mode_a_prode.label = "A: argmax, ρ=-0.13"
    mode_g = evaluate_mode_prode(valid, method="prode", rho=_DEFAULT_RHO)
    mode_g.label = "G: prode EPV"
    print_prode_comparison(mode_a_prode, mode_g, n_total=len(valid))

    print(
        f"\nEvaluando Modo H (top 3 diversificado vs puro EPV) "
        f"sobre {len(valid)} partidos...",
        file=sys.stderr,
    )
    mode_h = evaluate_mode_h(valid, rho=_DEFAULT_RHO)
    print_top3_coverage_comparison(mode_h)

    print(
        f"\nEvaluando Modo J (curva AH sintética) vs G sobre {len(valid)} partidos...",
        file=sys.stderr,
    )
    mode_j = evaluate_mode_j(valid, rho=_DEFAULT_RHO)
    print_ah_curve_comparison(mode_g, mode_j, n_total=len(valid))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
