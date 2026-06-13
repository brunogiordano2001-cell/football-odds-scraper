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

    @property
    def avg_epv(self) -> float:
        return self.epv_sum / self.n_matches if self.n_matches else 0.0

    @property
    def avg_points(self) -> float:
        return self.total_points / self.n_matches if self.n_matches else 0.0


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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
