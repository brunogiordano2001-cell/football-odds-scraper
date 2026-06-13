from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm

from football_odds_scraper.backtest import (
    DEFAULT_ODDS_COLUMNS,
    GOALS_AWAY_COL,
    GOALS_HOME_COL,
    _is_valid_odd,
    _parse_goals,
    load_football_data_csv,
    most_likely_exact_score,
    outcome_from_goals,
    predicted_outcome_from_matrix,
)
from football_odds_scraper.probability import remove_overround
from football_odds_scraper.score_predictor import GlobalModelParams, ScorePredictor

_EPS = 1e-15
_BENCHMARK_EXACT_PCT = 11.56
_SEASON_SUFFIX_RE = re.compile(r"_(\d{4})$", re.IGNORECASE)


@dataclass
class SeasonFile:
    path: Path
    season_id: str


@dataclass
class SplitResult:
    train: pd.DataFrame
    test: pd.DataFrame
    train_seasons: list[str]
    test_seasons: list[str]
    files_loaded: list[str]


@dataclass
class MatchEvaluation:
    season_id: str
    date: str | None
    home_team: str | None
    away_team: str | None
    actual_score: str
    predicted_score: str
    actual_outcome: str
    predicted_outcome: str
    exact_hit: bool
    direction_hit: bool
    log_loss: float
    brier_1x2: float
    prob_actual_score: float
    p_home: float
    p_draw: float
    p_away: float


@dataclass
class AdvancedBacktestReport:
    train_matches: int
    test_matches: int
    train_seasons: list[str]
    test_seasons: list[str]
    global_params: GlobalModelParams
    exact_hits: int = 0
    direction_hits: int = 0
    evaluated: int = 0
    skipped_test: int = 0
    mean_log_loss: float = 0.0
    mean_brier_1x2: float = 0.0
    rows: list[MatchEvaluation] = field(default_factory=list)

    @property
    def exact_hit_rate(self) -> float:
        return self.exact_hits / self.evaluated if self.evaluated else 0.0

    @property
    def direction_hit_rate(self) -> float:
        return self.direction_hits / self.evaluated if self.evaluated else 0.0


# ---------------------------------------------------------------------------
# Carga y partición temporal
# ---------------------------------------------------------------------------


def parse_season_id(path: Path) -> str:
    """Extrae id de temporada desde nombre de archivo (p. ej. E0_2122 → 2122)."""
    match = _SEASON_SUFFIX_RE.search(path.stem)
    if match:
        return match.group(1)
    return path.stem


def discover_csv_files(data_dir: str | Path) -> list[SeasonFile]:
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise NotADirectoryError(f"No es un directorio: {data_dir}")

    files = sorted(data_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No hay CSV en {data_dir}")

    return [SeasonFile(path=f, season_id=parse_season_id(f)) for f in files]


def load_season_folder(
    data_dir: str | Path,
    *,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Lee y concatena todos los CSV de una carpeta football-data.co.uk."""
    season_files = discover_csv_files(data_dir)
    frames: list[pd.DataFrame] = []

    iterator: Sequence[SeasonFile] = season_files
    if show_progress:
        iterator = tqdm(season_files, desc="Cargando CSV", unit="archivo")

    for sf in iterator:
        df = load_football_data_csv(sf.path)
        df = df.copy()
        df["_source_file"] = sf.path.name
        df["_season_id"] = sf.season_id
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True, sort=False)
    return combined


def temporal_train_test_split(
    df: pd.DataFrame,
    *,
    n_train_seasons: int = 4,
    season_col: str = "_season_id",
) -> SplitResult:
    """Partición temporal: hasta ``n_train_seasons`` antiguas → train; última → test."""
    if season_col not in df.columns:
        raise KeyError(f"Columna {season_col!r} ausente. Usa load_season_folder().")

    seasons = sorted(df[season_col].dropna().unique().tolist())
    if len(seasons) < 2:
        raise ValueError(
            f"Se necesitan al menos 2 temporadas distintas; encontradas: {seasons}"
        )

    test_seasons = [seasons[-1]]
    train_candidates = seasons[:-1]
    train_seasons = train_candidates[-n_train_seasons:]

    train = df[df[season_col].isin(train_seasons)].copy()
    test = df[df[season_col].isin(test_seasons)].copy()

    files_loaded = sorted(df["_source_file"].dropna().unique().tolist()) if "_source_file" in df.columns else []

    return SplitResult(
        train=train,
        test=test,
        train_seasons=train_seasons,
        test_seasons=test_seasons,
        files_loaded=files_loaded,
    )


# ---------------------------------------------------------------------------
# Entrenamiento y evaluación
# ---------------------------------------------------------------------------


def train_global_model(
    df_train: pd.DataFrame,
    *,
    max_goals: int = 5,
    goals_line: float = 2.5,
    recalibrate_rates: bool = False,
) -> GlobalModelParams:
    """MLE global de ρ y π sobre el conjunto de entrenamiento."""
    ScorePredictor._global_params = None
    return ScorePredictor.fit_global(
        df_train,
        max_goals=max_goals,
        goals_line=goals_line,
        recalibrate_rates=recalibrate_rates,
    )


def _extract_match_row(
    row: pd.Series,
    cols: dict[str, str],
) -> dict[str, Any] | None:
    h, d, a = row.get(cols["home"]), row.get(cols["draw"]), row.get(cols["away"])
    o, u = row.get(cols["over"]), row.get(cols["under"])

    if not all(_is_valid_odd(v) for v in (h, d, a, o, u)):
        return None

    fthg = _parse_goals(row.get(GOALS_HOME_COL))
    ftag = _parse_goals(row.get(GOALS_AWAY_COL))
    if fthg is None or ftag is None:
        return None

    return {
        "odds_1x2": {"home": float(h), "draw": float(d), "away": float(a)},
        "odds_ou": {"over": float(o), "under": float(u)},
        "fthg": fthg,
        "ftag": ftag,
        "season_id": str(row.get("_season_id", "")),
        "date": _str_or_none(row, "Date"),
        "home_team": _str_or_none(row, "HomeTeam"),
        "away_team": _str_or_none(row, "AwayTeam"),
    }


def _str_or_none(row: pd.Series, col: str) -> str | None:
    if col not in row.index or pd.isna(row[col]):
        return None
    return str(row[col])


def _outcome_probs_from_matrix(matrix: np.ndarray) -> dict[str, float]:
    n = matrix.shape[0]
    p_home = p_draw = p_away = 0.0
    for i in range(n):
        for j in range(n):
            p = float(matrix[i, j])
            if i > j:
                p_home += p
            elif i == j:
                p_draw += p
            else:
                p_away += p
    return {"home": p_home, "draw": p_draw, "away": p_away}


def evaluate_match(
    match: dict[str, Any],
    global_params: GlobalModelParams,
    *,
    max_goals: int = 5,
    goals_line: float = 2.5,
) -> MatchEvaluation:
    fair_1x2 = remove_overround(match["odds_1x2"])
    fair_ou = remove_overround(match["odds_ou"])

    predictor = ScorePredictor(
        fair_1x2,
        fair_ou,
        max_goals=max_goals,
        goals_line=goals_line,
        global_params=global_params,
    )
    predictor.fit()
    matrix = predictor.score_matrix()
    probs_1x2 = _outcome_probs_from_matrix(matrix)

    fthg, ftag = match["fthg"], match["ftag"]
    pred_h, pred_a, pred_score = most_likely_exact_score(predictor)
    pred_outcome = predicted_outcome_from_matrix(predictor)
    actual_outcome = outcome_from_goals(fthg, ftag)

    if fthg < matrix.shape[0] and ftag < matrix.shape[1]:
        prob_actual = float(matrix[fthg, ftag])
    else:
        prob_actual = _EPS

    prob_actual = max(prob_actual, _EPS)
    log_loss = -float(np.log(prob_actual))

    actual_vec = {
        "home": 1.0 if actual_outcome == "home" else 0.0,
        "draw": 1.0 if actual_outcome == "draw" else 0.0,
        "away": 1.0 if actual_outcome == "away" else 0.0,
    }
    brier = sum((probs_1x2[k] - actual_vec[k]) ** 2 for k in ("home", "draw", "away"))

    return MatchEvaluation(
        season_id=match["season_id"],
        date=match["date"],
        home_team=match["home_team"],
        away_team=match["away_team"],
        actual_score=f"{fthg}-{ftag}",
        predicted_score=pred_score,
        actual_outcome=actual_outcome,
        predicted_outcome=pred_outcome,
        exact_hit=(pred_h == fthg and pred_a == ftag),
        direction_hit=(pred_outcome == actual_outcome),
        log_loss=log_loss,
        brier_1x2=brier,
        prob_actual_score=prob_actual,
        p_home=probs_1x2["home"],
        p_draw=probs_1x2["draw"],
        p_away=probs_1x2["away"],
    )


def evaluate_test_set(
    df_test: pd.DataFrame,
    global_params: GlobalModelParams,
    *,
    odds_columns: dict[str, str] | None = None,
    max_goals: int = 5,
    goals_line: float = 2.5,
    show_progress: bool = True,
) -> AdvancedBacktestReport:
    """Evalúa cada partido del set de test con el modelo ya entrenado."""
    cols = {**DEFAULT_ODDS_COLUMNS, **(odds_columns or {})}
    report = AdvancedBacktestReport(
        train_matches=0,
        test_matches=len(df_test),
        train_seasons=[],
        test_seasons=sorted(df_test["_season_id"].dropna().unique().tolist())
        if "_season_id" in df_test.columns
        else [],
        global_params=global_params,
    )

    rows_iter = df_test.iterrows()
    if show_progress:
        rows_iter = tqdm(
            rows_iter,
            total=len(df_test),
            desc="Evaluando test",
            unit="partido",
        )

    for _, row in rows_iter:
        match = _extract_match_row(row, cols)
        if match is None:
            report.skipped_test += 1
            continue

        try:
            ev = evaluate_match(
                match,
                global_params,
                max_goals=max_goals,
                goals_line=goals_line,
            )
        except (ValueError, FloatingPointError):
            report.skipped_test += 1
            continue

        report.evaluated += 1
        if ev.exact_hit:
            report.exact_hits += 1
        if ev.direction_hit:
            report.direction_hits += 1
        report.rows.append(ev)

    if report.evaluated > 0:
        report.mean_log_loss = float(
            np.mean([r.log_loss for r in report.rows])
        )
        report.mean_brier_1x2 = float(
            np.mean([r.brier_1x2 for r in report.rows])
        )

    return report


def run_advanced_backtest(
    data_dir: str | Path,
    *,
    n_train_seasons: int = 4,
    max_goals: int = 5,
    goals_line: float = 2.5,
    recalibrate_rates: bool = False,
    show_progress: bool = True,
) -> AdvancedBacktestReport:
    """Pipeline completo: carga → split → MLE train → evaluación test."""
    df = load_season_folder(data_dir, show_progress=show_progress)
    split = temporal_train_test_split(df, n_train_seasons=n_train_seasons)

    if show_progress:
        print(f"\nEntrenando MLE global ({len(split.train):,} partidos)…")
    global_params = train_global_model(
        split.train,
        max_goals=max_goals,
        goals_line=goals_line,
        recalibrate_rates=recalibrate_rates,
    )

    report = evaluate_test_set(
        split.test,
        global_params,
        max_goals=max_goals,
        goals_line=goals_line,
        show_progress=show_progress,
    )
    report.train_matches = len(split.train)
    report.test_matches = len(split.test)
    report.train_seasons = split.train_seasons
    report.test_seasons = split.test_seasons
    return report


# ---------------------------------------------------------------------------
# Reporte
# ---------------------------------------------------------------------------


def print_advanced_report(
    report: AdvancedBacktestReport,
    *,
    benchmark_exact_pct: float = _BENCHMARK_EXACT_PCT,
) -> None:
    w = 62
    line = "═" * w
    thin = "─" * w
    gp = report.global_params

    exact_pct = report.exact_hit_rate * 100
    dir_pct = report.direction_hit_rate * 100
    delta_exact = exact_pct - benchmark_exact_pct

    print()
    print(line)
    print("  ADVANCED BACKTEST — Dixon-Coles + ZIP (MLE)")
    print(line)
    print(f"  Temporadas entrenamiento : {', '.join(report.train_seasons)}")
    print(f"  Temporada test           : {', '.join(report.test_seasons)}")
    print(thin)
    print(f"  Partidos entrenamiento   : {report.train_matches:>8,}")
    print(f"  Partidos test (filas)    : {report.test_matches:>8,}")
    print(f"  Partidos evaluados       : {report.evaluated:>8,}")
    print(f"  Omitidos en test         : {report.skipped_test:>8,}")
    print(thin)
    print("  Parámetros globales (MLE)")
    print(f"    ρ  Dixon-Coles         : {gp.rho:>10.5f}")
    print(f"    π  ZIP (0-0)           : {gp.pi:>10.5f}")
    print(f"    Log-verosimilitud      : {gp.log_likelihood:>10.2f}")
    print(f"    NLL entrenamiento      : {gp.neg_log_likelihood:>10.2f}")
    print(f"    Convergencia           : {'sí' if gp.converged else 'no'}")
    print(thin)
    print("  Precisión predictiva (test)")
    print(
        f"    Marcador exacto        : {exact_pct:6.2f}%  "
        f"({report.exact_hits:,}/{report.evaluated:,})"
    )
    print(
        f"    vs benchmark previo    : {benchmark_exact_pct:6.2f}%  "
        f"(Δ {delta_exact:+.2f} pp)"
    )
    print(
        f"    Dirección 1X2          : {dir_pct:6.2f}%  "
        f"({report.direction_hits:,}/{report.evaluated:,})"
    )
    print(thin)
    print("  Calibración probabilística (test)")
    print(f"    Log-Loss medio         : {report.mean_log_loss:>10.4f}")
    print(f"    Brier Score 1X2 medio  : {report.mean_brier_1x2:>10.4f}")
    print(thin)

    if report.evaluated >= 20 and report.rows:
        print("  Desglose 1X2 por resultado real:")
        for label, key in (("Local", "home"), ("Empate", "draw"), ("Visitante", "away")):
            subset = [r for r in report.rows if r.actual_outcome == key]
            if not subset:
                continue
            hits = sum(1 for r in subset if r.direction_hit)
            print(f"    {label:10} {hits / len(subset) * 100:5.1f}%  ({hits}/{len(subset)})")

    print(line)
    print()


def export_results(report: AdvancedBacktestReport, path: str | Path) -> None:
    """Exporta detalle partido a partido."""
    if not report.rows:
        return
    pd.DataFrame([vars(r) for r in report.rows]).to_csv(path, index=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Backtest avanzado multi-temporada (Dixon-Coles + ZIP)",
    )
    parser.add_argument(
        "data_dir",
        type=str,
        help="Carpeta con CSV football-data.co.uk (E0_2122.csv, …)",
    )
    parser.add_argument(
        "--train-seasons",
        type=int,
        default=4,
        help="Nº de temporadas antiguas para entrenar (default: 4)",
    )
    parser.add_argument("--goals-line", type=float, default=2.5)
    parser.add_argument("--max-goals", type=int, default=5)
    parser.add_argument(
        "--recalibrate-rates",
        action="store_true",
        help="Recalibra λ/μ en cada paso del MLE (lento)",
    )
    parser.add_argument(
        "--export",
        type=str,
        default=None,
        metavar="PATH",
        help="CSV con detalle por partido",
    )
    parser.add_argument(
        "--benchmark-exact",
        type=float,
        default=_BENCHMARK_EXACT_PCT,
        help="Benchmark histórico marcador exacto (%%)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Sin barras de progreso",
    )
    args = parser.parse_args(argv)

    try:
        report = run_advanced_backtest(
            args.data_dir,
            n_train_seasons=args.train_seasons,
            max_goals=args.max_goals,
            goals_line=args.goals_line,
            recalibrate_rates=args.recalibrate_rates,
            show_progress=not args.quiet,
        )
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print_advanced_report(report, benchmark_exact_pct=args.benchmark_exact)

    if args.export:
        export_results(report, args.export)
        print(f"Detalle exportado → {args.export}\n")

    return 0 if report.evaluated > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
