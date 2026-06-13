from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from football_odds_scraper.score_predictor import ScorePredictor

# Columnas estándar football-data.co.uk (Bet365)
DEFAULT_ODDS_COLUMNS = {
    "home": "B365H",
    "draw": "B365D",
    "away": "B365A",
    "over": "B365>2.5",
    "under": "B365<2.5",
}
GOALS_HOME_COL = "FTHG"
GOALS_AWAY_COL = "FTAG"

Outcome = str  # "home" | "draw" | "away"


@dataclass
class BacktestRow:
    """Resultado de una fila evaluada."""

    index: int
    date: str | None
    home_team: str | None
    away_team: str | None
    actual_score: str
    predicted_score: str
    actual_outcome: Outcome
    predicted_outcome: Outcome
    exact_hit: bool
    direction_hit: bool
    lambda_home: float
    mu_away: float


@dataclass
class BacktestReport:
    """Agregado del backtest."""

    source: str
    total_rows: int
    evaluated: int
    skipped: int
    skip_reasons: dict[str, int] = field(default_factory=dict)
    exact_hits: int = 0
    direction_hits: int = 0
    rows: list[BacktestRow] = field(default_factory=list)

    @property
    def exact_hit_rate(self) -> float:
        return self.exact_hits / self.evaluated if self.evaluated else 0.0

    @property
    def direction_hit_rate(self) -> float:
        return self.direction_hits / self.evaluated if self.evaluated else 0.0


def load_football_data_csv(path: str | Path) -> pd.DataFrame:
    """Carga CSV de football-data.co.uk (encoding automático)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No existe el archivo: {path}")

    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, low_memory=False)


def _is_valid_odd(value: Any) -> bool:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    try:
        return float(value) > 1.0
    except (TypeError, ValueError):
        return False


def _parse_goals(value: Any) -> int | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    try:
        g = int(value)
        return g if g >= 0 else None
    except (ValueError, TypeError):
        return None


def outcome_from_goals(home_goals: int, away_goals: int) -> Outcome:
    if home_goals > away_goals:
        return "home"
    if home_goals < away_goals:
        return "away"
    return "draw"


def predicted_outcome_from_matrix(predictor: ScorePredictor) -> Outcome:
    """1X2 modelado agregando la matriz de marcadores."""
    matrix = predictor.score_matrix()
    p_home = p_draw = p_away = 0.0
    n = predictor.max_goals + 1

    for i in range(n):
        for j in range(n):
            p = float(matrix[i, j])
            if i > j:
                p_home += p
            elif i == j:
                p_draw += p
            else:
                p_away += p

    probs = {"home": p_home, "draw": p_draw, "away": p_away}
    return max(probs, key=probs.get)


def most_likely_exact_score(predictor: ScorePredictor) -> tuple[int, int, str]:
    top = predictor.top_exact_scores(1)
    if top.empty:
        return 0, 0, "0-0"
    row = top.iloc[0]
    return int(row["home_goals"]), int(row["away_goals"]), str(row["score"])


def run_backtest(
    df: pd.DataFrame,
    *,
    odds_columns: dict[str, str] | None = None,
    goals_home_col: str = GOALS_HOME_COL,
    goals_away_col: str = GOALS_AWAY_COL,
    goals_line: float = 2.5,
    train_global_first: bool = False,
    max_goals: int = 5,
    source_label: str = "dataset",
) -> BacktestReport:
    """Ejecuta el backtest fila a fila sobre un DataFrame."""
    cols = {**DEFAULT_ODDS_COLUMNS, **(odds_columns or {})}
    report = BacktestReport(
        source=source_label,
        total_rows=len(df),
        evaluated=0,
        skipped=0,
    )

    if train_global_first:
        try:
            ScorePredictor.fit_global(df, max_goals=max_goals, goals_line=goals_line)
        except ValueError:
            pass  # pocos partidos: se usan defaults

    def _skip(reason: str) -> None:
        report.skipped += 1
        report.skip_reasons[reason] = report.skip_reasons.get(reason, 0) + 1

    for idx, row in df.iterrows():
        h_odd = row.get(cols["home"])
        d_odd = row.get(cols["draw"])
        a_odd = row.get(cols["away"])
        o_odd = row.get(cols["over"])
        u_odd = row.get(cols["under"])

        if not all(
            _is_valid_odd(v) for v in (h_odd, d_odd, a_odd, o_odd, u_odd)
        ):
            _skip("cuotas_invalidas_o_faltantes")
            continue

        fthg = _parse_goals(row.get(goals_home_col))
        ftag = _parse_goals(row.get(goals_away_col))
        if fthg is None or ftag is None:
            _skip("resultado_faltante")
            continue

        odds_1x2 = {
            "home": float(h_odd),
            "draw": float(d_odd),
            "away": float(a_odd),
        }
        odds_ou = {"over": float(o_odd), "under": float(u_odd)}

        try:
            predictor = ScorePredictor.from_odds(
                odds_1x2,
                odds_ou,
                max_goals=max_goals,
                goals_line=goals_line,
            )
            fit = predictor.fit()
            pred_h, pred_a, pred_score = most_likely_exact_score(predictor)
            pred_outcome = predicted_outcome_from_matrix(predictor)
        except (ValueError, KeyError):
            _skip("error_modelo")
            continue

        actual_score = f"{fthg}-{ftag}"
        actual_outcome = outcome_from_goals(fthg, ftag)
        exact_hit = pred_h == fthg and pred_a == ftag
        direction_hit = pred_outcome == actual_outcome

        report.evaluated += 1
        if exact_hit:
            report.exact_hits += 1
        if direction_hit:
            report.direction_hits += 1

        report.rows.append(
            BacktestRow(
                index=int(idx) if isinstance(idx, (int, np.integer)) else 0,
                date=_cell_str(row, "Date"),
                home_team=_cell_str(row, "HomeTeam"),
                away_team=_cell_str(row, "AwayTeam"),
                actual_score=actual_score,
                predicted_score=pred_score,
                actual_outcome=actual_outcome,
                predicted_outcome=pred_outcome,
                exact_hit=exact_hit,
                direction_hit=direction_hit,
                lambda_home=fit.lambda_home,
                mu_away=fit.mu_away,
            )
        )

    return report


def _cell_str(row: pd.Series, col: str) -> str | None:
    if col not in row.index:
        return None
    val = row[col]
    if pd.isna(val):
        return None
    return str(val)


def print_report(report: BacktestReport, *, verbose: bool = False) -> None:
    """Imprime reporte formateado en terminal."""
    w = 52
    line = "═" * w
    thin = "─" * w

    print()
    print(line)
    print("  BACKTEST — ScorePredictor (football-data.co.uk)")
    print(line)
    print(f"  Archivo:        {report.source}")
    print(f"  Filas totales:  {report.total_rows:,}")
    print(f"  Evaluadas:      {report.evaluated:,}")
    print(f"  Omitidas:       {report.skipped:,}")
    print(thin)

    if report.evaluated == 0:
        print("  Sin partidos evaluables.")
        if report.skip_reasons:
            print(thin)
            print("  Motivos de omisión:")
            for reason, count in sorted(
                report.skip_reasons.items(), key=lambda x: -x[1]
            ):
                print(f"    · {reason}: {count}")
        print(line)
        print()
        return

    exact_pct = report.exact_hit_rate * 100
    dir_pct = report.direction_hit_rate * 100

    print(f"  Acierto marcador exacto:  {exact_pct:6.2f}%  "
          f"({report.exact_hits:,}/{report.evaluated:,})")
    print(f"  Acierto dirección 1X2:    {dir_pct:6.2f}%  "
          f"({report.direction_hits:,}/{report.evaluated:,})")
    print(thin)

    if report.skip_reasons:
        print("  Omisiones:")
        for reason, count in sorted(
            report.skip_reasons.items(), key=lambda x: -x[1]
        ):
            print(f"    · {reason}: {count:,}")

    # Desglose por resultado real (útil para diagnosticar sesgo)
    if report.evaluated >= 10:
        print(thin)
        print("  Dirección 1X2 por resultado real:")
        for label, key in (("Local", "home"), ("Empate", "draw"), ("Visitante", "away")):
            subset = [r for r in report.rows if r.actual_outcome == key]
            if not subset:
                continue
            hits = sum(1 for r in subset if r.direction_hit)
            rate = hits / len(subset) * 100
            print(f"    {label:10} {rate:5.1f}%  ({hits}/{len(subset)})")

    if verbose and report.rows:
        print(thin)
        print("  Últimos 5 partidos:")
        for r in report.rows[-5:]:
            mark_e = "✓" if r.exact_hit else "✗"
            mark_d = "✓" if r.direction_hit else "✗"
            teams = ""
            if r.home_team and r.away_team:
                teams = f"{r.home_team} v {r.away_team} | "
            print(
                f"    {teams}pred {r.predicted_score} → real {r.actual_score}  "
                f"[exacto {mark_e} | 1X2 {mark_d}]"
            )

    print(line)
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Backtest de ScorePredictor sobre CSV de football-data.co.uk",
    )
    parser.add_argument(
        "csv",
        type=str,
        help="Ruta al CSV (p. ej. E0.csv de football-data.co.uk)",
    )
    parser.add_argument(
        "--goals-line",
        type=float,
        default=2.5,
        help="Línea over/under (default: 2.5)",
    )
    parser.add_argument(
        "--train-global",
        action="store_true",
        help="Entrena ρ y π globales (MLE) antes del backtest",
    )
    parser.add_argument(
        "--max-goals",
        type=int,
        default=5,
        help="Tamaño de la matriz de marcadores (default: 5)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Muestra detalle de partidos al final",
    )
    parser.add_argument(
        "--export",
        type=str,
        default=None,
        metavar="PATH",
        help="Exporta resultados fila a fila a CSV",
    )
    args = parser.parse_args(argv)

    try:
        df = load_football_data_csv(args.csv)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    report = run_backtest(
        df,
        goals_line=args.goals_line,
        train_global_first=args.train_global,
        max_goals=args.max_goals,
        source_label=str(Path(args.csv).name),
    )

    print_report(report, verbose=args.verbose)

    if args.export and report.rows:
        out = pd.DataFrame([vars(r) for r in report.rows])
        out.to_csv(args.export, index=False)
        print(f"Detalle exportado → {args.export}\n")

    return 0 if report.evaluated > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
