from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence, TypedDict

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize, minimize_scalar
from football_odds_scraper.probability import fair_probabilities, remove_overround

_EPS = 1e-15
_CALIB_MAX_GOALS = 8
_CS_MATRIX_SIZE = 8
_DEFAULT_RHO = -0.13
_LAMBDA_BOUNDS = (0.05, 5.5)
_RHO_BOUNDS = (-0.35, 0.05)
_PI_BOUNDS = (0.0, 0.45)

# Frecuencias históricas football-data.co.uk (prior de scores exactos)
BASE_SCORE_RATES: dict[tuple[int, int], float] = {
    (0, 0): 0.052,
    (1, 0): 0.101,
    (2, 0): 0.058,
    (3, 0): 0.040,
    (0, 1): 0.070,
    (1, 1): 0.127,
    (2, 1): 0.080,
    (3, 1): 0.030,
    (0, 2): 0.053,
    (1, 2): 0.074,
    (2, 2): 0.031,
    (3, 2): 0.013,
    (0, 3): 0.025,
    (1, 3): 0.027,
    (2, 3): 0.016,
    (4, 0): 0.018,
    (0, 4): 0.012,
    (4, 1): 0.010,
    (1, 4): 0.009,
}

_PINNACLE_ODDS_COLS = {
    "home": "PSH",
    "draw": "PSD",
    "away": "PSA",
    "over": "P>2.5",
    "under": "P<2.5",
}

# Columnas football-data.co.uk para entrenamiento global
_HIST_ODDS_COLS = {
    "home": "B365H",
    "draw": "B365D",
    "away": "B365A",
    "over": "B365>2.5",
    "under": "B365<2.5",
}
_HIST_GOALS_HOME = "FTHG"
_HIST_GOALS_AWAY = "FTAG"


def _devig_3way(h: float, d: float, a: float) -> tuple[float, float, float]:
    margin = 1.0 / h + 1.0 / d + 1.0 / a
    return 1.0 / h / margin, 1.0 / d / margin, 1.0 / a / margin


def _devig_2way(o: float, u: float) -> tuple[float, float]:
    margin = 1.0 / o + 1.0 / u
    return 1.0 / o / margin, 1.0 / u / margin


def _poisson_total_over_prob(lambda_total: float, line: float) -> float:
    """P(total goles > línea) con total ~ Poisson(λ_total).

    Líneas enteras (.0), medias (.5) y cuarto (.25/.75) según convención bookmaker.
    """
    lam = float(lambda_total)
    line_f = float(line)
    piso = int(np.floor(line_f))
    frac = round(line_f - piso, 2)

    if frac in (0.0, 1.0):
        return float(stats.poisson.sf(piso, lam))
    if frac == 0.5:
        return float(stats.poisson.sf(piso, lam))
    if frac == 0.25:
        return float(
            0.5 * stats.poisson.sf(piso, lam)
            + 0.5 * stats.poisson.sf(piso + 1, lam)
        )
    if frac == 0.75:
        return float(
            0.5 * stats.poisson.sf(piso + 1, lam)
            + 0.5 * stats.poisson.sf(piso + 2, lam)
        )
    return float(stats.poisson.sf(piso, lam))


_OU_LAMBDA_TOTAL_BOUNDS = (0.5, 5.5)
_LAMBDA_PER_TEAM_MAX = 4.0
_FOOTBALL_OU_LINE_MIN = 0.5
_FOOTBALL_OU_LINE_MAX = 5.5


def _filter_ou_curve_football(
    ou_curve: Sequence[tuple[float, float, float]] | None,
) -> list[tuple[float, float, float]] | None:
    if not ou_curve:
        return None
    filtered = [
        (float(p), float(o), float(u))
        for p, o, u in ou_curve
        if _FOOTBALL_OU_LINE_MIN <= float(p) <= _FOOTBALL_OU_LINE_MAX
    ]
    return filtered if filtered else None


def _clamp_lambdas_realistic(lambda_home: float, lambda_away: float) -> tuple[float, float]:
    """Tapa λ individuales y preserva proporción si ambos exceden el máximo."""
    lh, la = float(lambda_home), float(lambda_away)
    if lh <= _LAMBDA_PER_TEAM_MAX and la <= _LAMBDA_PER_TEAM_MAX:
        return max(0.1, lh), max(0.1, la)

    print("⚠️ Lambda fuera de rango realista, posible contaminación de mercado")
    if lh > _LAMBDA_PER_TEAM_MAX and la > _LAMBDA_PER_TEAM_MAX:
        scale = _LAMBDA_PER_TEAM_MAX / max(lh, la)
        lh *= scale
        la *= scale
    else:
        lh = min(lh, _LAMBDA_PER_TEAM_MAX)
        la = min(la, _LAMBDA_PER_TEAM_MAX)
    return max(0.1, lh), max(0.1, la)


def _has_tt_side(over_odd: float | None, under_odd: float | None) -> bool:
    try:
        return (
            over_odd is not None
            and under_odd is not None
            and float(over_odd) > 1.0
            and float(under_odd) > 1.0
        )
    except (TypeError, ValueError):
        return False


def _lambda_total_from_ou_curve(
    ou_curve: Sequence[tuple[float, float, float]],
    *,
    min_points: int = 1,
) -> float | None:
    """λ_total óptimo minimizando MSE contra curva O/U filtrada."""
    points = sorted(
        [(float(p), float(o), float(u)) for p, o, u in ou_curve],
        key=lambda item: item[0],
    )
    if len(points) < min_points:
        return None

    weights = [1.0 + 0.3 * i for i in range(len(points))]
    weight_sum = sum(weights)
    lo, hi = _OU_LAMBDA_TOTAL_BOUNDS

    def mse(lambda_total: float) -> float:
        if lambda_total < lo or lambda_total > hi:
            return 1e6
        err = 0.0
        for w, (point, over_price, under_price) in zip(weights, points):
            p_over_mkt, _ = _devig_2way(over_price, under_price)
            p_over_model = _poisson_total_over_prob(lambda_total, point)
            err += w * (p_over_model - p_over_mkt) ** 2
        return err / weight_sum

    result = minimize_scalar(mse, bounds=(lo, hi), method="bounded")
    return float(result.x)


def _refine_lambdas_with_ou_curve(
    lambda_home: float,
    lambda_away: float,
    ou_curve: Sequence[tuple[float, float, float]],
    *,
    discrepancy_threshold: float = 0.15,
) -> tuple[float, float]:
    """Escala λ_home/λ_away si λ_total difiere >15% del optimizado por curva O/U."""
    lambda_sum = lambda_home + lambda_away
    if lambda_sum <= _EPS:
        return lambda_home, lambda_away

    opt_total = _lambda_total_from_ou_curve(ou_curve, min_points=3)
    if opt_total is None:
        return lambda_home, lambda_away

    rel_diff = abs(opt_total - lambda_sum) / max(lambda_sum, _EPS)
    if rel_diff <= discrepancy_threshold:
        return lambda_home, lambda_away

    ratio = opt_total / lambda_sum
    return max(0.1, lambda_home * ratio), max(0.1, lambda_away * ratio)


def _estimate_lambda_total_fallback(goals_line: float, p_over: float) -> float:
    return float(goals_line) if p_over > 0.5 else max(0.5, float(goals_line) - 0.5)


def _calibrate_from_team_totals(
    *,
    tt_home_over: float | None,
    tt_home_under: float | None,
    tt_away_over: float | None,
    tt_away_under: float | None,
    ou_curve_filtered: list[tuple[float, float, float]] | None,
    goals_line: float,
    p_over: float,
    p_over_price: float,
) -> tuple[float, float]:
    """Calibración TT 0.5 (completa o parcial) + curva O/U."""
    has_home_tt = _has_tt_side(tt_home_over, tt_home_under)
    has_away_tt = _has_tt_side(tt_away_over, tt_away_under)

    lambda_total_opt = (
        _lambda_total_from_ou_curve(ou_curve_filtered, min_points=1)
        if ou_curve_filtered
        else None
    )
    if lambda_total_opt is None:
        lambda_total_opt = _estimate_lambda_total_fallback(goals_line, p_over)

    lh0: float | None = None
    la0: float | None = None
    if has_home_tt:
        lh0 = _lambda_from_team_total(float(tt_home_over), float(tt_home_under))  # type: ignore[arg-type]
    if has_away_tt:
        la0 = _lambda_from_team_total(float(tt_away_over), float(tt_away_under))  # type: ignore[arg-type]

    if has_home_tt and has_away_tt:
        assert lh0 is not None and la0 is not None
        lh, la = _refine_lambdas_with_ou_curve(lh0, la0, ou_curve_filtered or [])
        mode = "ambos TT 0.5"
    elif has_away_tt and not has_home_tt:
        assert la0 is not None
        la = la0
        lh = max(0.1, lambda_total_opt - la0)
        mode = "solo away TT 0.5 → λ_home = λ_total − λ_away"
    elif has_home_tt and not has_away_tt:
        assert lh0 is not None
        lh = lh0
        la = max(0.1, lambda_total_opt - lh0)
        mode = "solo home TT 0.5 → λ_away = λ_total − λ_home"
    else:
        raise ValueError("team totals no disponibles")

    if ou_curve_filtered and len(ou_curve_filtered) >= 3:
        curve_preview = [
            (p, round(o, 3), round(u, 3)) for p, o, u in ou_curve_filtered[:5]
        ]
        print("=== CALIBRACIÓN ===")
        print(f"  Modo: {mode}")
        print(f"  O/U main line: {goals_line} | over: {p_over_price}")
        if lh0 is not None and la0 is not None:
            print(
                f"  λ inicial (desde TT): home={lh0:.3f} away={la0:.3f} "
                f"total={lh0 + la0:.3f}"
            )
        elif la0 is not None:
            print(f"  λ away (desde TT): {la0:.3f} | λ_total O/U: {lambda_total_opt:.3f}")
        elif lh0 is not None:
            print(f"  λ home (desde TT): {lh0:.3f} | λ_total O/U: {lambda_total_opt:.3f}")
        print(
            f"  λ final: home={lh:.3f} away={la:.3f} total={lh + la:.3f}"
        )
        print(f"  Curva O/U usada: {curve_preview}")

    return lh, la


def _lambda_from_team_total(over_odd: float, under_odd: float) -> float:
    """λ desde team total 0.5: P(0 goles) = e^{-λ} = prob implícita del under."""
    _, p_under = _devig_2way(float(over_odd), float(under_odd))
    p_zero = float(np.clip(p_under, _EPS, 1.0 - _EPS))
    return max(0.05, -float(np.log(p_zero)))


def _calibrate_lambdas_joint_fallback(
    ph: float,
    pd: float,
    pa: float,
    po: float,
    *,
    ah_line: float | None,
    ah_home: float | None,
    ah_away: float | None,
    goals_line: float,
) -> tuple[float, float]:
    """Calibración conjunta 1X2 + O/U (+ AH) vía Nelder-Mead."""
    max_goals = _CALIB_MAX_GOALS

    def poisson_matrix(lh: float, la: float) -> np.ndarray:
        ph_vec = stats.poisson.pmf(np.arange(max_goals), lh)
        pa_vec = stats.poisson.pmf(np.arange(max_goals), la)
        return np.outer(ph_vec, pa_vec)

    def model_probs(lh: float, la: float) -> tuple[float, float, float, float]:
        matrix = poisson_matrix(lh, la)
        p_h = float(np.sum(np.tril(matrix, -1)))
        p_d = float(np.sum(np.diag(matrix)))
        p_a = float(np.sum(np.triu(matrix, 1)))
        p_o = 0.0
        for i in range(max_goals):
            for j in range(max_goals):
                if i + j > goals_line:
                    p_o += matrix[i, j]
        return p_h, p_d, p_a, float(p_o)

    def loss(params: np.ndarray) -> float:
        lh, la = float(params[0]), float(params[1])
        if lh < 0.05 or la < 0.05 or lh > 5.5 or la > 5.5:
            return 1e6
        mh, md, ma, mo = model_probs(lh, la)
        err = (
            2.0 * (mh - ph) ** 2
            + 1.5 * (md - pd) ** 2
            + 2.0 * (ma - pa) ** 2
            + 3.0 * (mo - po) ** 2
        )
        if ah_line is not None and ah_home is not None and ah_away is not None:
            p_ah_home, _ = _devig_2way(float(ah_home), float(ah_away))
            expected_diff = lh - la
            model_ah = 0.5 + (expected_diff + float(ah_line)) * 0.15
            model_ah = float(np.clip(model_ah, 0.05, 0.95))
            err += 2.0 * (model_ah - p_ah_home) ** 2
        return err

    total_goals_est = goals_line if po > 0.5 else max(0.5, goals_line - 0.5)
    diff_est = (ph - pa) * 1.5
    lh0 = max(0.3, (total_goals_est + diff_est) / 2.0)
    la0 = max(0.3, (total_goals_est - diff_est) / 2.0)

    result = minimize(
        loss,
        np.array([lh0, la0]),
        method="Nelder-Mead",
        options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 10000},
    )
    lh, la = float(result.x[0]), float(result.x[1])
    return max(0.1, lh), max(0.1, la)


def calibrate_lambdas(
    p_home: float,
    p_draw: float,
    p_away: float,
    p_over: float,
    p_under: float,
    *,
    ah_line: float | None = None,
    ah_home: float | None = None,
    ah_away: float | None = None,
    tt_home_over: float | None = None,
    tt_home_under: float | None = None,
    tt_away_over: float | None = None,
    tt_away_under: float | None = None,
    ou_curve: Sequence[tuple[float, float, float]] | None = None,
    goals_line: float = 2.5,
    input_is_odds: bool = True,
) -> tuple[float, float]:
    """Estima λ_home y λ_away desde team totals + curva O/U, o fallback conjunto."""
    if input_is_odds:
        ph, pd, pa = _devig_3way(p_home, p_draw, p_away)
        po, _ = _devig_2way(p_over, p_under)
    else:
        ph, pd, pa, po = float(p_home), float(p_draw), float(p_away), float(p_over)

    has_home_tt = _has_tt_side(tt_home_over, tt_home_under)
    has_away_tt = _has_tt_side(tt_away_over, tt_away_under)
    ou_curve_filtered = _filter_ou_curve_football(ou_curve)
    p_over_price = float(p_over) if input_is_odds else float(p_over)

    if has_home_tt or has_away_tt:
        lh, la = _calibrate_from_team_totals(
            tt_home_over=tt_home_over,
            tt_home_under=tt_home_under,
            tt_away_over=tt_away_over,
            tt_away_under=tt_away_under,
            ou_curve_filtered=ou_curve_filtered,
            goals_line=goals_line,
            p_over=po,
            p_over_price=p_over_price,
        )
        return _clamp_lambdas_realistic(lh, la)

    lh, la = _calibrate_lambdas_joint_fallback(
        ph,
        pd,
        pa,
        po,
        ah_line=ah_line,
        ah_home=ah_home,
        ah_away=ah_away,
        goals_line=goals_line,
    )
    return _clamp_lambdas_realistic(lh, la)


def calibrate_from_correct_score(
    cs_odds: Mapping[tuple[int, int], float],
    lambda_home: float,
    lambda_away: float,
    *,
    matrix_size: int = _CS_MATRIX_SIZE,
) -> np.ndarray:
    """Matriz de probabilidades desde mercado CS + relleno Poisson en celdas vacías."""
    raw_probs = {
        score: 1.0 / float(odd)
        for score, odd in cs_odds.items()
        if float(odd) > 1.0
    }
    if not raw_probs:
        raise ValueError("correct_score_odds vacío o inválido.")

    total = sum(raw_probs.values())
    cs_probs = {score: prob / total for score, prob in raw_probs.items()}

    matrix = np.zeros((matrix_size, matrix_size), dtype=float)
    for (h, a), prob in cs_probs.items():
        if 0 <= h < matrix_size and 0 <= a < matrix_size:
            matrix[h, a] = prob

    for h in range(matrix_size):
        for a in range(matrix_size):
            if matrix[h, a] == 0.0:
                matrix[h, a] = (
                    stats.poisson.pmf(h, lambda_home)
                    * stats.poisson.pmf(a, lambda_away)
                    * 0.1
                )

    matrix = np.clip(matrix, 0.0, None)
    matrix_sum = matrix.sum()
    if matrix_sum > _EPS:
        matrix /= matrix_sum
    return matrix


def _argmax_score_from_matrix(matrix: np.ndarray) -> tuple[int, int, str]:
    idx = int(np.argmax(matrix))
    h, a = np.unravel_index(idx, matrix.shape)
    return int(h), int(a), f"{h}-{a}"


def _value_ratio_score_from_matrix(matrix: np.ndarray) -> tuple[int, int, str]:
    best_score: tuple[int, int] | None = None
    best_ratio = -1.0
    for (h, a), base in BASE_SCORE_RATES.items():
        if h >= matrix.shape[0] or a >= matrix.shape[1] or base <= 0:
            continue
        ratio = float(matrix[h, a]) / base
        if ratio > best_ratio:
            best_ratio = ratio
            best_score = (h, a)
    if best_score is None:
        return _argmax_score_from_matrix(matrix)
    h, a = best_score
    return h, a, f"{h}-{a}"


class ProdeTopScore(TypedDict):
    score: str
    home_goals: int
    away_goals: int
    epv: float
    p_exact: float
    p_result: float


class ProdePrediction(TypedDict):
    score: tuple[int, int]
    score_str: str
    epv: float
    p_exact: float
    p_result: float
    epv_matrix: np.ndarray
    top5: list[ProdeTopScore]


def prode_result_prob(matrix: np.ndarray, home_goals: int, away_goals: int) -> float:
    """P(1X2) del resultado implícito en (home_goals, away_goals)."""
    p_home = float(np.sum(np.tril(matrix, -1)))
    p_draw = float(np.sum(np.diag(matrix)))
    p_away = float(np.sum(np.triu(matrix, 1)))
    if home_goals > away_goals:
        return p_home
    if home_goals == away_goals:
        return p_draw
    return p_away


def prode_epv(matrix: np.ndarray, home_goals: int, away_goals: int) -> float:
    """EPV = 3·P(exact) + 1·P(solo resultado) = 2·P(exact) + P(resultado)."""
    p_exact = float(matrix[home_goals, away_goals])
    return 2.0 * p_exact + prode_result_prob(matrix, home_goals, away_goals)


def prode_points_awarded(
    pred_home: int,
    pred_away: int,
    actual_home: int,
    actual_away: int,
) -> int:
    """Puntos prode: 3 exacto, 1 solo resultado, 0 si falla."""
    if pred_home == actual_home and pred_away == actual_away:
        return 3
    pred_out = (
        "H"
        if pred_home > pred_away
        else ("D" if pred_home == pred_away else "A")
    )
    actual_out = (
        "H"
        if actual_home > actual_away
        else ("D" if actual_home == actual_away else "A")
    )
    return 1 if pred_out == actual_out else 0


@dataclass(frozen=True)
class GlobalModelParams:
    """Parámetros globales Dixon-Coles + ZIP estimados por MLE."""

    rho: float
    pi: float
    log_likelihood: float
    neg_log_likelihood: float
    n_matches: int
    converged: bool
    message: str


@dataclass(frozen=True)
class PoissonFit:
    """Parámetros locales de un partido (λ, μ) con ρ y π globales."""

    lambda_home: float
    mu_away: float
    rho: float
    pi: float
    residual_sse: float
    calibrated_with_ah: bool = False
    ah_line: float | None = None
    calibrated_with_cs: bool = False
    calibrated_with_team_totals: bool = False


class ScorePredictor:
    """Dixon-Coles + Zero-Inflated Poisson con MLE global y calibración local.

    Modelo bivariante (x goles local, y visitante)::

        P_dc(x,y) = τ(x,y,ρ) · Poisson(x|λ) · Poisson(y|μ)

        P(x,y) = π·𝟙_{x=y=0} + (1-π)·P_dc(x,y)   si (x,y)=(0,0)
        P(x,y) = (1-π)·P_dc(x,y)                   en otro caso

    **Entrenamiento global** (``fit_global``): estima ρ y π por MLE sobre
    histórico con cuotas y resultados reales.

    **Predicción local** (``fit``): fija ρ y π globales y ajusta λ, μ para
    reproducir probabilidades de mercado 1X2 y Over/Under.
    """

    _global_params: GlobalModelParams | None = None

    def __init__(
        self,
        probs_1x2: Mapping[str, float],
        probs_ou: Mapping[str, float],
        *,
        max_goals: int = 5,
        goals_line: float = 2.5,
        rho: float | None = None,
        pi: float | None = None,
        ah_line: float | None = None,
        ah_home: float | None = None,
        ah_away: float | None = None,
        correct_score_odds: Mapping[tuple[int, int], float] | None = None,
        tt_home_over: float | None = None,
        tt_home_under: float | None = None,
        tt_away_over: float | None = None,
        tt_away_under: float | None = None,
        ou_curve: Sequence[tuple[float, float, float]] | None = None,
        global_params: GlobalModelParams | None = None,
        # Deprecado: se ignora si hay global_params o entrenamiento previo
        fit_rho: bool | None = None,
    ) -> None:
        self._validate_probs(probs_1x2, ("home", "draw", "away"), "1X2")
        self._validate_probs(probs_ou, ("over", "under"), "Over/Under")

        self.p_home = float(probs_1x2["home"])
        self.p_draw = float(probs_1x2["draw"])
        self.p_away = float(probs_1x2["away"])
        self.p_over = float(probs_ou["over"])
        self.p_under = float(probs_ou["under"])

        if max_goals < 1:
            raise ValueError("max_goals debe ser >= 1.")
        self.max_goals = max_goals
        self.goals_line = goals_line
        self.ah_line = float(ah_line) if ah_line is not None else None
        self.ah_home = float(ah_home) if ah_home is not None else None
        self.ah_away = float(ah_away) if ah_away is not None else None
        if correct_score_odds:
            self.correct_score_odds = {
                (int(h), int(a)): float(odd)
                for (h, a), odd in correct_score_odds.items()
                if float(odd) > 1.0
            }
        else:
            self.correct_score_odds = None

        self.tt_home_over = float(tt_home_over) if tt_home_over is not None else None
        self.tt_home_under = float(tt_home_under) if tt_home_under is not None else None
        self.tt_away_over = float(tt_away_over) if tt_away_over is not None else None
        self.tt_away_under = float(tt_away_under) if tt_away_under is not None else None
        self.ou_curve: list[tuple[float, float, float]] | None = (
            [(float(p), float(o), float(u)) for p, o, u in ou_curve]
            if ou_curve
            else None
        )

        gp = global_params or ScorePredictor._global_params
        if gp is not None:
            self.rho = gp.rho
            self.pi = gp.pi
        else:
            self.rho = float(rho) if rho is not None else _DEFAULT_RHO
            self.pi = float(pi) if pi is not None else 0.0

        if fit_rho is not None and fit_rho and gp is None and rho is None:
            # Compatibilidad: ajuste local de ρ solo si no hay globales
            self._fit_rho_locally = True
        else:
            self._fit_rho_locally = False

        self._fit: PoissonFit | None = None
        self._matrix: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Constructores
    # ------------------------------------------------------------------

    @classmethod
    def from_odds(
        cls,
        odds_1x2: Mapping[str, float],
        odds_ou: Mapping[str, float],
        **kwargs: object,
    ) -> ScorePredictor:
        return cls(
            remove_overround(odds_1x2),
            remove_overround(odds_ou),
            **kwargs,  # type: ignore[arg-type]
        )

    @classmethod
    def from_match_odds(
        cls,
        match: MatchOdds,
        *,
        overround_method: str = "multiplicative",
        **kwargs: object,
    ) -> ScorePredictor:
        fair = fair_probabilities(match, method=overround_method)
        return cls(fair["1x2"], fair["over_under"], **kwargs)  # type: ignore[arg-type]

    @classmethod
    def from_fair_probs(
        cls,
        fair: Mapping[str, Mapping[str, float]],
        **kwargs: object,
    ) -> ScorePredictor:
        return cls(fair["1x2"], fair["over_under"], **kwargs)  # type: ignore[arg-type]

    @classmethod
    def get_global_params(cls) -> GlobalModelParams | None:
        return cls._global_params

    @classmethod
    def fit_global(
        cls,
        df_historico: pd.DataFrame,
        *,
        odds_columns: Mapping[str, str] | None = None,
        goals_home_col: str = _HIST_GOALS_HOME,
        goals_away_col: str = _HIST_GOALS_AWAY,
        max_goals: int = 5,
        goals_line: float = 2.5,
        rho_bounds: tuple[float, float] = _RHO_BOUNDS,
        pi_bounds: tuple[float, float] = _PI_BOUNDS,
        recalibrate_rates: bool = False,
    ) -> GlobalModelParams:
        """MLE global de ρ y π sobre histórico football-data.co.uk.

        Para cada partido válido se calibran λᵢ, μᵢ desde las cuotas implícitas
        y se maximiza ∑ log P(xᵢ, yᵢ | λᵢ, μᵢ, ρ, π).

        Parameters
        ----------
        df_historico:
            DataFrame con goles finales y cuotas (B365*, FTHG, FTAG).
        recalibrate_rates:
            Si True, recalibra λᵢ, μᵢ en cada evaluación de (ρ, π) (más lento).
        """
        records = cls._prepare_historical_records(
            df_historico,
            odds_columns=odds_columns,
            goals_home_col=goals_home_col,
            goals_away_col=goals_away_col,
            max_goals=max_goals,
            goals_line=goals_line,
            rho_init=-0.13,
            pi_init=0.0,
        )
        if len(records) < 30:
            raise ValueError(
                f"Se requieren al menos 30 partidos válidos; hay {len(records)}."
            )

        def neg_log_likelihood(theta: np.ndarray) -> float:
            rho_v, pi_v = float(theta[0]), float(theta[1])
            if recalibrate_rates:
                total = 0.0
                for rec in records:
                    lam, mu = cls._calibrate_lambdas(
                        rec["targets"],
                        rho_v,
                        pi_v,
                        max_goals=max_goals,
                        goals_line=goals_line,
                    )
                    p = cls.bivariate_probability(
                        rec["x"],
                        rec["y"],
                        lam,
                        mu,
                        rho_v,
                        pi_v,
                        max_goals=max_goals,
                        normalized=True,
                    )
                    total -= cls._safe_log(p)
                return total

            ll = 0.0
            for rec in records:
                p = cls.bivariate_probability(
                    rec["x"],
                    rec["y"],
                    rec["lambda_home"],
                    rec["mu_away"],
                    rho_v,
                    pi_v,
                    max_goals=max_goals,
                    normalized=True,
                )
                ll += cls._safe_log(p)
            return -ll

        x0 = np.array([-0.13, 0.05])
        result = minimize(
            neg_log_likelihood,
            x0,
            method="L-BFGS-B",
            bounds=[rho_bounds, pi_bounds],
            options={"maxiter": 200, "ftol": 1e-8},
        )

        rho_opt, pi_opt = float(result.x[0]), float(result.x[1])
        nll = float(result.fun)
        params = GlobalModelParams(
            rho=rho_opt,
            pi=pi_opt,
            log_likelihood=-nll,
            neg_log_likelihood=nll,
            n_matches=len(records),
            converged=bool(result.success),
            message=str(result.message),
        )
        cls._global_params = params
        return params

    # ------------------------------------------------------------------
    # Probabilidad bivariante Dixon-Coles + ZIP
    # ------------------------------------------------------------------

    @staticmethod
    def dixon_coles_tau(
        home_goals: int,
        away_goals: int,
        lambda_home: float,
        mu_away: float,
        rho: float,
    ) -> float:
        if home_goals == 0 and away_goals == 0:
            return 1.0 - lambda_home * mu_away * rho
        if home_goals == 0 and away_goals == 1:
            return 1.0 + lambda_home * rho
        if home_goals == 1 and away_goals == 0:
            return 1.0 + mu_away * rho
        if home_goals == 1 and away_goals == 1:
            return 1.0 - rho
        return 1.0

    @classmethod
    def poisson_dc_mass(
        cls,
        home_goals: int,
        away_goals: int,
        lambda_home: float,
        mu_away: float,
        rho: float,
    ) -> float:
        """Masa Poisson independiente × factor τ (sin ZIP)."""
        base = (
            stats.poisson.pmf(home_goals, lambda_home)
            * stats.poisson.pmf(away_goals, mu_away)
        )
        tau = cls.dixon_coles_tau(home_goals, away_goals, lambda_home, mu_away, rho)
        return max(float(base * tau), 0.0)

    @classmethod
    def bivariate_probability(
        cls,
        home_goals: int,
        away_goals: int,
        lambda_home: float,
        mu_away: float,
        rho: float,
        pi: float,
        *,
        max_goals: int = 5,
        normalized: bool = True,
    ) -> float:
        """P(x,y) con capa ZIP en (0,0) y corrección Dixon-Coles."""
        if home_goals < 0 or away_goals < 0:
            return _EPS
        if home_goals > max_goals or away_goals > max_goals:
            return _EPS

        if normalized:
            matrix = cls.build_probability_matrix(
                lambda_home,
                mu_away,
                rho,
                pi,
                max_goals=max_goals,
                normalized=True,
            )
            return max(float(matrix[home_goals, away_goals]), _EPS)

        pi = float(np.clip(pi, 0.0, 1.0 - _EPS))
        mass_dc = cls.poisson_dc_mass(
            home_goals, away_goals, lambda_home, mu_away, rho
        )
        if home_goals == 0 and away_goals == 0:
            return max(pi + (1.0 - pi) * mass_dc, 0.0)
        return max((1.0 - pi) * mass_dc, 0.0)

    @classmethod
    def build_probability_matrix(
        cls,
        lambda_home: float,
        mu_away: float,
        rho: float,
        pi: float,
        *,
        max_goals: int = 5,
        normalized: bool = True,
    ) -> np.ndarray:
        """Matriz (max_goals+1)² de P(x,y) con Poisson + Dixon-Coles + ZIP."""
        n = max_goals + 1
        pi = float(np.clip(pi, 0.0, 1.0 - _EPS))

        ph_vec = stats.poisson.pmf(np.arange(n), lambda_home)
        pa_vec = stats.poisson.pmf(np.arange(n), mu_away)
        matrix = np.outer(ph_vec, pa_vec)

        for i, j in ((0, 0), (1, 0), (0, 1), (1, 1)):
            matrix[i, j] *= cls.dixon_coles_tau(
                i, j, lambda_home, mu_away, rho
            )

        matrix = np.clip(matrix, 0.0, None)
        total = matrix.sum()
        if total > _EPS:
            matrix /= total

        if pi > 0.0:
            adjusted = matrix.copy()
            adjusted[0, 0] = pi + (1.0 - pi) * matrix[0, 0]
            mask = np.ones((n, n), dtype=bool)
            mask[0, 0] = False
            adjusted[mask] = (1.0 - pi) * matrix[mask]
            matrix = adjusted

        matrix = np.clip(matrix, 0.0, None)
        if normalized:
            total = matrix.sum()
            if total > _EPS:
                matrix /= total
        return matrix

    @staticmethod
    def _safe_log(probability: float) -> float:
        return float(np.log(max(probability, _EPS)))

    # ------------------------------------------------------------------
    # Calibración local λ, μ
    # ------------------------------------------------------------------

    def fit(self) -> PoissonFit:
        """Calibra λ y μ para coincidir con mercados (ρ y π fijos globales)."""
        if self._fit is not None:
            return self._fit

        targets = np.array(
            [self.p_home, self.p_draw, self.p_away, self.p_over, self.p_under]
        )
        rho, pi = self.rho, self.pi
        use_ah = (
            self.ah_line is not None
            and self.ah_home is not None
            and self.ah_home > 0
            and self.ah_away is not None
            and self.ah_away > 0
        )
        calibrated_with_ah = use_ah
        ah_line_used: float | None = self.ah_line if use_ah else None
        use_cs = bool(self.correct_score_odds)
        use_tt = _has_tt_side(self.tt_home_over, self.tt_home_under) or _has_tt_side(
            self.tt_away_over, self.tt_away_under
        )

        if self._fit_rho_locally:
            x0 = np.array([1.35, 1.15, rho, pi])
            result = minimize(
                lambda x: self._market_sse(x[0], x[1], x[2], x[3], targets),
                x0,
                method="L-BFGS-B",
                bounds=[_LAMBDA_BOUNDS, _LAMBDA_BOUNDS, _RHO_BOUNDS, _PI_BOUNDS],
            )
            lam, mu, rho, pi = (
                float(result.x[0]),
                float(result.x[1]),
                float(result.x[2]),
                float(result.x[3]),
            )
            sse = float(result.fun)
        else:
            lam, mu = calibrate_lambdas(
                self.p_home,
                self.p_draw,
                self.p_away,
                self.p_over,
                self.p_under,
                ah_line=self.ah_line if use_ah else None,
                ah_home=self.ah_home if use_ah else None,
                ah_away=self.ah_away if use_ah else None,
                tt_home_over=self.tt_home_over,
                tt_home_under=self.tt_home_under,
                tt_away_over=self.tt_away_over,
                tt_away_under=self.tt_away_under,
                ou_curve=self.ou_curve,
                goals_line=self.goals_line,
                input_is_odds=False,
            )
            sse = self._market_sse(lam, mu, rho, pi, targets)

        self.rho, self.pi = rho, pi
        self._fit = PoissonFit(
            lambda_home=lam,
            mu_away=mu,
            rho=rho,
            pi=pi,
            residual_sse=sse,
            calibrated_with_ah=calibrated_with_ah,
            ah_line=ah_line_used,
            calibrated_with_cs=use_cs,
            calibrated_with_team_totals=use_tt,
        )
        self._matrix = None
        return self._fit

    def _score_matrix_from_cs(self, fit: PoissonFit) -> np.ndarray:
        if not self.correct_score_odds:
            raise ValueError("correct_score_odds no disponible.")
        cs_matrix = calibrate_from_correct_score(
            self.correct_score_odds,
            fit.lambda_home,
            fit.mu_away,
            matrix_size=max(self.max_goals + 1, _CS_MATRIX_SIZE),
        )
        n = self.max_goals + 1
        if cs_matrix.shape[0] >= n:
            return cs_matrix[:n, :n] / cs_matrix[:n, :n].sum()
        return cs_matrix

    @classmethod
    def _calibrate_lambdas(
        cls,
        targets: np.ndarray,
        rho: float,
        pi: float,
        *,
        max_goals: int,
        goals_line: float,
        odds_1x2: Mapping[str, float] | None = None,
        odds_ou: Mapping[str, float] | None = None,
    ) -> tuple[float, float]:
        if odds_1x2 is not None and odds_ou is not None:
            return calibrate_lambdas(
                odds_1x2["home"],
                odds_1x2["draw"],
                odds_1x2["away"],
                odds_ou["over"],
                odds_ou["under"],
                goals_line=goals_line,
                input_is_odds=True,
            )
        return calibrate_lambdas(
            float(targets[0]),
            float(targets[1]),
            float(targets[2]),
            float(targets[3]),
            float(targets[4]),
            goals_line=goals_line,
            input_is_odds=False,
        )

    def _market_sse(
        self,
        lambda_home: float,
        mu_away: float,
        rho: float,
        pi: float,
        targets: np.ndarray,
    ) -> float:
        return self._market_sse_static(
            lambda_home,
            mu_away,
            rho,
            pi,
            targets,
            max_goals=self.max_goals,
            goals_line=self.goals_line,
        )

    @classmethod
    def _market_sse_static(
        cls,
        lambda_home: float,
        mu_away: float,
        rho: float,
        pi: float,
        targets: np.ndarray,
        *,
        max_goals: int,
        goals_line: float,
    ) -> float:
        try:
            modeled = cls._market_probabilities_static(
                lambda_home,
                mu_away,
                rho,
                pi,
                max_goals=max_goals,
                goals_line=goals_line,
            )
        except (ValueError, FloatingPointError):
            return 1e6
        return float(np.sum((modeled - targets) ** 2))

    @classmethod
    def _market_probabilities_static(
        cls,
        lambda_home: float,
        mu_away: float,
        rho: float,
        pi: float,
        *,
        max_goals: int,
        goals_line: float,
    ) -> np.ndarray:
        matrix = cls.build_probability_matrix(
            lambda_home, mu_away, rho, pi, max_goals=max_goals, normalized=True
        )
        if matrix.sum() <= _EPS:
            raise ValueError("Matriz degenerada.")

        p_home = p_draw = p_away = p_over = p_under = 0.0
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                p = matrix[i, j]
                if i > j:
                    p_home += p
                elif i == j:
                    p_draw += p
                else:
                    p_away += p
                if i + j > goals_line:
                    p_over += p
                else:
                    p_under += p

        return np.array([p_home, p_draw, p_away, p_over, p_under])

    # ------------------------------------------------------------------
    # Salidas (compatibles Streamlit / backtest)
    # ------------------------------------------------------------------

    def score_matrix(self, *, normalized: bool = True) -> np.ndarray:
        fit = self.fit()
        if fit.calibrated_with_cs and self.correct_score_odds:
            matrix = self._score_matrix_from_cs(fit)
            if normalized:
                total = matrix.sum()
                if total > _EPS:
                    matrix = matrix / total
            self._matrix = matrix
            return matrix

        matrix = self.build_probability_matrix(
            fit.lambda_home,
            fit.mu_away,
            fit.rho,
            fit.pi,
            max_goals=self.max_goals,
            normalized=normalized,
        )
        self._matrix = matrix
        return matrix

    def score_matrix_df(self) -> pd.DataFrame:
        m = self.score_matrix()
        labels = [str(i) for i in range(self.max_goals + 1)]
        return pd.DataFrame(m, index=labels, columns=labels)

    def top_exact_scores(self, n: int = 5) -> pd.DataFrame:
        if n < 1:
            raise ValueError("n debe ser >= 1.")

        matrix = self.score_matrix()
        rows: list[dict[str, object]] = []

        for i in range(self.max_goals + 1):
            for j in range(self.max_goals + 1):
                prob = float(matrix[i, j])
                if prob <= 0:
                    continue
                rows.append(
                    {
                        "score": f"{i}-{j}",
                        "home_goals": i,
                        "away_goals": j,
                        "probability": prob,
                        "percentage": round(prob * 100.0, 2),
                    }
                )

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        return (
            df.sort_values("probability", ascending=False)
            .head(n)
            .reset_index(drop=True)
        )

    def most_likely_exact_score(
        self,
        *,
        method: str = "argmax",
        rho: float | None = None,
    ) -> tuple[int, int, str]:
        """Score más probable (argmax) o por ratio modelo/prior (value_ratio)."""
        fit = self.fit()
        rho_use = float(rho) if rho is not None else fit.rho
        matrix = self.build_probability_matrix(
            fit.lambda_home,
            fit.mu_away,
            rho_use,
            fit.pi,
            max_goals=self.max_goals,
            normalized=True,
        )
        if method == "value_ratio":
            return _value_ratio_score_from_matrix(matrix)
        return _argmax_score_from_matrix(matrix)

    def most_likely_exact_score_value(
        self,
        *,
        rho: float | None = None,
    ) -> tuple[int, int, str]:
        """Score con mayor ratio model_prob / base_rate histórico."""
        return self.most_likely_exact_score(method="value_ratio", rho=rho)

    def predict_for_prode(self) -> ProdePrediction:
        """Score con mayor EPV (Expected Prode Value): 2·P(exact) + P(resultado 1X2)."""
        matrix = self.score_matrix()
        n = matrix.shape[0]
        p_home = float(np.sum(np.tril(matrix, -1)))
        p_draw = float(np.sum(np.diag(matrix)))
        p_away = float(np.sum(np.triu(matrix, 1)))

        def result_prob(h: int, a: int) -> float:
            if h > a:
                return p_home
            if h == a:
                return p_draw
            return p_away

        epv_matrix = np.zeros((n, n), dtype=float)
        best_score: tuple[int, int] | None = None
        best_epv = -1.0

        for h in range(n):
            for a in range(n):
                epv = 2.0 * float(matrix[h, a]) + result_prob(h, a)
                epv_matrix[h, a] = epv
                if epv > best_epv:
                    best_epv = epv
                    best_score = (h, a)

        if best_score is None:
            best_score = (0, 0)
            best_epv = float(epv_matrix[0, 0])

        ranked: list[tuple[tuple[int, int], float]] = []
        for h in range(n):
            for a in range(n):
                ranked.append(((h, a), float(epv_matrix[h, a])))
        ranked.sort(key=lambda item: -item[1])

        top5: list[ProdeTopScore] = []
        for (h, a), epv in ranked[:5]:
            top5.append(
                {
                    "score": f"{h}-{a}",
                    "home_goals": h,
                    "away_goals": a,
                    "epv": epv,
                    "p_exact": float(matrix[h, a]),
                    "p_result": result_prob(h, a),
                }
            )

        bh, ba = best_score
        return {
            "score": best_score,
            "score_str": f"{bh}-{ba}",
            "epv": best_epv,
            "p_exact": float(matrix[bh, ba]),
            "p_result": result_prob(bh, ba),
            "epv_matrix": epv_matrix,
            "top5": top5,
        }

    @property
    def fitted_params(self) -> PoissonFit:
        return self.fit()

    # ------------------------------------------------------------------
    # Histórico para MLE
    # ------------------------------------------------------------------

    @classmethod
    def _prepare_historical_records(
        cls,
        df: pd.DataFrame,
        *,
        odds_columns: Mapping[str, str] | None,
        goals_home_col: str,
        goals_away_col: str,
        max_goals: int,
        goals_line: float,
        rho_init: float,
        pi_init: float,
    ) -> list[dict[str, Any]]:
        cols = {**_HIST_ODDS_COLS, **(odds_columns or {})}
        records: list[dict[str, Any]] = []

        for _, row in df.iterrows():
            try:
                odds_1x2 = {
                    "home": float(row[cols["home"]]),
                    "draw": float(row[cols["draw"]]),
                    "away": float(row[cols["away"]]),
                }
                odds_ou = {
                    "over": float(row[cols["over"]]),
                    "under": float(row[cols["under"]]),
                }
            except (KeyError, TypeError, ValueError):
                continue

            if not all(v > 1.0 for v in (*odds_1x2.values(), *odds_ou.values())):
                continue

            try:
                x = int(row[goals_home_col])
                y = int(row[goals_away_col])
            except (KeyError, TypeError, ValueError):
                continue
            if x < 0 or y < 0:
                continue

            fair_1x2 = remove_overround(odds_1x2)
            fair_ou = remove_overround(odds_ou)
            targets = np.array(
                [
                    fair_1x2["home"],
                    fair_1x2["draw"],
                    fair_1x2["away"],
                    fair_ou["over"],
                    fair_ou["under"],
                ]
            )
            lam, mu = cls._calibrate_lambdas(
                targets,
                rho_init,
                pi_init,
                max_goals=max_goals,
                goals_line=goals_line,
                odds_1x2=odds_1x2,
                odds_ou=odds_ou,
            )
            records.append(
                {
                    "x": x,
                    "y": y,
                    "lambda_home": lam,
                    "mu_away": mu,
                    "targets": targets,
                }
            )
        return records

    @staticmethod
    def _validate_probs(
        probs: Mapping[str, float],
        keys: Sequence[str],
        label: str,
    ) -> None:
        missing = [k for k in keys if k not in probs]
        if missing:
            raise KeyError(f"Faltan claves {missing} en probabilidades {label}.")
        total = 0.0
        for k in keys:
            p = float(probs[k])
            if not 0.0 < p < 1.0:
                raise ValueError(
                    f"Probabilidad {label}['{k}'] inválida: {p}. Debe estar en (0, 1)."
                )
            total += p
        if abs(total - 1.0) > 0.02:
            raise ValueError(
                f"Las probabilidades {label} deben sumar ~1 (suma={total:.4f})."
            )


def _predict_exact_from_fitted(
    fit: PoissonFit,
    *,
    rho: float,
    max_goals: int,
    method: str = "argmax",
) -> tuple[int, int, str]:
    matrix = ScorePredictor.build_probability_matrix(
        fit.lambda_home,
        fit.mu_away,
        rho,
        fit.pi,
        max_goals=max_goals,
        normalized=True,
    )
    if method == "value_ratio":
        return _value_ratio_score_from_matrix(matrix)
    return _argmax_score_from_matrix(matrix)


def find_optimal_rho(
    matches_df: pd.DataFrame,
    *,
    odds_columns: Mapping[str, str] | None = None,
    goals_home_col: str = _HIST_GOALS_HOME,
    goals_away_col: str = _HIST_GOALS_AWAY,
    max_goals: int = 5,
    goals_line: float = 2.5,
    rho_bounds: tuple[float, float] = (-0.5, 0.3),
    method: str = "argmax",
) -> float:
    """Busca ρ que maximiza Exact Score Accuracy sobre ``matches_df``."""
    from scipy.optimize import minimize_scalar

    cols = {**_PINNACLE_ODDS_COLS, **(odds_columns or {})}
    cached: list[tuple[PoissonFit, int, int]] = []

    for _, row in matches_df.iterrows():
        try:
            odds_1x2 = {
                "home": float(row[cols["home"]]),
                "draw": float(row[cols["draw"]]),
                "away": float(row[cols["away"]]),
            }
            odds_ou = {
                "over": float(row[cols["over"]]),
                "under": float(row[cols["under"]]),
            }
            fthg = int(row[goals_home_col])
            ftag = int(row[goals_away_col])
        except (KeyError, TypeError, ValueError):
            continue
        if not all(v > 1.0 for v in (*odds_1x2.values(), *odds_ou.values())):
            continue
        if fthg < 0 or ftag < 0:
            continue
        try:
            predictor = ScorePredictor.from_odds(
                odds_1x2,
                odds_ou,
                max_goals=max_goals,
                goals_line=goals_line,
            )
            fit = predictor.fit()
        except (ValueError, KeyError):
            continue
        cached.append((fit, fthg, ftag))

    if not cached:
        return _DEFAULT_RHO

    def neg_accuracy(rho: float) -> float:
        correct = 0
        for fit, fthg, ftag in cached:
            pred_h, pred_a, _ = _predict_exact_from_fitted(
                fit,
                rho=float(rho),
                max_goals=max_goals,
                method=method,
            )
            if pred_h == fthg and pred_a == ftag:
                correct += 1
        return -correct / len(cached)

    result = minimize_scalar(
        neg_accuracy,
        bounds=rho_bounds,
        method="bounded",
    )
    return float(result.x)
