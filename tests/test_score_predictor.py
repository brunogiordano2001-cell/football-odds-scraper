import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from football_odds_scraper.score_predictor import (
    ScorePredictor,
    _calibrate_from_both_curves,
    _lambda_from_team_total,
    _lambda_from_tt,
    _model_p_ah_home,
    _poisson_matrix,
    calibrate_from_correct_score,
    calibrate_lambdas,
    find_optimal_rho,
)

FAIR_1X2 = {"home": 0.45, "draw": 0.27, "away": 0.28}
FAIR_OU = {"over": 0.52, "under": 0.48}

FIXTURE = Path(__file__).parent / "fixtures" / "sample_football_data.csv"


def test_lambda_from_team_total():
    lam = _lambda_from_team_total(1.35, 3.20)
    assert 0.5 < lam < 3.0


def test_lambda_from_tt_line_15():
    lam_05 = _lambda_from_tt(1.35, 3.20, 0.5)
    lam_15 = _lambda_from_tt(1.85, 2.05, 1.5)
    assert lam_05 is not None
    assert lam_15 is not None
    assert lam_15 > lam_05


def test_calibrate_lambdas_with_team_totals():
    lh, la = calibrate_lambdas(
        2.0,
        3.5,
        4.0,
        1.9,
        2.0,
        tt_home_over=1.35,
        tt_home_under=3.20,
        tt_away_over=1.55,
        tt_away_under=2.45,
        ou_curve=[
            (1.5, 1.485, 2.73),
            (2.0, 1.869, 2.04),
            (2.5, 1.95, 1.90),
            (3.0, 2.10, 1.75),
        ],
    )
    assert lh > 0.1
    assert la > 0.1


def test_score_predictor_with_team_totals():
    pred = ScorePredictor.from_odds(
        {"home": 2.0, "draw": 3.5, "away": 4.0},
        {"over": 1.9, "under": 2.0},
        tt_home_over=1.35,
        tt_home_under=3.20,
        tt_away_over=1.55,
        tt_away_under=2.45,
        ou_curve=[(1.5, 1.485, 2.73), (2.0, 1.869, 2.04), (2.5, 1.95, 1.90)],
    )
    fit = pred.fit()
    assert fit.calibrated_with_team_totals is True
    assert fit.lambda_home >= 0.1
    assert fit.mu_away >= 0.1


def test_calibrate_lambdas_fallback_without_team_totals():
    lh_joint, la_joint = calibrate_lambdas(2.0, 3.5, 4.0, 1.9, 2.0)
    pred = ScorePredictor.from_odds(
        {"home": 2.0, "draw": 3.5, "away": 4.0},
        {"over": 1.9, "under": 2.0},
    )
    fit = pred.fit()
    assert fit.calibrated_with_team_totals is False
    assert abs(fit.lambda_home - lh_joint) < 0.05
    assert abs(fit.mu_away - la_joint) < 0.05


def test_poisson_total_over_prob_lines():
    from football_odds_scraper.score_predictor import _poisson_total_over_prob

    lam = 3.5
    assert _poisson_total_over_prob(lam, 2.5) == _poisson_total_over_prob(lam, 2.0 + 0.5)
    assert _poisson_total_over_prob(lam, 3.5) > _poisson_total_over_prob(lam, 4.0)


def test_curacao_tt_lambda_parsing():
    """Curazao TT Over 0.5 @ 2.670 / Under 0.5 @ 1.395 → λ ≈ 0.42."""
    from football_odds_scraper.score_predictor import _lambda_from_team_total

    lam_away = _lambda_from_team_total(2.670, 1.395)
    assert 0.35 < lam_away < 0.55, f"TT parsing bug: λ_away={lam_away:.3f}"


def test_germany_curacao_partial_team_totals(capsys):
    """Sin home/0.5: λ_away desde TT visitante, λ_home = λ_total − λ_away."""
    ou_curve = [
        (3.0, 1.75, 2.05),
        (3.5, 1.85, 1.95),
        (4.0, 2.05, 1.80),
        (4.25, 1.925, 1.909),
        (4.5, 2.15, 1.72),
    ]
    lh, la = calibrate_lambdas(
        1.05,
        15.0,
        25.0,
        1.925,
        1.909,
        tt_home_over=None,
        tt_home_under=None,
        tt_away_over=2.670,
        tt_away_under=1.395,
        ou_curve=ou_curve,
        goals_line=4.25,
    )
    assert 0.35 < la < 0.55, f"λ_away={la:.3f} debería ser ~0.42"
    assert lh > 3.5, f"λ_home={lh:.3f} debería ser ~3.8+"
    assert abs((lh + la) - 4.25) < 0.5

    pred = ScorePredictor.from_odds(
        {"home": 1.05, "draw": 15.0, "away": 25.0},
        {"over": 1.925, "under": 1.909},
        goals_line=4.25,
        tt_away_over=2.670,
        tt_away_under=1.395,
        ou_curve=ou_curve,
    )
    prode = pred.predict_for_prode()
    top3 = prode["top5"][:3]
    assert all(
        int(s.split("-")[1]) == 0 for s in (entry["score"] for entry in top3)
    ), f"Top 3 no deberían incluir goles visitante: {top3}"


def test_calibrate_lambdas_ignores_contaminated_ou_curve(capsys):
    """Líneas 8+ de otros deportes no deben inflar λ_total."""
    ou_curve = [
        (2.5, 1.55, 2.45),
        (3.0, 1.85, 2.00),
        (3.5, 2.15, 1.72),
        (8.0, 1.90, 1.90),
        (9.0, 2.10, 1.75),
    ]
    lh, la = calibrate_lambdas(
        1.55,
        4.20,
        5.50,
        2.15,
        1.72,
        tt_home_over=1.12,
        tt_home_under=6.50,
        tt_away_over=1.18,
        tt_away_under=5.00,
        ou_curve=ou_curve,
        goals_line=3.5,
    )
    assert lh + la <= 5.5
    captured = capsys.readouterr()
    assert "9.0" not in captured.out
    assert "8.0" not in captured.out


def test_strong_favorite_team_total_lambda():
    """PSH≤1.30 + TT home over 0.5 muy bajo → λ_home alto (Alemania vs Curazao)."""
    lh, la = calibrate_lambdas(
        1.20,
        7.00,
        5.00,
        1.85,
        2.00,
        tt_home_over=1.10,
        tt_home_under=17.00,
        tt_away_over=2.20,
        tt_away_under=1.65,
        goals_line=2.5,
    )
    assert lh >= 2.8
    assert la < lh


def test_calibrate_lambdas_high_scoring_with_ou_curve(capsys):
    """Partido alto scoring: λ_total debe acercarse a la línea principal O/U."""
    ou_curve = [
        (2.5, 1.55, 2.45),
        (3.0, 1.85, 2.00),
        (3.5, 2.15, 1.72),
        (4.0, 2.55, 1.52),
        (4.5, 3.05, 1.38),
    ]
    lh, la = calibrate_lambdas(
        1.55,
        4.20,
        5.50,
        2.15,
        1.72,
        tt_home_over=1.12,
        tt_home_under=6.50,
        tt_away_over=1.18,
        tt_away_under=5.00,
        ou_curve=ou_curve,
        goals_line=3.5,
    )
    total = lh + la
    assert total >= 3.0
    assert total <= 5.5
    assert abs(total - 3.5) <= 0.6
    captured = capsys.readouterr()
    assert "=== CALIBRACIÓN ===" in captured.out
    assert "λ final:" in captured.out


def test_predict_for_prode_prefers_result_over_1_1():
    """Con favorito claro, EPV elige score de victoria sobre 1-1 aunque sea menos probable."""
    pred = ScorePredictor.from_odds(
        {"home": 1.45, "draw": 4.50, "away": 7.00},
        {"over": 1.65, "under": 2.25},
        rho=-0.13,
    )
    prode = pred.predict_for_prode()
    argmax_h, argmax_a, _ = pred.most_likely_exact_score(method="argmax")
    assert prode["score_str"] != "1-1" or (argmax_h, argmax_a) == (1, 1)
    assert prode["epv"] >= 2 * prode["p_exact"] + prode["p_result"] - 1e-9
    assert len(prode["top5"]) == 5
    assert prode["top5"][0]["score"] == prode["score_str"]
    assert len(prode["top3"]) == 3
    assert len(prode["top3_pure"]) == 3
    assert prode["top3_coverage"] > 0


def test_top3_diverse_includes_second_outcome():
    """Con draw como 2do resultado (>20%), el top 3 diversificado lo cubre."""
    pred = ScorePredictor.from_odds(
        {"home": 1.45, "draw": 4.50, "away": 7.00},
        {"over": 1.65, "under": 2.25},
        rho=-0.13,
    )
    prode = pred.predict_for_prode()
    matrix = pred.score_matrix()
    p_home = float(np.sum(np.tril(matrix, -1)))
    p_draw = float(np.sum(np.diag(matrix)))
    p_away = float(np.sum(np.triu(matrix, 1)))
    second_prob = sorted([p_home, p_draw, p_away], reverse=True)[1]

    top3 = prode["top3"]
    top3_pure = prode["top3_pure"]
    assert len(top3) == 3
    assert top3[0]["score"] == prode["score_str"]
    assert abs(prode["top3_coverage"] - sum(e["p_exact"] for e in top3)) < 1e-9

    if second_prob > 0.20:
        pure_outcomes = {e["outcome"] for e in top3_pure}
        diverse_outcomes = {e["outcome"] for e in top3}
        if len(pure_outcomes) == 1:
            assert len(diverse_outcomes) >= 2
        assert any(e["is_coverage"] for e in top3)


def test_prode_edge_vs_base():
    from football_odds_scraper.score_predictor import BASE_SCORE_RATES, prode_edge_vs_base

    p_model = 0.162
    base = BASE_SCORE_RATES[(2, 0)]
    edge = prode_edge_vs_base(2, 0, p_model)
    assert abs(edge - (p_model / base - 1.0)) < 1e-9
    assert edge > 1.0  # +178% aprox vs histórico 5.8%


def test_prode_epv_formula():
    pred = ScorePredictor(FAIR_1X2, FAIR_OU)
    matrix = pred.score_matrix()
    prode = pred.predict_for_prode()
    h, a = prode["score"]
    expected = 2 * float(matrix[h, a]) + prode["p_result"]
    assert abs(prode["epv"] - expected) < 1e-9


def test_prode_points_awarded():
    from football_odds_scraper.score_predictor import prode_points_awarded

    assert prode_points_awarded(2, 0, 2, 0) == 3
    assert prode_points_awarded(2, 0, 1, 0) == 1
    assert prode_points_awarded(2, 0, 0, 1) == 0


def test_sensitivity_analysis_in_predict_for_prode():
    pred = ScorePredictor.from_odds(
        {"home": 1.45, "draw": 4.50, "away": 7.00},
        {"over": 1.65, "under": 2.25},
        rho=-0.13,
    )
    prode = pred.predict_for_prode()
    sens = prode["sensitivity"]

    assert "k_breaks" in sens
    assert "score_at_k2" in sens
    assert "score_at_inf" in sens
    assert "robustness" in sens
    assert sens["score_at_k2"] == prode["score"]
    assert isinstance(sens["robustness"], float)
    assert sens["robustness_label"] in {"🟢 Robusta", "🟡 Moderada", "🔴 Frágil"}


def test_sensitivity_analysis_k2_matches_argmax_at_infinity():
    pred = ScorePredictor(FAIR_1X2, FAIR_OU)
    sens = pred.sensitivity_analysis()
    h, a, _ = pred.most_likely_exact_score()
    assert sens["score_at_inf"] == (h, a)
    pred = ScorePredictor(FAIR_1X2, FAIR_OU, rho=-0.13, pi=0.05)
    fit = pred.fit()
    assert fit.lambda_home > 0
    assert fit.mu_away > 0
    assert fit.pi >= 0
    assert fit.residual_sse >= 0
    assert fit.calibrated_with_ah is False


def test_most_likely_exact_score_value_differs_from_argmax():
    odds_1x2 = {"home": 1.45, "draw": 4.50, "away": 7.00}
    odds_ou = {"over": 1.65, "under": 2.25}
    pred = ScorePredictor.from_odds(odds_1x2, odds_ou)
    argmax = pred.most_likely_exact_score(method="argmax")
    value = pred.most_likely_exact_score_value()
    assert argmax[2] != value[2] or argmax == value


def test_find_optimal_rho_on_tiny_sample():
    rows = []
    for i in range(20):
        rows.append(
            {
                "PSH": 2.0 + (i % 3) * 0.3,
                "PSD": 3.4,
                "PSA": 3.5,
                "P>2.5": 1.9,
                "P<2.5": 1.95,
                "FTHG": i % 4,
                "FTAG": (i + 1) % 3,
            }
        )
    df = pd.DataFrame(rows)
    rho = find_optimal_rho(
        df,
        odds_columns={
            "home": "PSH",
            "draw": "PSD",
            "away": "PSA",
            "over": "P>2.5",
            "under": "P<2.5",
        },
    )
    assert -0.5 <= rho <= 0.3


def test_calibrate_from_correct_score():
    cs_odds = {(1, 0): 7.5, (0, 0): 9.0, (1, 1): 6.5, (2, 1): 12.0}
    matrix = calibrate_from_correct_score(cs_odds, 1.4, 1.1, matrix_size=6)
    assert matrix.shape == (6, 6)
    assert abs(matrix.sum() - 1.0) < 1e-6
    assert matrix[1, 0] > matrix[2, 2]


def test_score_predictor_with_correct_score():
    cs = {(1, 0): 7.5, (0, 0): 9.0, (1, 1): 6.5, (2, 1): 12.0, (0, 1): 11.0}
    pred = ScorePredictor.from_odds(
        {"home": 2.0, "draw": 3.5, "away": 4.0},
        {"over": 1.9, "under": 2.0},
        correct_score_odds=cs,
    )
    fit = pred.fit()
    assert fit.calibrated_with_cs is True
    top = pred.top_exact_scores(1).iloc[0]["score"]
    assert top in {"1-0", "0-0", "1-1", "2-1", "0-1"}


def test_fit_with_asian_handicap():
    pred = ScorePredictor(
        FAIR_1X2,
        FAIR_OU,
        rho=-0.13,
        pi=0.05,
        ah_line=-0.25,
        ah_home=1.92,
        ah_away=1.98,
    )
    fit = pred.fit()
    assert fit.calibrated_with_ah is True
    assert fit.ah_line == -0.25
    assert fit.lambda_home >= 0.1
    assert fit.mu_away >= 0.1
    assert fit.lambda_home > fit.mu_away


def test_model_p_ah_home_favorite_covers():
    matrix = _poisson_matrix(1.8, 0.9)
    p_minus_half = _model_p_ah_home(matrix, -0.5)
    p_minus_one = _model_p_ah_home(matrix, -1.0)
    assert 0.0 < p_minus_half < 1.0
    assert p_minus_half > p_minus_one


def test_calibrate_from_both_curves(capsys):
    lh_ref, la_ref = 1.6, 1.1
    matrix = _poisson_matrix(lh_ref, la_ref)
    ou_curve = []
    for line in (2.0, 2.5, 3.0):
        lam_total = lh_ref + la_ref
        from scipy import stats

        p_over = float(stats.poisson.sf(int(line), lam_total))
        p_under = max(1.0 - p_over, 0.05)
        ou_curve.append((line, 1.05 / p_over, 1.05 / p_under))
    ah_curve = []
    for line in (-0.5, -0.25, 0.25):
        p_home = _model_p_ah_home(matrix, line)
        p_away = max(1.0 - p_home, 0.05)
        ah_curve.append((line, 1.05 / p_home, 1.05 / p_away))

    result = _calibrate_from_both_curves(ou_curve, ah_curve, min_points=3)
    assert result is not None
    lh, la = result
    assert abs(lh - lh_ref) < 0.35
    assert abs(la - la_ref) < 0.35
    captured = capsys.readouterr()
    assert "AH curve: 3 puntos" in captured.out
    assert "λ_diff desde AH:" in captured.out


def test_calibrate_lambdas_with_ah_curve():
    lh_ref, la_ref = 1.7, 1.0
    matrix = _poisson_matrix(lh_ref, la_ref)
    ah_curve = []
    for line in (-1.0, -0.5, -0.25, 0.0, 0.25):
        p_home = _model_p_ah_home(matrix, line)
        p_away = max(1.0 - p_home, 0.05)
        ah_curve.append((line, 1.05 / p_home, 1.05 / p_away))

    lh, la = calibrate_lambdas(
        1.55,
        4.20,
        5.50,
        2.15,
        1.72,
        ah_curve=ah_curve,
        goals_line=2.5,
    )
    assert lh > la
    assert 0.1 <= lh <= 4.0
    assert 0.1 <= la <= 4.0


def test_calibration_does_not_default_to_1_1():
    cases = [
        {"home": 1.45, "draw": 4.50, "away": 7.00, "over": 1.65, "under": 2.25},
        {"home": 4.50, "draw": 3.60, "away": 1.75, "over": 2.10, "under": 1.75},
        {"home": 2.80, "draw": 3.20, "away": 2.60, "over": 1.95, "under": 1.90},
    ]
    top_scores = set()
    for odds in cases:
        pred = ScorePredictor.from_odds(
            {k: odds[k] for k in ("home", "draw", "away")},
            {k: odds[k] for k in ("over", "under")},
        )
        top = pred.top_exact_scores(1).iloc[0]["score"]
        top_scores.add(top)
    assert len(top_scores) >= 2
    assert "1-1" not in top_scores or len(top_scores) > 1


def test_score_matrix_shape_and_sum():
    pred = ScorePredictor(FAIR_1X2, FAIR_OU, max_goals=5)
    m = pred.score_matrix()
    assert m.shape == (6, 6)
    assert abs(m.sum() - 1.0) < 1e-6
    assert (m >= 0).all()


def test_dixon_coles_inflates_low_draws_vs_independent_poisson():
    lam, mu = 1.35, 1.15
    m_dc = ScorePredictor.build_probability_matrix(lam, mu, rho=-0.15, pi=0.0)
    m_ind = ScorePredictor.build_probability_matrix(lam, mu, rho=0.0, pi=0.0)

    low_draw_mass_dc = m_dc[0, 0] + m_dc[1, 1]
    low_draw_mass_ind = m_ind[0, 0] + m_ind[1, 1]
    assert low_draw_mass_dc > low_draw_mass_ind


def test_zip_inflates_zero_zero():
    lam, mu = 1.35, 1.15
    m_zip = ScorePredictor.build_probability_matrix(lam, mu, rho=-0.13, pi=0.15)
    m_no = ScorePredictor.build_probability_matrix(lam, mu, rho=-0.13, pi=0.0)

    assert m_zip[0, 0] > m_no[0, 0]


def test_bivariate_probability_safe_log():
    p = ScorePredictor.bivariate_probability(0, 0, 1.3, 1.1, -0.1, 0.1)
    assert p > 0
    assert np.log(p) > -50


def test_top_exact_scores_dataframe():
    pred = ScorePredictor(FAIR_1X2, FAIR_OU)
    df = pred.top_exact_scores(5)
    assert len(df) == 5
    assert list(df.columns) == [
        "score",
        "home_goals",
        "away_goals",
        "probability",
        "percentage",
    ]
    assert df["probability"].is_monotonic_decreasing
    assert np.allclose(
        df["percentage"],
        (df["probability"] * 100).round(2),
        atol=0.01,
    )


def test_from_odds():
    odds_1x2 = {"home": 2.0, "draw": 3.5, "away": 4.0}
    odds_ou = {"over": 1.9, "under": 2.0}
    pred = ScorePredictor.from_odds(odds_1x2, odds_ou)
    df = pred.top_exact_scores(3)
    assert len(df) == 3


def test_invalid_probs_raise():
    with pytest.raises(ValueError):
        ScorePredictor(
            {"home": 0.5, "draw": 0.5, "away": 0.5},
            FAIR_OU,
        )


def test_fit_global_requires_minimum_matches():
    df = pd.read_csv(FIXTURE)
    with pytest.raises(ValueError, match="30 partidos"):
        ScorePredictor.fit_global(df)


def test_fit_global_on_synthetic_history():
    rng = np.random.default_rng(42)
    rows = []
    for i in range(40):
        lam, mu = 1.2 + rng.random() * 0.8, 0.9 + rng.random() * 0.8
        x = int(rng.poisson(lam))
        y = int(rng.poisson(mu))
        h, d, a = 2.0 + rng.random(), 3.2 + rng.random(), 3.5 + rng.random()
        o, u = 1.85 + rng.random() * 0.2, 1.85 + rng.random() * 0.2
        rows.append(
            {
                "FTHG": x,
                "FTAG": y,
                "B365H": h,
                "B365D": d,
                "B365A": a,
                "B365>2.5": o,
                "B365<2.5": u,
            }
        )
    hist = pd.DataFrame(rows)
    params = ScorePredictor.fit_global(hist)
    assert params.converged or params.neg_log_likelihood < 1e6
    assert 0.0 <= params.pi <= 0.45
    assert ScorePredictor.get_global_params() is not None

    pred = ScorePredictor(FAIR_1X2, FAIR_OU)
    fit = pred.fit()
    assert fit.rho == params.rho
    assert fit.pi == params.pi
