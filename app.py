"""Streamlit — predicción de marcadores con OddsPapi (Pinnacle)."""

from __future__ import annotations

from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
from io import BytesIO
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from football_odds_scraper.oddspapi_client import (
    AR_TZ,
    OddsPapiError,
    explain_pinnacle_odds_missing,
    extract_pinnacle_odds,
    fetch_worldcup_fixtures_with_odds,
    fixture_has_started,
    format_fixture_match_label,
    format_pinnacle_ah,
    format_pinnacle_calibration,
    get_oddspapi_key,
    get_pinnacle_markets_debug,
    get_world_cup_tournament_id,
    parse_correct_score_input,
    parse_fixture_start_datetime,
    parse_ou_curve_input,
    set_api_request_callback,
)
from football_odds_scraper.probability import overround, remove_overround
from football_odds_scraper.score_predictor import PoissonFit, ScorePredictor

WC_RESULT_PREFIX = "wc_result_"
FIXTURES_CACHE_TTL_SECONDS = 86400
DAILY_REQUEST_LIMIT = 250
REQUEST_LIMIT_WARN = 245

st.set_page_config(
    page_title="Football Score Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

_CUSTOM_CSS = """
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1500px; }
    h1 { font-weight: 600; letter-spacing: -0.02em; font-size: 1.75rem !important; }
    [data-testid="stMetric"] {
        background: #151B24; border: 1px solid #243044; border-radius: 12px; padding: 0.85rem 1rem;
    }
    [data-testid="stMetricLabel"] { font-size: 0.8rem; opacity: 0.75; }
    [data-testid="stMetricValue"] { font-size: 1.65rem !important; }
    div[data-testid="stSidebar"] { border-right: 1px solid #243044; }
    .subtitle { color: #94A3B8; font-size: 0.95rem; margin-bottom: 1.25rem; }
    hr.divider { border: none; border-top: 1px solid #243044; margin: 1.25rem 0; }
</style>
"""
st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)


def _init_session_state() -> None:
    for key, default in (
        ("ind_h", 2.10),
        ("ind_d", 3.40),
        ("ind_a", 3.50),
        ("ind_o", 1.95),
        ("ind_u", 1.90),
        ("ind_line", 2.5),
        ("ind_ah_line", 0.0),
        ("ind_ah_home", 0.0),
        ("ind_ah_away", 0.0),
        ("ind_cs_text", ""),
        ("ind_tt_home_over", 0.0),
        ("ind_tt_home_under", 0.0),
        ("ind_tt_away_over", 0.0),
        ("ind_tt_away_under", 0.0),
        ("ind_ou_curve_text", ""),
    ):
        st.session_state.setdefault(key, default)
    _reset_requests_if_new_day()


def _reset_requests_if_new_day() -> None:
    today = date.today().isoformat()
    if st.session_state.get("requests_date") != today:
        st.session_state["requests_date"] = today
        st.session_state["requests_used"] = 0


def _on_api_request() -> None:
    _reset_requests_if_new_day()
    st.session_state["requests_used"] = int(st.session_state.get("requests_used", 0)) + 1
    st.session_state["last_api_fetch_at"] = datetime.now(timezone.utc).isoformat()


def _requests_used() -> int:
    _reset_requests_if_new_day()
    return int(st.session_state.get("requests_used", 0))


def _last_api_fetch() -> datetime | None:
    raw = st.session_state.get("last_api_fetch_at")
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw))
    except ValueError:
        return None


def _can_refresh_fixture_cache() -> bool:
    last = _last_api_fetch()
    if last is None:
        return True
    expires = last + timedelta(seconds=FIXTURES_CACHE_TTL_SECONDS)
    return datetime.now(timezone.utc) >= expires


def _cache_refresh_label() -> str:
    last = _last_api_fetch()
    if last is None:
        return "disponible ahora"
    expires = last + timedelta(seconds=FIXTURES_CACHE_TTL_SECONDS)
    remaining = expires - datetime.now(timezone.utc)
    if remaining.total_seconds() <= 0:
        return "disponible ahora"
    hours, rem = divmod(int(remaining.total_seconds()), 3600)
    minutes = rem // 60
    return f"{hours}h {minutes}m"


def _render_request_counter() -> None:
    used = _requests_used()
    st.sidebar.caption(f"📡 Requests usados hoy: {used} / {DAILY_REQUEST_LIMIT}")


def _analyze_blocked() -> bool:
    return _requests_used() >= REQUEST_LIMIT_WARN


def _wc_result_key(fixture_id: str) -> str:
    return f"{WC_RESULT_PREFIX}{fixture_id}"


@st.cache_data(show_spinner=False)
def train_global_cached(csv_bytes: bytes) -> dict[str, Any]:
    df = pd.read_csv(BytesIO(csv_bytes))
    params = ScorePredictor.fit_global(df)
    return {
        "rho": params.rho,
        "pi": params.pi,
        "n_matches": params.n_matches,
        "neg_log_likelihood": params.neg_log_likelihood,
        "converged": params.converged,
    }


@st.cache_data(show_spinner=False)
def build_prediction(
    home_odd: float,
    draw_odd: float,
    away_odd: float,
    over_odd: float,
    under_odd: float,
    goals_line: float,
    top_n: int,
    ah_line: float | None = None,
    ah_home: float | None = None,
    ah_away: float | None = None,
    cs_odds_key: str = "",
    tt_home_over: float | None = None,
    tt_home_under: float | None = None,
    tt_away_over: float | None = None,
    tt_away_under: float | None = None,
    ou_curve_key: tuple[tuple[float, float, float], ...] = (),
) -> dict[str, Any]:
    odds_1x2 = {"home": home_odd, "draw": draw_odd, "away": away_odd}
    odds_ou = {"over": over_odd, "under": under_odd}
    fair = {"1x2": remove_overround(odds_1x2), "over_under": remove_overround(odds_ou)}
    cs_odds = parse_correct_score_input(cs_odds_key) if cs_odds_key.strip() else None
    ou_curve = list(ou_curve_key) if ou_curve_key else None
    predictor = ScorePredictor(
        fair["1x2"],
        fair["over_under"],
        goals_line=goals_line,
        ah_line=ah_line,
        ah_home=ah_home,
        ah_away=ah_away,
        correct_score_odds=cs_odds,
        tt_home_over=tt_home_over,
        tt_home_under=tt_home_under,
        tt_away_over=tt_away_over,
        tt_away_under=tt_away_under,
        ou_curve=ou_curve,
    )
    fit = predictor.fit()
    top = predictor.top_exact_scores(top_n)
    matrix = predictor.score_matrix()
    matrix_df = predictor.score_matrix_df()
    prode = predictor.predict_for_prode()
    return {
        "fair": fair,
        "overround": {"1x2": overround(odds_1x2), "over_under": overround(odds_ou)},
        "fit": asdict(fit),
        "top_scores": top.to_dict(orient="records"),
        "matrix": matrix.tolist(),
        "matrix_labels": list(matrix_df.index.astype(str)),
        "prode": {
            "score": prode["score_str"],
            "home_goals": prode["score"][0],
            "away_goals": prode["score"][1],
            "epv": prode["epv"],
            "p_exact": prode["p_exact"],
            "p_result": prode["p_result"],
            "top5": prode["top5"],
        },
    }


@st.cache_data(ttl=FIXTURES_CACHE_TTL_SECONDS, show_spinner=False)
def _get_world_cup_tournament_id_cached(api_key: str) -> str:
    return get_world_cup_tournament_id(api_key)


@st.cache_data(ttl=FIXTURES_CACHE_TTL_SECONDS, show_spinner="Cargando Mundial + odds Pinnacle…")
def _fetch_worldcup_fixtures_cached(
    api_key: str,
    tournament_id: str,
) -> tuple[list[dict[str, Any]], str | None]:
    fixtures, _, empty_msg = fetch_worldcup_fixtures_with_odds(
        api_key, tournament_id=tournament_id
    )
    return fixtures, empty_msg


def _load_worldcup_fixtures(api_key: str) -> tuple[list[dict[str, Any]], str, str | None]:
    tournament_id = _get_world_cup_tournament_id_cached(api_key)
    fixtures, empty_msg = _fetch_worldcup_fixtures_cached(api_key, tournament_id)
    return fixtures, tournament_id, empty_msg


def _global_params_banner() -> None:
    gp = ScorePredictor.get_global_params()
    if gp is not None:
        st.caption(f"MLE global — ρ={gp.rho:.3f}, π={gp.pi:.3f} (n={gp.n_matches})")
    else:
        st.caption("MLE global: defaults. Sube CSV histórico en la barra lateral.")


def _sidebar_global_training() -> None:
    st.sidebar.markdown("### OddsPapi")
    _render_request_counter()
    api_key = get_oddspapi_key()
    if api_key:
        st.sidebar.success("ODDSPAPI_KEY configurada ✓")
    else:
        st.sidebar.warning("Falta ODDSPAPI_KEY")
        st.sidebar.caption(
            "Define en `.streamlit/secrets.toml`:\n\n"
            "```\nODDSPAPI_KEY = \"tu_api_key\"\n```\n\n"
            "Conseguir key gratis en https://oddspapi.io"
        )

    hist_file = st.sidebar.file_uploader(
        "CSV histórico (football-data.co.uk)",
        type=["csv"],
        key="hist_csv_global",
    )
    if hist_file is not None:
        try:
            trained = train_global_cached(hist_file.getvalue())
            st.sidebar.success(
                f"MLE: ρ={trained['rho']:.3f}, π={trained['pi']:.3f} "
                f"({trained['n_matches']} partidos)"
            )
        except Exception as exc:
            st.sidebar.error(f"Entrenamiento falló: {exc}")


def _format_match_label(
    fixture: dict[str, Any],
    *,
    live: bool = False,
) -> str:
    return format_fixture_match_label(fixture, live=live)


def _ah_params_for_prediction(odds: dict[str, Any]) -> tuple[float | None, float | None, float | None]:
    ah_line = odds.get("ah_line")
    ah_home = odds.get("ah_home")
    ah_away = odds.get("ah_away")
    if ah_line is None or ah_home is None or ah_away is None:
        return None, None, None
    try:
        if float(ah_home) <= 0 or float(ah_away) <= 0:
            return None, None, None
    except (TypeError, ValueError):
        return None, None, None
    return float(ah_line), float(ah_home), float(ah_away)


def _cs_odds_key_from_dict(odds: dict[str, Any]) -> str:
    cs = odds.get("correct_score_odds")
    if not cs:
        return ""
    parts = []
    for (h, a), price in sorted(cs.items(), key=lambda x: (x[0][0], x[0][1])):
        parts.append(f"{h}-{a}:{float(price):.2f}")
    return ", ".join(parts)


def _ou_curve_text_from_dict(odds: dict[str, Any]) -> str:
    curve = odds.get("ou_curve")
    if not isinstance(curve, list) or not curve:
        return ""
    parts = []
    for point, over, under in sorted(curve, key=lambda item: item[0]):
        parts.append(f"{float(point):g}:{float(over):.2f}/{float(under):.2f}")
    return ", ".join(parts)


def _ou_curve_key_from_text(text: str) -> tuple[tuple[float, float, float], ...]:
    curve = parse_ou_curve_input(text)
    if not curve:
        return ()
    return tuple(curve)


def _ou_curve_key_from_dict(odds: dict[str, Any]) -> tuple[tuple[float, float, float], ...]:
    curve = odds.get("ou_curve")
    if not isinstance(curve, list) or not curve:
        return ()
    return tuple((float(p), float(o), float(u)) for p, o, u in curve)


def _optional_odd(value: float | None) -> float | None:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if v > 1.0 else None


def _manual_tt_params() -> tuple[float | None, float | None, float | None, float | None]:
    return (
        _optional_odd(st.session_state.get("ind_tt_home_over")),
        _optional_odd(st.session_state.get("ind_tt_home_under")),
        _optional_odd(st.session_state.get("ind_tt_away_over")),
        _optional_odd(st.session_state.get("ind_tt_away_under")),
    )


def _fixture_start_sort_key(fixture: dict[str, Any]) -> datetime:
    start = parse_fixture_start_datetime(fixture.get("startTime"))
    if start is None:
        return datetime.max.replace(tzinfo=timezone.utc)
    return start


def _predictions_updated_label() -> str:
    raw = st.session_state.get("predictions_updated_at")
    if not raw:
        return "—"
    try:
        dt = datetime.fromisoformat(str(raw))
    except ValueError:
        return "—"
    return dt.astimezone(AR_TZ).strftime("%H:%M (UTC-3)")


def _clear_fixture_caches() -> None:
    _get_world_cup_tournament_id_cached.clear()
    _fetch_worldcup_fixtures_cached.clear()
    build_prediction.clear()


def _analyzed_fixture_ids() -> list[str]:
    prefix_len = len(WC_RESULT_PREFIX)
    return [
        key[prefix_len:]
        for key in st.session_state
        if isinstance(key, str) and key.startswith(WC_RESULT_PREFIX)
    ]


def _reanalyze_fixtures(fixtures: list[dict[str, Any]], fixture_ids: list[str]) -> None:
    by_id = {str(f["fixtureId"]): f for f in fixtures}
    for fixture_id in fixture_ids:
        fixture = by_id.get(fixture_id)
        if fixture is not None:
            _analyze_fixture(fixture)


def _wc_analysis_for_copy() -> dict[str, Any] | None:
    last_key = st.session_state.get("wc_last_analysis")
    if isinstance(last_key, str):
        analysis = st.session_state.get(last_key)
        if isinstance(analysis, dict) and analysis.get("odds"):
            return analysis
    for key, value in st.session_state.items():
        if key.startswith(WC_RESULT_PREFIX) and isinstance(value, dict) and value.get("odds"):
            return value
    return None


def _copy_wc_inputs_to_individual(analysis: dict[str, Any]) -> None:
    odds = analysis["odds"]
    st.session_state["ind_h"] = float(odds["home"])
    st.session_state["ind_d"] = float(odds["draw"])
    st.session_state["ind_a"] = float(odds["away"])
    st.session_state["ind_o"] = float(odds["over"])
    st.session_state["ind_u"] = float(odds["under"])
    st.session_state["ind_line"] = float(odds.get("line", 2.5))
    st.session_state["ind_cs_text"] = _cs_odds_key_from_dict(odds)
    st.session_state["ind_ou_curve_text"] = _ou_curve_text_from_dict(odds)

    ah_line, ah_home, ah_away = _ah_params_for_prediction(odds)
    st.session_state["ind_ah_line"] = float(ah_line) if ah_line is not None else 0.0
    st.session_state["ind_ah_home"] = float(ah_home) if ah_home is not None else 0.0
    st.session_state["ind_ah_away"] = float(ah_away) if ah_away is not None else 0.0

    for key, src in (
        ("ind_tt_home_over", "tt_home_over"),
        ("ind_tt_home_under", "tt_home_under"),
        ("ind_tt_away_over", "tt_away_over"),
        ("ind_tt_away_under", "tt_away_under"),
    ):
        val = odds.get(src)
        st.session_state[key] = float(val) if val is not None else 0.0


def _prode_value_label(epv: float) -> str:
    if epv > 0.90:
        return "🔥 Alto"
    if epv >= 0.65:
        return "✅ Bueno"
    return "⚠️ Bajo"


def _render_lambda_summary(fit: PoissonFit) -> None:
    total = fit.lambda_home + fit.mu_away
    st.info(
        f"λ local: **{fit.lambda_home:.2f}** | "
        f"λ visitante: **{fit.mu_away:.2f}** | "
        f"Total esperado: **{total:.2f}** goles"
    )


def _render_prode_prediction(prode: dict[str, Any]) -> None:
    st.markdown(f"**Predicción recomendada (max EPV):** `{prode['score']}`")
    st.markdown(
        f"- Valor prode esperado: **{prode['epv']:.2f}**\n"
        f"- P(exact score): **{prode['p_exact'] * 100:.1f}%** → vale 3 pts\n"
        f"- P(solo resultado): **{prode['p_result'] * 100:.1f}%** → vale 1 pt"
    )
    alts = prode.get("top5") or []
    if len(alts) > 1:
        st.markdown("**Alternativas:**")
        for idx, alt in enumerate(alts[1:3], start=2):
            st.markdown(
                f"#{idx}: `{alt['score']}`  EPV: **{alt['epv']:.2f}**  "
                f"(P: {alt['p_exact'] * 100:.1f}% | Res: {alt['p_result'] * 100:.1f}%)"
            )


def _fixture_prode_row(
    fixture: dict[str, Any],
    *,
    live: bool,
) -> dict[str, Any] | None:
    """Calcula EPV para ranking del fixture (sin mutar session_state)."""
    bookmaker_odds = fixture.get("bookmakerOdds", {})
    odds = extract_pinnacle_odds(bookmaker_odds) if isinstance(bookmaker_odds, dict) else None
    if odds is None:
        odds = fixture.get("odds")
    if not odds:
        return None

    try:
        ah_line, ah_home, ah_away = _ah_params_for_prediction(odds)
        cs_key = _cs_odds_key_from_dict(odds)
        prediction = build_prediction(
            float(odds["home"]),
            float(odds["draw"]),
            float(odds["away"]),
            float(odds["over"]),
            float(odds["under"]),
            float(odds.get("line", 2.5)),
            top_n=3,
            ah_line=ah_line,
            ah_home=ah_home,
            ah_away=ah_away,
            cs_odds_key=cs_key,
            tt_home_over=odds.get("tt_home_over"),
            tt_home_under=odds.get("tt_home_under"),
            tt_away_over=odds.get("tt_away_over"),
            tt_away_under=odds.get("tt_away_under"),
            ou_curve_key=_ou_curve_key_from_dict(odds),
        )
    except Exception:
        return None

    prode = prediction["prode"]
    return {
        "fixture": fixture,
        "fixture_id": str(fixture["fixtureId"]),
        "label": _format_match_label(fixture, live=live),
        "live": live,
        "start_time": _fixture_start_sort_key(fixture),
        "prediction": prediction,
        "score": prode["score"],
        "best_epv": float(prode["epv"]),
        "p_exact": float(prode["p_exact"]),
        "p_result": float(prode["p_result"]),
        "valor": _prode_value_label(float(prode["epv"])),
    }


def _render_fixture_prode_table(rows: list[dict[str, Any]], *, api_key: str) -> None:
    if not rows:
        return

    btn_col, ts_col = st.columns([2, 3])
    with btn_col:
        if st.button(
            "🔄 Actualizar odds y predicciones",
            type="primary",
            key="fixture_refresh_predictions_btn",
            disabled=_analyze_blocked(),
        ):
            analyzed_ids = _analyzed_fixture_ids()
            _clear_fixture_caches()
            st.session_state["predictions_updated_at"] = datetime.now(timezone.utc).isoformat()
            try:
                fixtures, _, _ = _load_worldcup_fixtures(api_key)
                _reanalyze_fixtures(fixtures, analyzed_ids)
            except OddsPapiError as exc:
                st.error(str(exc))
            else:
                st.rerun()
    with ts_col:
        st.caption(f"Última actualización: {_predictions_updated_label()}")

    best = max(rows, key=lambda r: r["best_epv"])
    st.info(
        f"🔥 **Mejor partido del día para el prode:** "
        f"{best['label']} → Predicción: **{best['score']}** "
        f"(EPV: **{best['best_epv']:.2f}** | "
        f"{best['p_exact'] * 100:.1f}% de acertar el score exacto)"
    )

    table = pd.DataFrame(
        [
            {
                "Partido": row["label"],
                "Predicción": row["score"],
                "P(exact)": f"{row['p_exact'] * 100:.1f}%",
                "P(1X2)": f"{row['p_result'] * 100:.1f}%",
                "EPV": f"{row['best_epv']:.2f}",
                "Valor prode": row["valor"],
            }
            for row in rows
        ]
    )
    st.dataframe(table, use_container_width=True, hide_index=True)


def _calibration_caption(fit: PoissonFit) -> str:
    if fit.calibrated_with_cs:
        return "📐 Calibrado con mercado Correct Score (Pinnacle CS)"
    if fit.calibrated_with_team_totals:
        return (
            f"📐 Calibración completa (TT + curva O/U) — "
            f"λ local: {fit.lambda_home:.2f} | λ visitante: {fit.mu_away:.2f}"
        )
    if fit.calibrated_with_ah and fit.ah_line is not None:
        line_f = float(fit.ah_line)
        line_str = f"+{line_f:g}" if line_f > 0 else f"{line_f:g}"
        return (
            f"📐 Calibración básica (5 inputs + AH {line_str}) — "
            f"λ local: {fit.lambda_home:.2f} | λ visitante: {fit.mu_away:.2f}"
        )
    return "📐 Calibración básica (5 inputs)"


def _analyze_fixture(fixture: dict[str, Any]) -> str | None:
    """Corre el modelo con odds del batch (re-parsea desde bookmakerOdds)."""
    fixture_id = str(fixture["fixtureId"])
    bookmaker_odds = fixture.get("bookmakerOdds", {})
    odds = extract_pinnacle_odds(bookmaker_odds) if isinstance(bookmaker_odds, dict) else None
    if odds is None:
        odds = fixture.get("odds")

    if not odds:
        st.session_state.pop(_wc_result_key(fixture_id), None)
        if isinstance(bookmaker_odds, dict) and bookmaker_odds:
            return explain_pinnacle_odds_missing(bookmaker_odds)
        return "⚠️ Pinnacle no disponible para este partido"

    try:
        ah_line, ah_home, ah_away = _ah_params_for_prediction(odds)
        cs_key = _cs_odds_key_from_dict(odds)
        prediction = build_prediction(
            float(odds["home"]),
            float(odds["draw"]),
            float(odds["away"]),
            float(odds["over"]),
            float(odds["under"]),
            float(odds.get("line", 2.5)),
            top_n=3,
            ah_line=ah_line,
            ah_home=ah_home,
            ah_away=ah_away,
            cs_odds_key=cs_key,
            tt_home_over=odds.get("tt_home_over"),
            tt_home_under=odds.get("tt_home_under"),
            tt_away_over=odds.get("tt_away_over"),
            tt_away_under=odds.get("tt_away_under"),
            ou_curve_key=_ou_curve_key_from_dict(odds),
        )
    except Exception as exc:
        return f"Error en el modelo: {exc}"

    st.session_state[_wc_result_key(fixture_id)] = {
        "fixture": fixture,
        "odds": odds,
        "prediction": prediction,
    }
    st.session_state["wc_last_analysis"] = _wc_result_key(fixture_id)
    return None


def _render_fixture_analysis(fixture_id: str) -> None:
    analysis = st.session_state.get(_wc_result_key(fixture_id))
    if not analysis:
        return

    odds = analysis["odds"]
    prediction = analysis["prediction"]
    prode = prediction.get("prode")

    with st.container(border=True):
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("1", f"{odds['home']:.2f}")
        c2.metric("X", f"{odds['draw']:.2f}")
        c3.metric("2", f"{odds['away']:.2f}")
        c4.metric(f"O{odds.get('line', 2.5):g}", f"{odds['over']:.2f}")
        c5.metric(f"U{odds.get('line', 2.5):g}", f"{odds['under']:.2f}")

        if prode:
            _render_prode_prediction(prode)

        fit = PoissonFit(**prediction["fit"])
        _render_lambda_summary(fit)
        st.caption(_calibration_caption(fit))
        st.caption(f"ρ={fit.rho:.3f} · π={fit.pi:.3f}")
        with st.expander("Top marcadores por probabilidad (argmax)"):
            _render_top_scores(prediction["top_scores"])


def _metric_row(fit: PoissonFit, fair: dict[str, dict[str, float]]) -> None:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Goles esperados (local)", f"{fit.lambda_home:.2f}")
    c2.metric("Goles esperados (visitante)", f"{fit.mu_away:.2f}")
    c3.metric("ρ Dixon-Coles", f"{fit.rho:.3f}")
    c4.metric("π ZIP (0-0)", f"{fit.pi:.3f}")
    c5.metric("P(empate) justa", f"{fair['1x2']['draw'] * 100:.1f}%")


def _plot_heatmap(matrix: np.ndarray, labels: list[str]) -> None:
    pct = np.asarray(matrix, dtype=float) * 100.0
    fig = go.Figure(
        data=go.Heatmap(
            z=pct,
            x=[f"{g} visit." for g in labels],
            y=[f"{g} local" for g in labels],
            colorscale=[
                [0.0, "#0B0F14"], [0.25, "#134E4A"], [0.5, "#14B8A6"],
                [0.75, "#5EEAD4"], [1.0, "#CCFBF1"],
            ],
            colorbar=dict(title="%", ticksuffix="%"),
            hovertemplate="Local %{y}<br>Visitante %{x}<br>%{z:.2f}%<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=460, title=dict(text="Matriz de marcadores exactos", font=dict(size=14)),
        xaxis=dict(side="top"), yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _render_top_scores(records: list[dict[str, Any]]) -> None:
    df = pd.DataFrame(records)
    if df.empty:
        st.info("Sin resultados.")
        return
    display = df[["score", "percentage", "probability"]].copy()
    display.columns = ["Marcador", "Prob. %", "Prob."]
    st.dataframe(display, use_container_width=True, hide_index=True)


def render_individual_tab() -> None:
    _global_params_banner()

    wc_analysis = _wc_analysis_for_copy()
    if st.session_state.pop("ind_copy_notice", False):
        st.success("Inputs copiados desde Pestaña 2 (incluye TT y curva O/U).")
    copy_col, _ = st.columns([2, 4])
    with copy_col:
        if st.button(
            "📋 Copiar inputs de Pestaña 2",
            key="ind_copy_from_fixture",
            disabled=wc_analysis is None,
        ):
            _copy_wc_inputs_to_individual(wc_analysis)  # type: ignore[arg-type]
            st.session_state["ind_copy_notice"] = True
            st.rerun()

    with st.expander("Cuotas manuales", expanded=True):
        c1, c2, c3 = st.columns(3)
        c1.number_input("Local", 1.01, step=0.01, format="%.2f", key="ind_h")
        c2.number_input("Empate", 1.01, step=0.01, format="%.2f", key="ind_d")
        c3.number_input("Visit.", 1.01, step=0.01, format="%.2f", key="ind_a")
        c4, c5 = st.columns(2)
        c4.number_input("Más O/U", 1.01, step=0.01, format="%.2f", key="ind_o")
        c5.number_input("Menos O/U", 1.01, step=0.01, format="%.2f", key="ind_u")

        st.markdown("**Asian Handicap (opcional)**")
        ah1, ah2, ah3 = st.columns(3)
        ah1.number_input(
            "Línea AH (ej: -0.25 = local favorito)",
            step=0.25,
            format="%.2f",
            key="ind_ah_line",
        )
        ah2.number_input("Cuota AH Home", min_value=0.0, step=0.01, format="%.2f", key="ind_ah_home")
        ah3.number_input("Cuota AH Away", min_value=0.0, step=0.01, format="%.2f", key="ind_ah_away")

        st.text_area(
            "Cuotas Correct Score (opcional, formato: 1-0:7.5, 0-0:9.0, 1-1:6.5)",
            key="ind_cs_text",
            height=80,
        )

        with st.expander("Datos adicionales (mejoran la predicción)"):
            st.caption("Team totals 0.5 y curva O/U completa — mismo pipeline que Pestaña 2.")
            tt1, tt2 = st.columns(2)
            tt1.number_input(
                "TT Home Over 0.5",
                min_value=0.0,
                step=0.01,
                format="%.2f",
                key="ind_tt_home_over",
            )
            tt1.number_input(
                "TT Home Under 0.5",
                min_value=0.0,
                step=0.01,
                format="%.2f",
                key="ind_tt_home_under",
            )
            tt2.number_input(
                "TT Away Over 0.5",
                min_value=0.0,
                step=0.01,
                format="%.2f",
                key="ind_tt_away_over",
            )
            tt2.number_input(
                "TT Away Under 0.5",
                min_value=0.0,
                step=0.01,
                format="%.2f",
                key="ind_tt_away_under",
            )
            st.text_input(
                "Líneas O/U adicionales (ej: 1.5:1.48/2.73, 2.0:1.87/2.04, 3.0:4.01/1.26)",
                key="ind_ou_curve_text",
            )

        g1, g2, g3 = st.columns([1, 1, 1])
        g1.number_input("Línea O/U", 0.5, 5.5, step=0.5, key="ind_line")
        top_n = g2.slider("Top marcadores", 3, 10, 5, key="ind_topn")
        run = g3.button("Analizar partido", type="primary", use_container_width=True, key="ind_run")

    if not run:
        st.info("Edita las cuotas manualmente y pulsa **Analizar partido**.")
        return

    ah_line = ah_home = ah_away = None
    if float(st.session_state["ind_ah_home"]) > 0:
        ah_line = float(st.session_state["ind_ah_line"])
        ah_home = float(st.session_state["ind_ah_home"])
        ah_away = float(st.session_state["ind_ah_away"])

    tt_home_over, tt_home_under, tt_away_over, tt_away_under = _manual_tt_params()
    ou_curve_key = _ou_curve_key_from_text(str(st.session_state.get("ind_ou_curve_text", "")))

    try:
        result = build_prediction(
            st.session_state["ind_h"],
            st.session_state["ind_d"],
            st.session_state["ind_a"],
            st.session_state["ind_o"],
            st.session_state["ind_u"],
            st.session_state["ind_line"],
            top_n,
            ah_line=ah_line,
            ah_home=ah_home,
            ah_away=ah_away,
            cs_odds_key=str(st.session_state.get("ind_cs_text", "")),
            tt_home_over=tt_home_over,
            tt_home_under=tt_home_under,
            tt_away_over=tt_away_over,
            tt_away_under=tt_away_under,
            ou_curve_key=ou_curve_key,
        )
    except Exception as exc:
        st.error(f"Error en el modelo: {exc}")
        return

    fit = PoissonFit(**result["fit"])
    _render_lambda_summary(fit)
    st.caption(_calibration_caption(fit))
    _metric_row(fit, result["fair"])
    if result.get("prode"):
        _render_prode_prediction(result["prode"])
    left, right = st.columns((1, 1.35))
    with left:
        with st.expander("Top marcadores por probabilidad (argmax)"):
            _render_top_scores(result["top_scores"])
    with right:
        _plot_heatmap(np.array(result["matrix"]), result["matrix_labels"])


def render_fixture_tab(api_key: str) -> None:
    _global_params_banner()
    _render_request_counter()

    if not api_key:
        st.warning(
            "Configura `ODDSPAPI_KEY` en `.streamlit/secrets.toml` o como variable de entorno "
            "para cargar el fixture del día."
        )
        return

    if _analyze_blocked():
        st.warning(
            "⚠️ Límite de requests casi alcanzado. "
            "Reiniciá mañana o usá una nueva API key."
        )

    can_refresh = _can_refresh_fixture_cache()
    if st.button(
        "🔄 Actualizar lista",
        type="secondary",
        key="fixture_refresh_list_btn",
        disabled=not can_refresh,
    ):
        _get_world_cup_tournament_id_cached.clear()
        _fetch_worldcup_fixtures_cached.clear()
        st.rerun()

    st.caption(f"Próxima actualización disponible en: {_cache_refresh_label()}")

    try:
        fixtures, tournament_id, empty_msg = _load_worldcup_fixtures(api_key)
    except OddsPapiError as exc:
        st.error(str(exc))
        return
    except Exception as exc:
        st.error(f"Error inesperado al cargar partidos: {exc}")
        return

    if not fixtures:
        if empty_msg:
            st.info(empty_msg)
        else:
            st.info("No hay partidos del Mundial con odds Pinnacle disponibles.")
        return

    now = datetime.now(timezone.utc)
    show_filter = st.radio(
        "Mostrar:",
        options=["Solo próximos", "Todos"],
        horizontal=True,
        key="fixture_show_filter",
    )

    visible_fixtures: list[tuple[dict[str, Any], bool]] = []
    for fixture in fixtures:
        live = fixture_has_started(fixture, now=now)
        if show_filter == "Solo próximos" and live:
            continue
        visible_fixtures.append((fixture, live))

    if not visible_fixtures:
        st.info("No hay partidos próximos. Cambiá el filtro a **Todos** para ver los en curso.")
        return

    st.caption(f"{len(visible_fixtures)} partidos · tournamentId `{tournament_id}`")

    with st.expander("🔍 Debug odds raw"):
        first_fixture = visible_fixtures[0][0]
        first_id = str(first_fixture["fixtureId"])
        first_label = _format_match_label(first_fixture)
        st.caption(f"Primer partido visible: {first_label} (fixtureId={first_id})")
        st.json(get_pinnacle_markets_debug(first_fixture.get("bookmakerOdds", {})))

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    prode_rows: list[dict[str, Any]] = []
    for fixture, live in visible_fixtures:
        row = _fixture_prode_row(fixture, live=live)
        if row is not None:
            prode_rows.append(row)
    prode_rows.sort(key=lambda r: r["start_time"])

    if prode_rows:
        st.markdown("#### Predicciones del día")
        _render_fixture_prode_table(prode_rows, api_key=api_key)
        st.markdown('<hr class="divider">', unsafe_allow_html=True)

    fixture_order = [row["fixture"] for row in prode_rows] + [
        fixture
        for fixture, _ in visible_fixtures
        if str(fixture["fixtureId"]) not in {r["fixture_id"] for r in prode_rows}
    ]
    fixture_order.sort(key=_fixture_start_sort_key)
    live_by_id = {str(f["fixtureId"]): live for f, live in visible_fixtures}

    for fixture in fixture_order:
        fixture_id = str(fixture["fixtureId"])
        live = live_by_id.get(fixture_id, False)
        label = _format_match_label(fixture, live=live)
        result_key = _wc_result_key(fixture_id)
        analyzed = result_key in st.session_state
        action: str | None = None

        row_left, row_right = st.columns([5, 1])
        with row_left:
            st.markdown(f"**{label}**")
            odds_preview = fixture.get("odds")
            if odds_preview is None and isinstance(fixture.get("bookmakerOdds"), dict):
                odds_preview = extract_pinnacle_odds(fixture["bookmakerOdds"])
            ah_text = format_pinnacle_ah(odds_preview)
            st.caption(ah_text)
            st.caption(format_pinnacle_calibration(odds_preview))
        with row_right:
            if analyzed:
                if st.button("🔄 Refrescar", key=f"fixture_refresh_{fixture_id}"):
                    action = "refresh"
            elif st.button(
                "📊 Analizar",
                key=f"fixture_analyze_{fixture_id}",
                disabled=live or _analyze_blocked(),
            ):
                action = "analyze"

        if action in ("analyze", "refresh"):
            err = _analyze_fixture(fixture)
            if err:
                st.warning(err)
            else:
                st.rerun()

        if result_key in st.session_state:
            _render_fixture_analysis(fixture_id)

        st.markdown('<hr class="divider">', unsafe_allow_html=True)


def main() -> None:
    _init_session_state()
    set_api_request_callback(_on_api_request)
    _sidebar_global_training()
    api_key = get_oddspapi_key()

    st.title("Football Score Predictor")
    st.markdown(
        '<p class="subtitle">Dixon-Coles + ZIP · Mundial 2026 · OddsPapi (Pinnacle)</p>',
        unsafe_allow_html=True,
    )

    tab_ind, tab_fixture = st.tabs(["Analizador Individual", "Fixture del Día"])
    with tab_ind:
        render_individual_tab()
    with tab_fixture:
        render_fixture_tab(api_key)


if __name__ == "__main__":
    main()
