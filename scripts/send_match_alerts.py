#!/usr/bin/env python3
"""Envía alertas Telegram ~20 min antes de cada partido del Mundial."""

from __future__ import annotations

import html
import io
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests

# Permite ejecutar desde scripts/ o desde la raíz del repo
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from football_odds_scraper.oddspapi_client import (  # noqa: E402
    AR_TZ,
    OddsPapiError,
    extract_pinnacle_odds,
    fetch_worldcup_fixtures_with_odds,
    fixture_is_finished,
    parse_fixture_start_datetime,
)
from football_odds_scraper.prediction_snapshots import (  # noqa: E402
    fixture_history,
    load_snapshots_today,
    snapshot_timestamp_hhmm,
)
from football_odds_scraper.score_predictor import ScorePredictor  # noqa: E402
from football_odds_scraper.team_colors import get_team_color  # noqa: E402
from football_odds_scraper.world_cup_teams import get_team_display  # noqa: E402

SENT_FILE = Path(__file__).resolve().parent / "sent_alerts.json"
ALERT_MIN_MINUTES = 15.0
ALERT_MAX_MINUTES = 25.0


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Variable de entorno requerida: {name}")
    return value


def load_sent() -> set[str]:
    if not SENT_FILE.is_file():
        return set()
    try:
        data = json.loads(SENT_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return set()
    if isinstance(data, list):
        return {str(item) for item in data}
    return set()


def save_sent(sent: set[str]) -> None:
    SENT_FILE.parent.mkdir(parents=True, exist_ok=True)
    SENT_FILE.write_text(
        json.dumps(sorted(sent), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def minutes_until_kickoff(fixture: dict[str, Any], *, now: datetime | None = None) -> float | None:
    start = parse_fixture_start_datetime(fixture.get("startTime"))
    if start is None:
        return None
    current = now or datetime.now(timezone.utc)
    return (start - current).total_seconds() / 60.0


def should_alert(fixture: dict[str, Any], *, now: datetime | None = None) -> bool:
    """True si el partido empieza en 15–25 minutos."""
    if fixture_is_finished(fixture):
        return False
    diff = minutes_until_kickoff(fixture, now=now)
    if diff is None:
        return False
    return ALERT_MIN_MINUTES <= diff <= ALERT_MAX_MINUTES


def run_model(fixture: dict[str, Any]) -> dict[str, Any]:
    """Corre ScorePredictor y retorna payload compacto para el mensaje."""
    bookmaker_odds = fixture.get("bookmakerOdds", {})
    odds = fixture.get("odds")
    if odds is None and isinstance(bookmaker_odds, dict):
        odds = extract_pinnacle_odds(bookmaker_odds)
    if not odds:
        raise ValueError("Sin odds Pinnacle para el partido")

    ah_line = odds.get("ah_line")
    ah_home = odds.get("ah_home")
    ah_away = odds.get("ah_away")
    use_ah = (
        ah_line is not None
        and ah_home is not None
        and float(ah_home) > 1.0
        and ah_away is not None
        and float(ah_away) > 1.0
    )

    predictor = ScorePredictor.from_odds(
        {
            "home": float(odds["home"]),
            "draw": float(odds["draw"]),
            "away": float(odds["away"]),
        },
        {
            "over": float(odds["over"]),
            "under": float(odds["under"]),
        },
        goals_line=float(odds.get("line", 2.5)),
        ah_line=float(ah_line) if use_ah else None,
        ah_home=float(ah_home) if use_ah else None,
        ah_away=float(ah_away) if use_ah else None,
        tt_home_over=odds.get("tt_home_over"),
        tt_home_under=odds.get("tt_home_under"),
        tt_away_over=odds.get("tt_away_over"),
        tt_away_under=odds.get("tt_away_under"),
        tt_home_line=odds.get("tt_home_line"),
        tt_away_line=odds.get("tt_away_line"),
        ou_curve=odds.get("ou_curve"),
        ah_curve=odds.get("ah_curve"),
    )
    fit = predictor.fit()
    prode = predictor.predict_for_prode()

    return {
        "odds": odds,
        "top3": prode["top3"],
        "top3_coverage": float(prode["top3_coverage"]),
        "sensitivity": prode["sensitivity"],
        "lambda_home": float(fit.lambda_home),
        "lambda_away": float(fit.mu_away),
    }


def build_message(fixture: dict[str, Any], prediction: dict[str, Any]) -> str:
    """Construye mensaje Telegram HTML."""
    home = html.escape(get_team_display(fixture.get("participant1Id")))
    away = html.escape(get_team_display(fixture.get("participant2Id")))
    start = parse_fixture_start_datetime(fixture.get("startTime"))
    hora = start.astimezone(AR_TZ).strftime("%H:%M") if start else "—"

    odds = prediction["odds"]
    top3 = prediction["top3"]
    sens = prediction["sensitivity"]
    lh = prediction["lambda_home"]
    la = prediction["lambda_away"]

    rob_label = html.escape(str(sens.get("robustness_label", "—")))
    rob_desc = html.escape(str(sens.get("robustness_desc", "")))
    rob_emoji = rob_label.split()[0] if rob_label else "⚪"

    score_at_inf = sens.get("score_at_inf", (0, 0))
    if isinstance(score_at_inf, (list, tuple)) and len(score_at_inf) >= 2:
        argmax_str = f"{int(score_at_inf[0])}-{int(score_at_inf[1])}"
    else:
        argmax_str = html.escape(str(score_at_inf))
    coincide = "✅ coincide" if sens.get("coincide_with_argmax") else "⚠️ difiere"

    medals = ("🥇", "🥈", "🥉")
    top_lines: list[str] = []
    for medal, entry in zip(medals, top3[:3]):
        top_lines.append(
            f"{medal} {html.escape(entry['score'])} — "
            f"EPV: {entry['epv']:.2f} | "
            f"P(exact): {entry['p_exact']:.1%} | "
            f"P(resultado): {entry['p_result']:.1%}"
        )

    msg = f"""
⚽ <b>PARTIDO EN 20 MINUTOS</b>

{home} vs {away}
🕐 {hora} (Argentina)

<b>📊 Odds Pinnacle</b>
1: {float(odds['home']):.3f} | X: {float(odds['draw']):.3f} | 2: {float(odds['away']):.3f}
O/U {float(odds.get('line', 2.5)):g}: {float(odds['over']):.3f} / {float(odds['under']):.3f}

<b>🤖 Modelo</b>
λ local: {lh:.2f} | λ visitante: {la:.2f}

<b>🏆 Top 3 EPV (para el prode)</b>
{chr(10).join(top_lines)}

<b>📐 Decisión</b>
{rob_emoji} {rob_label} — {rob_desc}
Argmax: {argmax_str} {coincide}

<b>📈 Cobertura</b>
Estos 3 scores cubren el {prediction['top3_coverage']:.1%} de probabilidad total
""".strip()
    return msg


def send_telegram(
    message: str,
    *,
    token: str,
    chat_id: str,
    chart_bytes: bytes | None = None,
) -> None:
    base = f"https://api.telegram.org/bot{token}"
    response = requests.post(
        f"{base}/sendMessage",
        json={
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        },
        timeout=30,
    )
    response.raise_for_status()

    if chart_bytes:
        photo_resp = requests.post(
            f"{base}/sendPhoto",
            data={"chat_id": chat_id},
            files={"photo": ("chart.png", chart_bytes, "image/png")},
            timeout=30,
        )
        photo_resp.raise_for_status()


def generate_chart(
    fixture: dict[str, Any],
    snapshots: list[dict[str, Any]] | None = None,
) -> bytes | None:
    """Genera PNG de evolución de probabilidades 1X2 (None si < 2 puntos)."""
    history = fixture_history(str(fixture["fixtureId"]), snapshots)
    if len(history) < 2:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times = [snapshot_timestamp_hhmm(entry["timestamp"]) for entry in history]
    p_home = [float(entry.get("p_home", 0.0)) for entry in history]
    p_draw = [float(entry.get("p_draw", 0.0)) for entry in history]
    p_away = [float(entry.get("p_away", 0.0)) for entry in history]

    home_color = get_team_color(fixture.get("participant1Id"), fallback="#2563EB")
    away_color = get_team_color(fixture.get("participant2Id"), fallback="#DC2626")

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    ax.plot(times, p_home, color=home_color, marker="o", label="Local", linewidth=2)
    ax.plot(times, p_draw, color="#94a3b8", linestyle="--", label="Empate", linewidth=2)
    ax.plot(times, p_away, color=away_color, marker="o", label="Visitante", linewidth=2)

    ax.set_ylabel("Probabilidad", color="white")
    ax.tick_params(colors="white")
    ax.legend(facecolor="#1a1a2e", labelcolor="white")
    for spine in ("bottom", "left"):
        ax.spines[spine].set_color("#444444")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", facecolor="#1a1a2e")
    buf.seek(0)
    plt.close(fig)
    return buf.read()


def main() -> int:
    api_key = _require_env("ODDSPAPI_KEY")
    telegram_token = _require_env("TELEGRAM_TOKEN")
    telegram_chat_id = _require_env("TELEGRAM_CHAT_ID")

    sent = load_sent()
    now = datetime.now(timezone.utc)

    try:
        fixtures, tournament_id, empty_msg = fetch_worldcup_fixtures_with_odds(api_key)
    except OddsPapiError as exc:
        print(f"Error OddsPapi: {exc}", file=sys.stderr)
        return 1

    if not fixtures:
        print(empty_msg or f"Sin fixtures (tournamentId={tournament_id})")
        return 0

    snapshots = load_snapshots_today()
    alerts_sent = 0

    for fixture in fixtures:
        fid = str(fixture["fixtureId"])

        if not fixture.get("hasOdds") and not fixture.get("odds"):
            continue
        if not should_alert(fixture, now=now):
            continue
        if fid in sent:
            continue

        try:
            prediction = run_model(fixture)
        except Exception as exc:
            print(f"Error en modelo para {fid}: {exc}", file=sys.stderr)
            continue

        chart = generate_chart(fixture, snapshots)
        message = build_message(fixture, prediction)

        try:
            send_telegram(
                message,
                token=telegram_token,
                chat_id=telegram_chat_id,
                chart_bytes=chart,
            )
        except requests.RequestException as exc:
            print(f"Error Telegram para {fid}: {exc}", file=sys.stderr)
            continue

        sent.add(fid)
        alerts_sent += 1
        print(f"✅ Alerta enviada para {fid}")

    save_sent(sent)
    print(f"Listo — {alerts_sent} alerta(s) enviada(s), {len(sent)} fixture(s) en sent_alerts.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
