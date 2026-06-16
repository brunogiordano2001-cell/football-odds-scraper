"""Parser de odds pegadas en texto libre para el analizador individual."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from football_odds_scraper.oddspapi_client import parse_ou_curve_input

_ODD = r"(\d+(?:\.\d+)?)"
_PAIR = rf"{_ODD}\s*[/|]\s*{_ODD}"
_TRIPLE = rf"{_ODD}\s*[/|]\s*{_ODD}\s*[/|]\s*{_ODD}"


@dataclass
class ParsedOddsText:
    home: float | None = None
    draw: float | None = None
    away: float | None = None
    over: float | None = None
    under: float | None = None
    goals_line: float | None = None
    tt_home_over: float | None = None
    tt_home_under: float | None = None
    tt_away_over: float | None = None
    tt_away_under: float | None = None
    ou_curve_text: str | None = None
    ah_line: float | None = None
    ah_home: float | None = None
    ah_away: float | None = None
    detected: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)


def _valid_odd(value: str) -> float | None:
    try:
        odd = float(value)
    except (TypeError, ValueError):
        return None
    return odd if odd > 1.0 else None


def _search_pair(pattern: str, text: str) -> tuple[float, float] | None:
    match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
    if not match:
        return None
    a = _valid_odd(match.group(1))
    b = _valid_odd(match.group(2))
    if a is None or b is None:
        return None
    return a, b


def _search_triple(pattern: str, text: str) -> tuple[float, float, float] | None:
    match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
    if not match:
        return None
    a = _valid_odd(match.group(1))
    b = _valid_odd(match.group(2))
    c = _valid_odd(match.group(3))
    if a is None or b is None or c is None:
        return None
    return a, b, c


def _parse_1x2(text: str) -> tuple[float, float, float] | None:
    patterns = [
        rf"(?:1x2|moneyline)\s*:?\s*{_TRIPLE}",
        rf"1x2\s*{_TRIPLE}",
    ]
    for pattern in patterns:
        triple = _search_triple(pattern, text)
        if triple:
            return triple
    return None


def _parse_ou_main(text: str) -> tuple[float, float, float] | None:
    """O/U principal (línea + over/under). Prioriza 2.5 explícito."""
    patterns = [
        rf"o/u\s*2\.5\s*:?\s*{_PAIR}",
        rf"(?:over|total)\s*2\.5\s*:?\s*{_PAIR}",
        rf"o/u\s*(\d+(?:\.\d+)?)\s*:?\s*{_PAIR}",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if not match:
            continue
        groups = match.groups()
        if len(groups) == 2:
            over = _valid_odd(groups[0])
            under = _valid_odd(groups[1])
            line = 2.5
        else:
            line_raw = groups[0]
            over = _valid_odd(groups[1])
            under = _valid_odd(groups[2])
            try:
                line = float(line_raw)
            except (TypeError, ValueError):
                continue
        if over is None or under is None or line <= 0:
            continue
        return over, under, line

    fallback = _search_pair(
        rf"(?<!team\s)(?<!tt\s)(?:^|\n)\s*(?:over|total)\s*:?\s*{_PAIR}",
        text,
    )
    if fallback:
        return fallback[0], fallback[1], 2.5
    return None


def _parse_tt_home(text: str) -> tuple[float, float] | None:
    patterns = [
        rf"tt\s*home(?:\s*0\.5)?\s*:?\s*{_PAIR}",
        rf"home\s*0\.5\s*:?\s*{_PAIR}",
        rf"total\s*home(?:\s*0\.5)?\s*:?\s*{_PAIR}",
    ]
    for pattern in patterns:
        pair = _search_pair(pattern, text)
        if pair:
            return pair
    return None


def _parse_tt_away(text: str) -> tuple[float, float] | None:
    patterns = [
        rf"tt\s*away(?:\s*0\.5)?\s*:?\s*{_PAIR}",
        rf"away\s*0\.5\s*:?\s*{_PAIR}",
        rf"total\s*away(?:\s*0\.5)?\s*:?\s*{_PAIR}",
    ]
    for pattern in patterns:
        pair = _search_pair(pattern, text)
        if pair:
            return pair
    return None


def _parse_ou_curve_section(text: str) -> str | None:
    section_patterns = [
        r"(?:o/u\s*adicional|curva(?:\s*o/u)?)\s*:?\s*(.+)$",
    ]
    for pattern in section_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if not match:
            continue
        chunk = match.group(1).strip()
        chunk = re.split(r"\n\s*(?:ah|handicap|spread)\b", chunk, flags=re.IGNORECASE)[0]
        chunk = chunk.strip(" ,")
        if parse_ou_curve_input(chunk):
            return chunk
    return None


def _parse_ah(text: str) -> tuple[float, float, float] | None:
    patterns = [
        rf"(?:ah|handicap|spread)\s*:?\s*([-+]?\d+(?:\.\d+)?)\s*[/|]\s*{_ODD}\s*[/|]\s*{_ODD}",
        rf"(?:ah|handicap|spread)\s*:?\s*([-+]?\d+(?:\.\d+)?)\s+{_ODD}\s+{_ODD}",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if not match:
            continue
        try:
            line = float(match.group(1))
        except (TypeError, ValueError):
            continue
        home = _valid_odd(match.group(2))
        away = _valid_odd(match.group(3))
        if home is None or away is None:
            continue
        return line, home, away
    return None


def parse_odds_paste_text(text: str) -> ParsedOddsText:
    """Parsea bloque de texto libre con odds Pinnacle-style."""
    result = ParsedOddsText()
    if not text or not str(text).strip():
        result.missing = [
            "1X2",
            "O/U 2.5",
            "TT Home",
            "TT Away",
            "Curva O/U",
            "AH",
        ]
        return result

    normalized = str(text).strip()

    one_x_two = _parse_1x2(normalized)
    if one_x_two:
        result.home, result.draw, result.away = one_x_two
        result.detected.append(
            f"1X2 ({one_x_two[0]:g}/{one_x_two[1]:g}/{one_x_two[2]:g})"
        )
    else:
        result.missing.append("1X2")

    ou_main = _parse_ou_main(normalized)
    if ou_main:
        result.over, result.under, result.goals_line = ou_main
        line_label = f"{result.goals_line:g}"
        result.detected.append(
            f"O/U {line_label} ({result.over:g}/{result.under:g})"
        )
    else:
        result.missing.append("O/U 2.5")

    tt_home = _parse_tt_home(normalized)
    if tt_home:
        result.tt_home_over, result.tt_home_under = tt_home
        result.detected.append("TT Home ✅")
    else:
        result.missing.append("TT Home")

    tt_away = _parse_tt_away(normalized)
    if tt_away:
        result.tt_away_over, result.tt_away_under = tt_away
        result.detected.append("TT Away ✅")
    else:
        result.missing.append("TT Away")

    ou_curve = _parse_ou_curve_section(normalized)
    if ou_curve:
        curve = parse_ou_curve_input(ou_curve) or []
        result.ou_curve_text = ou_curve
        result.detected.append(f"Curva O/U: {len(curve)} puntos")
    else:
        result.missing.append("Curva O/U")

    ah = _parse_ah(normalized)
    if ah:
        result.ah_line, result.ah_home, result.ah_away = ah
        line_s = f"+{result.ah_line:g}" if result.ah_line > 0 else f"{result.ah_line:g}"
        result.detected.append(f"AH: {line_s}")
    else:
        result.missing.append("AH")

    return result


def format_parse_success_message(parsed: ParsedOddsText) -> str | None:
    if not parsed.detected:
        return None
    return "✅ Detectado: " + " | ".join(parsed.detected)


def format_parse_warning_message(parsed: ParsedOddsText) -> str | None:
    if not parsed.missing:
        return None
    return (
        "⚠️ No detectado: "
        + ", ".join(parsed.missing)
        + " — podés cargarlo manualmente abajo"
    )
