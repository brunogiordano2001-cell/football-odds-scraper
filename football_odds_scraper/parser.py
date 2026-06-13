from __future__ import annotations

import re
from typing import TYPE_CHECKING

from bs4 import BeautifulSoup, Tag

from football_odds_scraper.exceptions import ParseError
from football_odds_scraper.models import MatchOdds, SelectorConfig

if TYPE_CHECKING:
    from collections.abc import Iterable

# Cuotas decimales típicas: 1.50, 2.10, 12.00
_ODDS_PATTERN = re.compile(r"^\s*(\d{1,2}(?:[.,]\d{1,3})?)\s*$")
# Extrae el primer número decimal dentro de un texto más largo
_ODDS_SEARCH = re.compile(r"\b(\d{1,2}(?:[.,]\d{1,3})?)\b")


def parse_decimal_odds(raw: str) -> float:
    """Convierte texto de cuota a float decimal."""
    if not raw or not str(raw).strip():
        raise ValueError("Texto de cuota vacío.")

    cleaned = str(raw).strip().replace(",", ".")
    match = _ODDS_PATTERN.match(cleaned)
    if match:
        value = float(match.group(1))
    else:
        found = _ODDS_SEARCH.search(cleaned)
        if not found:
            raise ValueError(f"No se reconoce cuota decimal en: {raw!r}")
        value = float(found.group(1))

    if value <= 1.0:
        raise ValueError(f"Cuota decimal inválida: {value}")
    return value


def _root(soup: BeautifulSoup, config: SelectorConfig) -> Tag | BeautifulSoup:
    if not config.market_root:
        return soup
    node = soup.select_one(config.market_root)
    if node is None:
        raise ParseError(
            f"Contenedor de mercado no encontrado: {config.market_root!r}",
            field="market_root",
        )
    return node


def _extract_single(
    root: Tag | BeautifulSoup,
    selector: str,
    *,
    field: str,
    attribute: str | None = None,
) -> tuple[float, str]:
    element = root.select_one(selector)
    if element is None:
        raise ParseError(
            f"Selector no encontró elemento para '{field}': {selector!r}",
            field=field,
        )

    if attribute:
        raw = element.get(attribute)
        if raw is None:
            raise ParseError(
                f"Atributo {attribute!r} ausente en '{field}'",
                field=field,
            )
        raw_text = str(raw)
    else:
        raw_text = element.get_text(strip=True)

    try:
        return parse_decimal_odds(raw_text), raw_text
    except ValueError as exc:
        raise ParseError(
            f"Cuota inválida para '{field}': {raw_text!r} ({exc})",
            field=field,
        ) from exc


def parse_match_odds(
    html: str,
    url: str,
    config: SelectorConfig,
    *,
    bookmaker: str | None = None,
    goals_line: float = 2.5,
) -> MatchOdds:
    """Parsea HTML y devuelve cuotas 1X2 y O/U 2.5."""
    soup = BeautifulSoup(html, "lxml")
    root = _root(soup, config)
    attr = config.odds_attribute

    fields: Iterable[tuple[str, str]] = (
        ("home", config.home),
        ("draw", config.draw),
        ("away", config.away),
        ("over_25", config.over_25),
        ("under_25", config.under_25),
    )

    values: dict[str, float] = {}
    raw: dict[str, str] = {}

    for name, selector in fields:
        price, text = _extract_single(
            root, selector, field=name, attribute=attr
        )
        values[name] = price
        raw[name] = text

    return MatchOdds(
        url=url,
        home=values["home"],
        draw=values["draw"],
        away=values["away"],
        over_25=values["over_25"],
        under_25=values["under_25"],
        goals_line=goals_line,
        bookmaker=bookmaker,
        raw=raw,
    )
