from __future__ import annotations

from football_odds_scraper.models import SelectorConfig

# ---------------------------------------------------------------------------
# Plantillas de selectores — DEBES ajustarlas al portal que uses.
# Cada clave es un alias lógico; copia y modifica según el DOM real.
# ---------------------------------------------------------------------------

GENERIC_TEMPLATE = SelectorConfig(
    # Ejemplo: tres botones/celdas del mercado resultado final
    home="[data-market='1x2'] [data-outcome='home'], .market-1x2 .odd-home",
    draw="[data-market='1x2'] [data-outcome='draw'], .market-1x2 .odd-draw",
    away="[data-market='1x2'] [data-outcome='away'], .market-1x2 .odd-away",
    over_25=(
        "[data-market='ou25'] [data-outcome='over'], "
        ".market-totals .line-2\\.5 .odd-over"
    ),
    under_25=(
        "[data-market='ou25'] [data-outcome='under'], "
        ".market-totals .line-2\\.5 .odd-under"
    ),
    market_root=".match-odds, #event-markets, main",
    odds_attribute=None,  # o "data-odds" si el sitio guarda la cuota en atributo
)

# Ejemplo orientativo para agregadores tabulares.
# Inspecciona el HTML del partido y ajusta índices de fila/columna.
ODDSCHECKER_LIKE = SelectorConfig(
    home="table.eventTable tbody tr.market-1x2 td.odd:nth-of-type(1)",
    draw="table.eventTable tbody tr.market-1x2 td.odd:nth-of-type(2)",
    away="table.eventTable tbody tr.market-1x2 td.odd:nth-of-type(3)",
    over_25="table.eventTable tbody tr.market-ou-25 td.odd-over",
    under_25="table.eventTable tbody tr.market-ou-25 td.odd-under",
    market_root="#betting-events, .event-container",
)

# Cuotas en atributos data (común en SPAs de casas de apuestas)
DATA_ODDS_SPA = SelectorConfig(
    home="[data-testid='outcome-home']",
    draw="[data-testid='outcome-draw']",
    away="[data-testid='outcome-away']",
    over_25="[data-testid='outcome-over-2.5']",
    under_25="[data-testid='outcome-under-2.5']",
    market_root="[data-testid='match-markets']",
    odds_attribute="data-odds",
)

PRESETS: dict[str, SelectorConfig] = {
    "generic": GENERIC_TEMPLATE,
    "oddschecker_like": ODDSCHECKER_LIKE,
    "data_odds_spa": DATA_ODDS_SPA,
}


def get_preset(name: str) -> SelectorConfig:
    try:
        return PRESETS[name]
    except KeyError as exc:
        available = ", ".join(sorted(PRESETS))
        raise KeyError(
            f"Preset desconocido: {name!r}. Disponibles: {available}"
        ) from exc
