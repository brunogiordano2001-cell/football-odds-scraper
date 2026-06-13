from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SelectorConfig:
    """Selectores CSS para un portal concreto.

    Cada selector debe apuntar a un único elemento cuyo texto sea la cuota decimal
  (p. ej. ``2.10``). Ajusta estos valores cuando el sitio cambie su HTML.
    """

    home: str
    draw: str
    away: str
    over_25: str
    under_25: str
    # Contenedor opcional donde buscar mercados (acelera el parseo)
    market_root: str | None = None
    # Atributo alternativo si la cuota no está en textContent (data-odds, etc.)
    odds_attribute: str | None = None


@dataclass(frozen=True)
class MatchOdds:
    """Cuotas decimales extraídas de un partido."""

    url: str
    home: float
    draw: float
    away: float
    over_25: float
    under_25: float
    goals_line: float = 2.5  # línea O/U (por defecto 2.5)
    bookmaker: str | None = None
    raw: dict[str, str] = field(default_factory=dict, repr=False)

    @property
    def market_1x2(self) -> dict[str, float]:
        return {"home": self.home, "draw": self.draw, "away": self.away}

    @property
    def market_over_under(self) -> dict[str, float]:
        return {"over": self.over_25, "under": self.under_25}

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "bookmaker": self.bookmaker,
            "goals_line": self.goals_line,
            "1x2": self.market_1x2,
            "over_under": self.market_over_under,
        }
