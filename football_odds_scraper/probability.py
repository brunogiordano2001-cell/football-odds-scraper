from __future__ import annotations

from typing import Mapping, Sequence

# Métodos de eliminación de overround soportados
OverroundMethod = str  # "multiplicative" | "additive" | "power"


def implied_probabilities(odds: Mapping[str, float]) -> dict[str, float]:
    """Probabilidades implícitas brutas (1 / cuota) sin normalizar."""
    if not odds:
        raise ValueError("Se requiere al menos una cuota.")
    result: dict[str, float] = {}
    for key, price in odds.items():
        if price <= 1.0:
            raise ValueError(
                f"Cuota inválida para '{key}': {price}. "
                "Las cuotas decimales deben ser > 1.0."
            )
        result[key] = 1.0 / price
    return result


def overround(odds: Mapping[str, float]) -> float:
    """Margen de la casa: suma de probabilidades implícitas menos 1."""
    implied = implied_probabilities(odds)
    return sum(implied.values()) - 1.0


def remove_overround(
    odds: Mapping[str, float],
    *,
    method: OverroundMethod = "multiplicative",
    power_iterations: int = 50,
    power_tolerance: float = 1e-9,
) -> dict[str, float]:
    """Elimina el overround y devuelve probabilidades justas que suman 1.

    Parameters
    ----------
    odds:
        Cuotas decimales por resultado (p. ej. ``{"home": 2.1, "draw": 3.4}``).
    method:
        - ``multiplicative`` (por defecto): divide cada prob. implícita por la suma.
        - ``additive``: resta el overround repartido por igual entre outcomes.
        - ``power``: método de Shin/Clarke (iterativo); útil en mercados asimétricos.

    Returns
    -------
    dict
        Probabilidades justas en [0, 1] que suman 1.
    """
    implied = implied_probabilities(odds)
    n = len(implied)
    if n == 0:
        raise ValueError("Mercado vacío.")

    total = sum(implied.values())
    if total <= 1.0 + 1e-12:
        # Sin margen (o datos de prueba): devolver implícitas normalizadas
        return {k: v / total for k, v in implied.items()}

    if method == "multiplicative":
        return {k: v / total for k, v in implied.items()}

    if method == "additive":
        margin = total - 1.0
        adjustment = margin / n
        adjusted = {k: max(v - adjustment, 1e-12) for k, v in implied.items()}
        adj_total = sum(adjusted.values())
        return {k: v / adj_total for k, v in adjusted.items()}

    if method == "power":
        return _power_method(implied, power_iterations, power_tolerance)

    raise ValueError(
        f"Método desconocido: {method!r}. "
        "Use 'multiplicative', 'additive' o 'power'."
    )


def fair_probabilities(
    match_odds: "MatchOddsLike",
    *,
    method: OverroundMethod = "multiplicative",
) -> dict[str, dict[str, float]]:
    """Calcula probabilidades justas para 1X2 y O/U por separado.

    Cada mercado se normaliza de forma independiente (margen propio).
    """
    from football_odds_scraper.models import MatchOdds

    if isinstance(match_odds, MatchOdds):
        return {
            "1x2": remove_overround(match_odds.market_1x2, method=method),
            "over_under": remove_overround(
                match_odds.market_over_under, method=method
            ),
        }

    # Duck typing para dicts u otros objetos con las mismas propiedades
    return {
        "1x2": remove_overround(
            {
                "home": match_odds.home,
                "draw": match_odds.draw,
                "away": match_odds.away,
            },
            method=method,
        ),
        "over_under": remove_overround(
            {
                "over": match_odds.over_25,
                "under": match_odds.under_25,
            },
            method=method,
        ),
    }


class MatchOddsLike:
    home: float
    draw: float
    away: float
    over_25: float
    under_25: float


def _power_method(
    implied: Mapping[str, float],
    max_iterations: int,
    tolerance: float,
) -> dict[str, float]:
    """Método potencia: encuentra k tal que sum(pi^k) = 1."""
    keys: Sequence[str] = list(implied.keys())
    values = [implied[k] for k in keys]

    lo, hi = 0.5, 2.0
    for _ in range(max_iterations):
        k = (lo + hi) / 2.0
        powered_sum = sum(p**k for p in values)
        if abs(powered_sum - 1.0) < tolerance:
            fair = [p**k for p in values]
            total = sum(fair)
            return dict(zip(keys, (f / total for f in fair)))
        if powered_sum > 1.0:
            lo = k
        else:
            hi = k

    # Fallback a multiplicativo si no converge
    total = sum(values)
    return dict(zip(keys, (v / total for v in values)))
