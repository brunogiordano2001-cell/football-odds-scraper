# Football Odds Scraper

Scraper asíncrono en Python (Playwright + BeautifulSoup) para extraer cuotas **1X2** y **Más/Menos 2.5 goles**, con eliminación de overround.

## Instalación

```bash
cd football_odds_scraper
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
playwright install chromium
```

## Uso como librería

```python
import asyncio
from football_odds_scraper import (
    OddsScraper,
    SelectorConfig,
    fair_probabilities,
    remove_overround,
    scrape_match_odds,
)

# 1. Define selectores inspeccionando el DOM del portal objetivo
SELECTORS = SelectorConfig(
    home=".odd-home",
    draw=".odd-draw",
    away=".odd-away",
    over_25=".odd-over",
    under_25=".odd-under",
    market_root=".match-odds",
)

async def main():
    url = "https://ejemplo.com/partido/123"

    # Opción A: context manager (varias URLs)
    async with OddsScraper(SELECTORS) as scraper:
        odds = await scraper.scrape(url)
        print(odds.to_dict())

    # Opción B: atajo una sola URL
    odds = await scrape_match_odds(url, SELECTORS)

    # Probabilidades justas (sin margen de la casa)
    fair = fair_probabilities(odds, method="multiplicative")
    print(fair["1x2"])        # home, draw, away
    print(fair["over_under"]) # over, under

    # Solo mercado 1X2
    p = remove_overround(odds.market_1x2)
    print(p)

asyncio.run(main())
```

## CLI

```bash
football-odds "https://ejemplo.com/partido/123" --preset generic --fair
```

## Selectores

Cada portal usa HTML distinto. Abre DevTools en la página del partido, localiza los nodos de cada cuota y rellena `SelectorConfig`. Hay plantillas en `football_odds_scraper/selectors.py` (`generic`, `data_odds_spa`, …) como punto de partida.

Si el sitio guarda la cuota en un atributo:

```python
SelectorConfig(..., odds_attribute="data-odds")
```

## Overround

Para un mercado con cuotas decimales \(O_i\):

- Probabilidad implícita bruta: \(q_i = 1/O_i\)
- Overround: \(\sum q_i - 1\)
- **Multiplicativo** (por defecto): \(p_i = q_i / \sum q_j\) → probabilidades justas que suman 1.

También están disponibles los métodos `additive` y `power`.

## Motor de predicción (`ScorePredictor`)

Modelo **Dixon-Coles + Zero-Inflated Poisson (ZIP)**:

- **MLE global** (`ScorePredictor.fit_global(df)`): estima ρ y π sobre histórico football-data.co.uk.
- **Calibración local** (`predictor.fit()`): ajusta λ y μ para reproducir 1X2 y O/U del partido.

```python
import pandas as pd
from football_odds_scraper import ScorePredictor

hist = pd.read_csv("E0.csv")
ScorePredictor.fit_global(hist)  # ρ, π globales

predictor = ScorePredictor.from_odds(
    {"home": 2.1, "draw": 3.4, "away": 3.5},
    {"over": 1.95, "under": 1.90},
)
print(predictor.top_exact_scores(5))
```

A partir de probabilidades justas (1X2 + O/U 2.5):

```python
from football_odds_scraper import ScorePredictor, fair_probabilities

fair = fair_probabilities(match_odds)  # tras scrape

predictor = ScorePredictor.from_fair_probs(fair)
print(predictor.fitted_params)   # λ, μ, ρ
print(predictor.score_matrix_df())
print(predictor.top_exact_scores(5))
```

También puedes pasar cuotas directamente: `ScorePredictor.from_odds(odds_1x2, odds_ou)`.

## App Streamlit

```bash
pip install -e ".[app]"
playwright install chromium
streamlit run app.py
```

**Pestañas:**
1. **Analizador Individual** — un partido, heatmap y top marcadores.
2. **Fixture Mundial 2026** — 72 partidos de fase de grupos, tabla editable (`st.data_editor`), scraping masivo OddsPortal (playwright-stealth) y batch con `ScorePredictor`.

Sube CSV histórico en la barra lateral para entrenar ρ y π globales (MLE).

## Backtest avanzado multi-temporada

Coloca varios CSV en una carpeta (`E0_2122.csv`, `E0_2223.csv`, …) y ejecuta:

```bash
pip install -e .
python advanced_backtester.py /ruta/a/carpeta_csv/
# o
football-advanced-backtest /ruta/a/carpeta_csv/ --export resultados_test.csv
```

- **Train:** las 4 temporadas más antiguas → `ScorePredictor.fit_global()` (ρ, π por MLE).
- **Test:** última temporada → acierto marcador exacto, 1X2, Log-Loss y Brier Score.

## Backtest (football-data.co.uk)

Descarga un CSV de [football-data.co.uk](https://www.football-data.co.uk) (p. ej. `E0.csv` Premier League) y ejecuta:

```bash
football-backtest /ruta/a/E0.csv
# o
python -m football_odds_scraper.backtest /ruta/a/E0.csv -v --export resultados.csv
```

Usa columnas Bet365: `B365H`, `B365D`, `B365A`, `B365>2.5`, `B365<2.5` y resultados `FTHG`, `FTAG`.

## Alertas Telegram (GitHub Actions)

El workflow `.github/workflows/match_alerts.yml` corre cada **15 minutos** y envía un mensaje Telegram cuando un partido del Mundial empieza en **15–25 minutos** (~20 min de anticipación).

### Secrets en GitHub

En el repositorio: **Settings → Secrets and variables → Actions → New repository secret**

| Secret | Descripción |
|--------|-------------|
| `ODDSPAPI_KEY` | API key de [OddsPapi](https://oddspapi.io) |
| `TELEGRAM_TOKEN` | Token del bot de [@BotFather](https://t.me/BotFather) |
| `TELEGRAM_CHAT_ID` | ID del chat o canal destino |

Nunca commitear estos valores en el código. Solo configurarlos como secrets.

### Ejecución manual

```bash
export ODDSPAPI_KEY=...
export TELEGRAM_TOKEN=...
export TELEGRAM_CHAT_ID=...
pip install -e ".[app]" matplotlib
python scripts/send_match_alerts.py
```

También podés disparar el workflow desde **Actions → Match Alerts → Run workflow**.

El script reutiliza `ScorePredictor`, `extract_pinnacle_odds` y los snapshots del día (`scripts/snapshots/`) para adjuntar un gráfico de evolución si hay ≥ 2 puntos.

Los fixture ya alertados se guardan en `scripts/sent_alerts.json` (cacheado entre runs de GitHub Actions).

## Tests

```bash
pytest
```
