from __future__ import annotations

import asyncio
import logging
import re
import traceback
import urllib.parse
from typing import Any, Callable, Coroutine, Optional, TypeVar

import pandas as pd

from football_odds_scraper.exceptions import FetchError
from football_odds_scraper.worldcup_2026 import strip_team_display

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int, str], None]
T = TypeVar("T")

_DECIMAL_RE = re.compile(r"\b(\d{1,2}(?:[.,]\d{1,3})?)\b")

_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)


def oddsportal_search_url(home: str, away: str) -> str:
    home = strip_team_display(home)
    away = strip_team_display(away)
    query = urllib.parse.quote_plus(f"{home} {away}")
    return f"https://www.oddsportal.com/search/{query}/"


def run_async(coro: Coroutine[Any, Any, T]) -> T:
    """
    Ejecuta una corutina desde Streamlit (síncrono).

    Streamlit a veces ya tiene un event loop activo; ``asyncio.run`` falla con
    RuntimeError. Este helper crea un loop nuevo si hace falta.
    """
    try:
        return asyncio.run(coro)
    except RuntimeError as exc:
        msg = str(exc).lower()
        if "cannot be called from a running event loop" in msg or "already running" in msg:
            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                return loop.run_until_complete(coro)
            finally:
                loop.close()
                asyncio.set_event_loop(None)
        raise


def _parse_decimal(text: str) -> float | None:
    if not text:
        return None
    m = _DECIMAL_RE.search(text.replace(",", "."))
    if not m:
        return None
    val = float(m.group(1))
    return val if val > 1.0 else None


async def _apply_stealth(page: Any) -> None:
    """Stealth opcional; si falla la importación, solo User-Agent realista."""
    try:
        from playwright_stealth import stealth

        result = stealth(page)
        if asyncio.iscoroutine(result):
            await result
    except ImportError:
        logger.debug("playwright-stealth no disponible; usando solo User-Agent.")
    except Exception as exc:
        logger.warning("playwright-stealth falló (%s); continuando sin stealth.", exc)


async def _scrape_oddsportal_match(
    page: Any,
    home: str,
    away: str,
    *,
    timeout_ms: int = 35_000,
) -> dict[str, float | None]:
    """OddsPortal único proveedor — búsqueda + página del partido."""
    home_clean = strip_team_display(home)
    away_clean = strip_team_display(away)
    url = oddsportal_search_url(home_clean, away_clean)

    await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
    await page.wait_for_timeout(2000)

    # Aceptar cookies si aparece
    for sel in (
        "#onetrust-accept-btn-handler",
        "button:has-text('Accept')",
        "button:has-text('I Accept')",
    ):
        try:
            btn = page.locator(sel).first
            if await btn.count() > 0:
                await btn.click(timeout=3000)
                await page.wait_for_timeout(500)
                break
        except Exception:
            pass

    # Enlace al partido
    clicked = False
    for sel in (
        f"a[href*='/football/']:has-text('{home_clean[:4]}')",
        "a[href*='/football/world/']",
        "a[href*='/football/']",
        "a[href*='world-cup']",
    ):
        try:
            link = page.locator(sel).first
            if await link.count() > 0:
                await link.click(timeout=timeout_ms)
                await page.wait_for_load_state("domcontentloaded", timeout=timeout_ms)
                await page.wait_for_timeout(2500)
                clicked = True
                break
        except Exception:
            continue

    if not clicked:
        raise FetchError(
            f"No se encontró enlace al partido {home_clean} vs {away_clean} en OddsPortal."
        )

    odds_texts: list[str] = []
    selectors = [
        "div[data-testid='over-under-expanded'] span",
        "p.average-odds",
        "span[class*='odds']",
        "div.odds-nowrap",
        ".height-content",
    ]
    for selector in selectors:
        loc = page.locator(selector)
        count = min(await loc.count(), 30)
        for i in range(count):
            try:
                t = (await loc.nth(i).inner_text()).strip()
                if t and any(c.isdigit() for c in t):
                    odds_texts.append(t)
            except Exception:
                continue

    decimals: list[float] = []
    seen: set[float] = set()
    for t in odds_texts:
        v = _parse_decimal(t)
        if v is not None and v not in seen:
            seen.add(v)
            decimals.append(v)

    result: dict[str, float | None] = {
        "Cuota 1": None,
        "Cuota X": None,
        "Cuota 2": None,
        "Cuota Over": None,
        "Cuota Under": None,
        "Línea O/U": 2.5,
    }

    if len(decimals) < 3:
        raise FetchError(
            f"Cuotas 1X2 insuficientes para {home_clean} vs {away_clean} "
            f"(encontradas: {len(decimals)})."
        )

    result["Cuota 1"], result["Cuota X"], result["Cuota 2"] = (
        decimals[0],
        decimals[1],
        decimals[2],
    )
    if len(decimals) >= 5:
        result["Cuota Over"], result["Cuota Under"] = decimals[3], decimals[4]

    return result


async def scrape_worldcup_odds_batch(
    matches: list[dict[str, str]],
    *,
    headless: bool = True,
    timeout_ms: int = 35_000,
    delay_ms: int = 1500,
    on_progress: Optional[ProgressCallback] = None,
) -> list[dict[str, float | None]]:
    """Scraping masivo exclusivamente vía OddsPortal + playwright-stealth."""
    from playwright.async_api import async_playwright

    if not matches:
        return []

    total = len(matches)
    results: list[dict[str, float | None]] = []

    try:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(
                headless=headless,
                args=["--disable-blink-features=AutomationControlled"],
            )
            context = await browser.new_context(
                locale="es-ES",
                user_agent=_USER_AGENT,
                viewport={"width": 1280, "height": 900},
            )
            page = await context.new_page()
            await _apply_stealth(page)

            try:
                for i, match in enumerate(matches):
                    home = match["home"]
                    away = match["away"]
                    label = f"{strip_team_display(home)} vs {strip_team_display(away)}"

                    if on_progress:
                        on_progress(i, total, label)

                    try:
                        data = await _scrape_oddsportal_match(
                            page,
                            home,
                            away,
                            timeout_ms=timeout_ms,
                        )
                    except Exception as exc:
                        logger.error("Scrape falló %s: %s\n%s", label, exc, traceback.format_exc())
                        data = {
                            "Cuota 1": None,
                            "Cuota X": None,
                            "Cuota 2": None,
                            "Cuota Over": None,
                            "Cuota Under": None,
                            "Línea O/U": 2.5,
                            "error": str(exc),
                        }

                    results.append(data)
                    if delay_ms > 0 and i < total - 1:
                        await page.wait_for_timeout(delay_ms)
            finally:
                await context.close()
                await browser.close()
    except Exception as exc:
        raise FetchError(
            f"Error al iniciar Playwright: {exc}\n\n"
            "Verifica: pip install playwright playwright-stealth && playwright install chromium"
        ) from exc

    if on_progress:
        on_progress(total, total, "Completado")

    return results


def scrape_worldcup_odds_sync(
    matches: list[dict[str, str]],
    on_progress: Optional[ProgressCallback] = None,
    **kwargs: Any,
) -> list[dict[str, float | None]]:
    """Wrapper síncrono para Streamlit — usa ``run_async``."""
    return run_async(
        scrape_worldcup_odds_batch(matches, on_progress=on_progress, **kwargs)
    )


def merge_odds_into_dataframe(
    df: pd.DataFrame,
    indices: list[int],
    odds_list: list[dict[str, float | None]],
) -> pd.DataFrame:
    out = df.copy()
    for idx, odds in zip(indices, odds_list):
        for col in ("Cuota 1", "Cuota X", "Cuota 2", "Cuota Over", "Cuota Under"):
            val = odds.get(col)
            if val is not None:
                out.at[idx, col] = val
        if odds.get("Línea O/U") is not None:
            out.at[idx, "Línea O/U"] = odds["Línea O/U"]
    return out
