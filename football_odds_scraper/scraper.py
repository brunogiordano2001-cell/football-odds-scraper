from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from playwright.async_api import Browser, Page, TimeoutError as PlaywrightTimeout
from playwright.async_api import async_playwright

from football_odds_scraper.exceptions import FetchError, ParseError, ScraperError
from football_odds_scraper.models import MatchOdds, SelectorConfig
from football_odds_scraper.parser import parse_match_odds

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

logger = logging.getLogger(__name__)


class OddsScraper:
    """Cliente asíncrono Playwright + BeautifulSoup para cuotas de un partido."""

    def __init__(
        self,
        selectors: SelectorConfig,
        *,
        headless: bool = True,
        timeout_ms: int = 30_000,
        wait_until: str = "domcontentloaded",
        user_agent: str | None = None,
        locale: str = "es-ES",
    ) -> None:
        self.selectors = selectors
        self.headless = headless
        self.timeout_ms = timeout_ms
        self.wait_until = wait_until
        self.user_agent = user_agent
        self.locale = locale
        self._browser: Browser | None = None
        self._playwright: Any = None

    async def __aenter__(self) -> OddsScraper:
        await self.start()
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()

    async def start(self) -> None:
        if self._browser is not None:
            return
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=self.headless
        )

    async def close(self) -> None:
        if self._browser:
            await self._browser.close()
            self._browser = None
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None

    @asynccontextmanager
    async def _page(self) -> AsyncIterator[Page]:
        if self._browser is None:
            await self.start()
        assert self._browser is not None
        context = await self._browser.new_context(
            locale=self.locale,
            user_agent=self.user_agent,
        )
        page = await context.new_page()
        page.set_default_timeout(self.timeout_ms)
        try:
            yield page
        finally:
            await context.close()

    async def fetch_html(self, url: str) -> str:
        """Carga la URL y devuelve el HTML renderizado."""
        try:
            async with self._page() as page:
                response = await page.goto(
                    url,
                    wait_until=self.wait_until,
                    timeout=self.timeout_ms,
                )
                if response is None or not response.ok:
                    status = response.status if response else "sin respuesta"
                    raise FetchError(
                        f"Respuesta HTTP no válida para {url}: {status}"
                    )
                # Espera opcional a que aparezca el contenedor de mercados
                if self.selectors.market_root:
                    try:
                        await page.wait_for_selector(
                            self.selectors.market_root,
                            timeout=self.timeout_ms,
                        )
                    except PlaywrightTimeout:
                        logger.warning(
                            "market_root no visible en %s; "
                            "se parseará el HTML actual",
                            url,
                        )
                return await page.content()
        except PlaywrightTimeout as exc:
            raise FetchError(f"Timeout cargando {url}") from exc
        except ScraperError:
            raise
        except Exception as exc:
            raise FetchError(f"Error de red/navegador en {url}: {exc}") from exc

    async def scrape(self, url: str, *, bookmaker: str | None = None) -> MatchOdds:
        """Obtiene HTML y extrae cuotas."""
        html = await self.fetch_html(url)
        try:
            return parse_match_odds(
                html,
                url,
                self.selectors,
                bookmaker=bookmaker,
            )
        except ParseError:
            raise
        except Exception as exc:
            raise ParseError(f"Error inesperado parseando {url}: {exc}") from exc

    async def scrape_many(
        self,
        urls: Sequence[str],
        *,
        concurrency: int = 3,
        bookmaker: str | None = None,
    ) -> list[MatchOdds | ScraperError]:
        """Scrapea varias URLs con límite de concurrencia.

        Los fallos se devuelven como excepciones en la lista (no se propagan).
        """
        sem = asyncio.Semaphore(max(1, concurrency))

        async def _one(u: str) -> MatchOdds | ScraperError:
            async with sem:
                try:
                    return await self.scrape(u, bookmaker=bookmaker)
                except ScraperError as err:
                    logger.error("Fallo en %s: %s", u, err)
                    return err

        return list(await asyncio.gather(*(_one(u) for u in urls)))


async def scrape_match_odds(
    url: str,
    selectors: SelectorConfig,
    *,
    headless: bool = True,
    timeout_ms: int = 30_000,
    bookmaker: str | None = None,
) -> MatchOdds:
    """Atajo: scrapea una URL y cierra el navegador."""
    async with OddsScraper(
        selectors,
        headless=headless,
        timeout_ms=timeout_ms,
    ) as scraper:
        return await scraper.scrape(url, bookmaker=bookmaker)
