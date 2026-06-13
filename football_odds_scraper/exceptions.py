from __future__ import annotations


class ScraperError(Exception):
    """Error base del scraper."""


class FetchError(ScraperError):
    """Fallo al cargar la página con Playwright."""


class ParseError(ScraperError):
    """No se pudieron localizar o interpretar las cuotas."""

    def __init__(self, message: str, *, field: str | None = None) -> None:
        super().__init__(message)
        self.field = field
