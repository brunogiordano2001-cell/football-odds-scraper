from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys

from football_odds_scraper.models import SelectorConfig
from football_odds_scraper.probability import fair_probabilities, overround
from football_odds_scraper.scraper import OddsScraper
from football_odds_scraper.selectors import get_preset


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extrae cuotas 1X2 y Más/Menos 2.5 de partidos de fútbol.",
    )
    p.add_argument(
        "urls",
        nargs="+",
        help="URL(s) de la página del partido",
    )
    p.add_argument(
        "--preset",
        default="generic",
        help="Plantilla de selectores (generic, data_odds_spa, ...)",
    )
    p.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--timeout", type=int, default=30_000, help="Timeout en ms")
    p.add_argument("--bookmaker", default=None)
    p.add_argument(
        "--fair",
        action="store_true",
        help="Incluir probabilidades sin overround",
    )
    p.add_argument(
        "--method",
        choices=["multiplicative", "additive", "power"],
        default="multiplicative",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p


async def _run(args: argparse.Namespace) -> int:
    selectors: SelectorConfig = get_preset(args.preset)
    exit_code = 0

    async with OddsScraper(
        selectors,
        headless=args.headless,
        timeout_ms=args.timeout,
    ) as scraper:
        results = await scraper.scrape_many(args.urls)

    output: list[dict] = []
    for item in results:
        if isinstance(item, Exception):
            output.append({"error": str(item)})
            exit_code = 1
            continue

        row = item.to_dict()
        row["overround"] = {
            "1x2": round(overround(item.market_1x2), 6),
            "over_under": round(overround(item.market_over_under), 6),
        }
        if args.fair:
            row["fair_probabilities"] = fair_probabilities(
                item, method=args.method
            )
        output.append(row)

    print(json.dumps(output, indent=2, ensure_ascii=False))
    return exit_code


def main() -> None:
    args = _build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    raise SystemExit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
