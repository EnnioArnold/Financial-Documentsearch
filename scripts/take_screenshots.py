"""Take the UI screenshots for the README/docs from a RUNNING app (default http://127.0.0.1:8000).

    .venv/bin/python scripts/take_screenshots.py [--base-url URL] [--out-dir docs/screenshots]

Writes ui_search.png, ui_compare.png, ui_ask.png (1280x900) and ui_mobile.png (390 wide).
Run it against the real, fully indexed corpus; override the queries if your corpus differs.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from urllib.parse import urlencode

from playwright.sync_api import Page, sync_playwright

log = logging.getLogger("screenshots")

DESKTOP = {"width": 1280, "height": 900}
MOBILE = {"width": 390, "height": 844}
TIMEOUT_MS = 60_000  # first query loads the embedding model


def shot(page: Page, url: str, ready: str, out: Path) -> None:
    page.goto(url)
    page.wait_for_selector(ready, timeout=TIMEOUT_MS)
    page.wait_for_selector(".spinner", state="detached", timeout=TIMEOUT_MS)
    page.screenshot(path=str(out))
    log.info("wrote %s", out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--out-dir", type=Path, default=Path("docs/screenshots"))
    parser.add_argument("--search-query", default="Leitzins Inflation")
    parser.add_argument("--compare-query", default="Teuerung und Zinsen")
    parser.add_argument(
        "--question", default="Welche Risiken für die Finanzstabilität nennt die SNB?"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    base = args.base_url.rstrip("/")

    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(
            viewport=DESKTOP, reduced_motion="reduce", color_scheme="light", locale="de-CH"
        )
        page = ctx.new_page()
        search = urlencode({"q": args.search_query, "method": "hybrid"})
        shot(page, f"{base}/?{search}", "#results .card", args.out_dir / "ui_search.png")
        compare = urlencode({"q": args.compare_query})
        shot(page, f"{base}/vergleich?{compare}", "#columns .col", args.out_dir / "ui_compare.png")
        ask = urlencode({"q": args.question, "method": "hybrid"})
        shot(page, f"{base}/frage?{ask}", "#answer:not([hidden])", args.out_dir / "ui_ask.png")
        ctx.close()

        mobile = browser.new_context(
            viewport=MOBILE, device_scale_factor=2, reduced_motion="reduce", locale="de-CH"
        )
        page = mobile.new_page()
        shot(page, f"{base}/?{search}", "#results .card", args.out_dir / "ui_mobile.png")
        browser.close()


if __name__ == "__main__":
    main()
