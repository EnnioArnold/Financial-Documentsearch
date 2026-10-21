"""Browser smoke test: real app (uvicorn subprocess) on a throwaway synthetic index, headless Chromium.

Run: .venv/bin/python -m pytest -m e2e -o addopts="" tests/e2e
"""

from __future__ import annotations

import socket
import subprocess
import sys
import time
import uuid
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from playwright.sync_api import Browser, Page, expect, sync_playwright

from finsearch.config import Settings
from finsearch.search.es_client import get_es

pytestmark = pytest.mark.e2e

SERVER = Path(__file__).with_name("serve_synthetic.py")
START_TIMEOUT_S = 90
UI_TIMEOUT_MS = 15_000


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture(scope="module")
def base_url() -> Iterator[str]:
    port, name = free_port(), f"finsearch-test-e2e-{uuid.uuid4().hex[:8]}"
    proc = subprocess.Popen([sys.executable, str(SERVER), "--port", str(port), "--name", name])
    url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + START_TIMEOUT_S
        while True:
            assert proc.poll() is None, "server exited during startup"
            try:
                if httpx.get(f"{url}/health", timeout=2).json()["status"] == "ok":
                    break
            except httpx.HTTPError:
                pass
            assert time.monotonic() < deadline, "server did not become healthy"
            time.sleep(0.5)
        yield url
    finally:
        proc.terminate()
        proc.wait(timeout=30)
        es = get_es(Settings())
        for index in (
            name,
            f"{name}-queries",
        ):  # belt and braces: the server cleans up on SIGTERM too
            es.indices.delete(index=index, ignore_unavailable=True)
        es.close()


@pytest.fixture(scope="module")
def browser() -> Iterator[Browser]:
    with sync_playwright() as p:
        b = p.chromium.launch()
        yield b
        b.close()


@pytest.fixture
def page(browser: Browser, base_url: str) -> Iterator[Page]:
    """A page that fails the test on any console error, JS exception, failed request or alert()."""
    context = browser.new_context(locale="de-CH", viewport={"width": 1280, "height": 900})
    pg = context.new_page()
    pg.set_default_timeout(UI_TIMEOUT_MS)
    problems: list[str] = []
    pg.on("console", lambda m: problems.append(f"console: {m.text}") if m.type == "error" else None)
    pg.on("pageerror", lambda e: problems.append(f"pageerror: {e}"))
    pg.on("requestfailed", lambda r: problems.append(f"request failed: {r.url}"))
    pg.on("dialog", lambda d: (problems.append(f"dialog: {d.message}"), d.dismiss()))
    pg.goto(base_url)
    yield pg
    context.close()
    assert not problems, problems


def choose(page: Page, method: str) -> None:
    page.locator(f'label.seg:has(input[value="{method}"])').click()


def search(page: Page, query: str, button: str = "Suchen") -> None:
    page.fill("#q", query)
    page.get_by_role("button", name=button, exact=True).click()


def test_search_methods_and_xss(page: Page) -> None:
    expect(page.get_by_role("heading", level=1)).to_have_text("Dokumente durchsuchen")
    search(page, "Leitzins gesenkt")
    first = page.locator("#results .card").first
    expect(first).to_contain_text("SYNTHETIC SNB Rede")
    expect(first.locator("em").first).to_have_text("Leitzins")
    expect(page.locator("#status")).to_contain_text("Hybrid")
    assert page.url.endswith("q=Leitzins+gesenkt&method=hybrid&k=10")  # shareable state

    choose(page, "knn")
    search(page, "Leitzins gesenkt")
    expect(page.locator("#status")).to_contain_text("kNN")
    expect(page.locator("#results .card").first).to_be_visible()

    choose(page, "bm25")
    search(page, "Inflation outlook")
    hostile = page.locator("#results .card", has_text="SYNTHETIC bank annual report").first
    expect(hostile).to_contain_text("<script>alert(1)</script>")  # shown as text, never executed
    assert page.locator("#results script").count() == 0

    choose(page, "hybrid")
    page.get_by_text("Filter", exact=True).click()
    page.select_option("#f-source", "fed")
    search(page, "Zinsen")
    expect(page.locator("#results .card").first).to_be_visible()
    badges = page.locator("#results .card .badge").evaluate_all(
        "els => els.map(e => e.textContent)"
    )
    assert badges and set(badges) == {
        "fed"
    }  # CSS upper-cases the label; the text node is lower-case


def test_search_without_hits_and_without_query(page: Page) -> None:
    choose(page, "bm25")
    search(page, "xyzzyqwerty")
    expect(page.locator("#results .empty")).to_contain_text("Keine Treffer")
    page.fill("#q", "   ")
    page.get_by_role("button", name="Suchen", exact=True).click()
    expect(page.locator("#error")).to_contain_text("Bitte eine Suchanfrage eingeben")


def test_example_chip_runs_a_search(page: Page) -> None:
    page.get_by_role("button", name="Was sagt die Fed zur Inflation?").click()
    expect(page.locator("#q")).to_have_value("Was sagt die Fed zur Inflation?")
    expect(page.locator("#results .card").first).to_be_visible()


def test_compare_shows_three_methods_and_overlap(page: Page) -> None:
    page.get_by_role("link", name="Vergleich").click()
    search(page, "Leitzins Zinsen Inflation", button="Vergleichen")
    expect(page.locator("#columns .col")).to_have_count(3)
    expect(page.locator("#overlap")).to_contain_text("Hybrid-Treffern")
    expect(page.locator("#columns .badge.multi").first).to_be_visible()
    for method in ("bm25", "knn", "hybrid"):
        expect(page.locator(f'.col[data-method="{method}"] .latency strong')).to_contain_text("ms")
    # hovering a hit highlights the same chunk in the other lists
    card = page.locator('.col[data-method="bm25"] .card').first
    card.hover()
    chunk = card.get_attribute("data-chunk")
    expect(page.locator(f'.card.linked[data-chunk="{chunk}"]')).not_to_have_count(0)


def test_ask_answer_with_clickable_sources_and_not_found(page: Page) -> None:
    page.get_by_role("link", name="Frage stellen").click()
    search(page, "Wie hoch ist der Leitzins der Nationalbank?", button="Antwort suchen")
    answer = page.locator("#answer")
    expect(answer).to_be_visible()
    expect(answer.locator(".answer-text")).to_contain_text("Leitzins")
    expect(answer).to_contain_text("Anbieter: extractive")
    cite = page.locator("#answer a.cite").first
    number = cite.inner_text().strip("[]")
    cite.click()
    expect(page.locator(f"#src-{number}")).to_be_focused()
    expect(page.locator("#sources .card").first).to_contain_text("Seite")

    search(page, "Wie funktioniert Quantenverschränkung?", button="Antwort suchen")
    expect(answer).to_contain_text("In den Quellen nicht gefunden")
    expect(answer).to_have_class("answer-panel not-found")


def test_layout_has_no_horizontal_scroll_on_a_small_phone(page: Page, base_url: str) -> None:
    page.set_viewport_size({"width": 360, "height": 740})
    for path in ("/?q=Leitzins", "/vergleich?q=Leitzins", "/frage?q=Leitzins+der+Nationalbank"):
        page.goto(base_url + path)
        loaded = page.locator("#results .card, #columns .col, #answer:not([hidden])")
        expect(loaded.first).to_be_visible()
        expect(page.locator(".spinner")).to_have_count(0)
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth"), path


def test_health_banner_not_shown_and_footer_notice(page: Page) -> None:
    expect(page.locator("#corpus-info")).to_contain_text("Abschnitte indexiert")
    expect(page.locator("#system-banner")).to_be_hidden()
    expect(page.get_by_role("contentinfo")).to_contain_text("nicht für Produktion")
