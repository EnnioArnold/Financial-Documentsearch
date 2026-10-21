"""Extraction from the SYNTHETIC fixtures (tests/fixtures/make_fixtures.py)."""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from finsearch.ingest import extract as extract_module
from finsearch.ingest.extract import ExtractError, extract, extract_html, extract_pdf

FIXTURES = Path(__file__).parents[1] / "fixtures"


def test_pdf_pages_are_numbered_from_one() -> None:
    result = extract_pdf((FIXTURES / "synthetic_report_de.pdf").read_bytes())
    assert [n for n, _ in result.pages] == [1, 2, 3, 4]
    assert result.n_pages == 4
    assert "Musterbank AG" in result.pages[0][1]
    assert "Zins-" in result.pages[0][1]  # raw text still has the hyphenation


def test_pdf_max_pages_caps_pages_but_keeps_total() -> None:
    result = extract_pdf((FIXTURES / "synthetic_report_de.pdf").read_bytes(), max_pages=2)
    assert [n for n, _ in result.pages] == [1, 2]
    assert result.n_pages == 4


def test_corrupt_pdf_raises_extract_error() -> None:
    with pytest.raises(ExtractError):
        extract_pdf(b"%PDF-1.7 this is not a pdf")


def test_encrypted_pdf_raises_extract_error() -> None:
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "secret")
    data = doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw="owner", user_pw="user")
    with pytest.raises(ExtractError, match="encrypted"):
        extract_pdf(data)


def test_html_article_is_page_one_without_navigation() -> None:
    result = extract((FIXTURES / "synthetic_speech_en.html").read_bytes(), "html")
    assert result.n_pages == 1
    page_no, text = result.pages[0]
    assert page_no == 1
    assert "policy rate unchanged" in text
    assert "Home | News | Contact" not in text


def test_html_falls_back_to_beautifulsoup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(extract_module.trafilatura, "extract", lambda *a, **k: None)
    html = b"<html><body><nav>menu</nav><script>x()</script><p>Visible text.</p></body></html>"
    text = extract_html(html).pages[0][1]
    assert "Visible text." in text
    assert "menu" not in text
    assert "x()" not in text


def test_extract_dispatches_on_kind() -> None:
    data = (FIXTURES / "synthetic_report_de.pdf").read_bytes()
    assert extract(data, "pdf", max_pages=1).pages[0][0] == 1
