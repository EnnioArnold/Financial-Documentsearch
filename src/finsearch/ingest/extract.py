"""Raw bytes -> list of (page_no, text). Page numbers are 1-based.

PDF: one entry per page (pymupdf). HTML: the whole article is page 1 (trafilatura, with a
BeautifulSoup fallback) - an HTML page has no real pages, so citing "page 1" is the honest answer.
"""

from __future__ import annotations

from dataclasses import dataclass

import pymupdf
import trafilatura
from bs4 import BeautifulSoup

Page = tuple[int, str]

HTML_NOISE_TAGS = ("script", "style", "nav", "header", "footer", "aside", "form", "noscript")


class ExtractError(RuntimeError):
    """The file could not be parsed (corrupt, encrypted, wrong type)."""


@dataclass(frozen=True)
class Extracted:
    pages: list[Page]
    n_pages: int  # pages in the source document (may exceed len(pages) when max_pages applies)


def extract_pdf(data: bytes, max_pages: int | None = None) -> Extracted:
    try:
        with pymupdf.open(stream=data, filetype="pdf") as doc:
            if doc.needs_pass:
                raise ExtractError("PDF is encrypted")
            total = doc.page_count
            last = total if max_pages is None else min(total, max_pages)
            pages = [(i + 1, str(doc[i].get_text("text"))) for i in range(last)]
    except ExtractError:
        raise
    except Exception as exc:  # pymupdf raises a variety of RuntimeError subclasses
        raise ExtractError(f"cannot read PDF: {exc}") from exc
    return Extracted(pages, total)


def extract_html(data: bytes) -> Extracted:
    html = data.decode("utf-8", errors="replace")
    text = trafilatura.extract(html, include_comments=False, include_tables=True)
    if not text or not text.strip():
        text = _bs4_text(html)
    return Extracted([(1, text)], 1)


def _bs4_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(HTML_NOISE_TAGS):
        tag.decompose()
    return soup.get_text("\n")


def extract(data: bytes, kind: str, max_pages: int | None = None) -> Extracted:
    """kind is "pdf" or "html" (see fetch.detect_kind)."""
    if kind == "pdf":
        return extract_pdf(data, max_pages)
    return extract_html(data)
