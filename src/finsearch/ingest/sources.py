"""Load sources.yaml into a flat list of documents to ingest."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

import yaml

DOC_TYPES = ("speech", "annual_report", "market_comment", "other")
DOC_ID_LENGTH = 16  # hex chars of sha256(url)


@dataclass(frozen=True)
class SourceDoc:
    """One entry of sources.yaml (a document to download), with its family-level notes."""

    url: str
    source: str  # family key, e.g. "snb"
    title: str
    doc_type: str
    lang_hint: str
    date: date | None = None
    max_pages: int | None = None  # PDF only: ignore pages after this one (annual reports are huge)
    license_note: str = ""
    robots_checked: str = ""


def doc_id_for(url: str) -> str:
    """Stable document id; also the cache file name, so it can never contain a path separator."""
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:DOC_ID_LENGTH]


def _parse_date(value: object) -> date | None:
    if value is None or isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def _parse_doc(source: str, family: dict[str, object], raw: dict[str, object]) -> SourceDoc:
    url = str(raw["url"])
    if urlparse(url).scheme != "https":
        raise ValueError(f"only https URLs are allowed: {url}")
    doc_type = str(raw["doc_type"])
    if doc_type not in DOC_TYPES:
        raise ValueError(f"unknown doc_type {doc_type!r} for {url}")
    max_pages = raw.get("max_pages")
    return SourceDoc(
        url=url,
        source=source,
        title=str(raw["title"]),
        doc_type=doc_type,
        lang_hint=str(raw.get("lang", "")),
        date=_parse_date(raw.get("date")),
        max_pages=int(str(max_pages)) if max_pages is not None else None,
        license_note=str(family.get("license_note", "")),
        robots_checked=str(family.get("robots_checked", "")),
    )


def load_sources(path: Path) -> list[SourceDoc]:
    """Read sources.yaml; keeps file order (=> deterministic output). Duplicate URLs are rejected."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    docs: list[SourceDoc] = []
    seen: set[str] = set()
    for source, family in data["sources"].items():
        for raw in family["documents"]:
            doc = _parse_doc(source, family, raw)
            if doc.url in seen:
                raise ValueError(f"duplicate URL in sources file: {doc.url}")
            seen.add(doc.url)
            docs.append(doc)
    return docs
