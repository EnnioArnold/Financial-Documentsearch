"""Ingestion pipeline: sources.yaml -> fetch -> extract -> clean -> chunk -> JSONL (-> index).

Run: python -m finsearch.ingest.pipeline [--no-index] [--limit N] [--source KEY]
A failing document is logged and skipped; the output files are fully rewritten on every run.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import math
import statistics
import sys
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from finsearch.config import Settings, get_settings
from finsearch.contracts import Chunk, DocumentMeta, Lang
from finsearch.ingest.chunk import chunk_document, estimate_tokens
from finsearch.ingest.clean import clean_pages, detect_language
from finsearch.ingest.extract import extract
from finsearch.ingest.fetch import Fetcher, Resolver, resolve_host
from finsearch.ingest.io import write_jsonl
from finsearch.ingest.sources import SourceDoc, doc_id_for, load_sources
from finsearch.logging import setup_logging

log = logging.getLogger(__name__)

DOCUMENTS_FILE = "documents.jsonl"
CHUNKS_FILE = "chunks.jsonl"
MIN_TEXT_CHARS = 200  # less text than this = scanned PDF / empty shell page => skip the document
P95 = 0.95


@dataclass
class RunResult:
    documents: list[DocumentMeta] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)
    failures: list[tuple[str, str]] = field(default_factory=list)  # (url, reason)


def process_document(entry: SourceDoc, fetcher: Fetcher) -> tuple[DocumentMeta, list[Chunk]]:
    """Fetch -> extract -> clean -> detect language -> chunk one document."""
    fetched = fetcher.fetch(entry.url)
    extracted = extract(fetched.read(), fetched.kind, entry.max_pages)
    pages = clean_pages(extracted.pages)
    text = " ".join(page_text for _, page_text in pages)
    if len(text) < MIN_TEXT_CHARS:
        raise ValueError(f"only {len(text)} characters of text after cleaning")
    lang: Lang = detect_language(text)
    if entry.lang_hint and entry.lang_hint != lang:
        log.warning("language hint %s != detected %s for %s", entry.lang_hint, lang, entry.url)
    doc = DocumentMeta(
        doc_id=doc_id_for(entry.url),
        source=entry.source,
        title=entry.title,
        url=entry.url,
        lang=lang,
        doc_type=entry.doc_type,
        date=entry.date,
        n_pages=extracted.n_pages,
        content_hash=fetched.content_hash,
    )
    return doc, chunk_document(doc, pages)


def run(
    settings: Settings | None = None,
    limit: int | None = None,
    source: str | None = None,
    client: httpx.Client | None = None,
    resolver: Resolver = resolve_host,
) -> RunResult:
    """Process all (or a subset of) sources and rewrite documents.jsonl / chunks.jsonl."""
    settings = settings or get_settings()
    entries = load_sources(settings.sources_file)
    fetcher = Fetcher(settings, [e.url for e in entries], client=client, resolver=resolver)
    selected = [e for e in entries if source is None or e.source == source]
    if limit is not None:
        selected = selected[:limit]
    result = RunResult()
    hashes: dict[str, str] = {}
    for entry in selected:
        try:
            doc, chunks = process_document(entry, fetcher)
            if doc.content_hash in hashes:
                raise ValueError(f"duplicate content of {hashes[doc.content_hash]}")
        except Exception as exc:  # one bad document must never stop the run
            log.warning("skipped %s: %s", entry.url, exc)
            result.failures.append((entry.url, f"{type(exc).__name__}: {exc}"))
            continue
        hashes[doc.content_hash] = doc.url
        result.documents.append(doc)
        result.chunks.extend(chunks)
        log.info("ok %s: %d chunks", entry.url, len(chunks))
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(settings.processed_dir / DOCUMENTS_FILE, result.documents)
    write_jsonl(settings.processed_dir / CHUNKS_FILE, result.chunks)
    return result


def _percentile(sorted_values: list[int], q: float) -> int:
    """Nearest-rank percentile."""
    return sorted_values[max(0, math.ceil(q * len(sorted_values)) - 1)]


def format_stats(result: RunResult) -> str:
    lengths = sorted(estimate_tokens(c.text) for c in result.chunks)
    lines = [
        f"documents: {len(result.documents)}   chunks: {len(result.chunks)}   "
        f"failures: {len(result.failures)}",
        "documents per source: " + _counts(d.source for d in result.documents),
        "documents per language: " + _counts(d.lang for d in result.documents),
        "chunks per language: " + _counts(c.lang for c in result.chunks),
    ]
    if lengths:
        lines.append(
            f"chunk length (estimated tokens): median {statistics.median(lengths):.0f}, "
            f"p95 {_percentile(lengths, P95)}, max {lengths[-1]}"
        )
    lines += [f"FAILED {url}: {reason}" for url, reason in result.failures]
    return "\n".join(lines)


def _counts(values: Iterable[str]) -> str:
    return ", ".join(f"{k}={v}" for k, v in sorted(Counter(values).items()))


def index_chunks(chunks_path: Path) -> None:
    """Hand the JSONL to the search package (implemented by the indexer; needs Elasticsearch)."""
    try:
        indexer = importlib.import_module("finsearch.search.indexer")
        index_fn = indexer.index_chunks_jsonl
    except (ImportError, AttributeError) as exc:
        raise SystemExit(
            f"cannot index: finsearch.search.indexer.index_chunks_jsonl is not available ({exc}). "
            "Re-run with --no-index to only write the JSONL files."
        ) from exc
    log.info("indexing result: %s", index_fn(chunks_path))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="finsearch.ingest.pipeline", description=__doc__)
    parser.add_argument("--no-index", action="store_true", help="only write the JSONL files")
    parser.add_argument("--limit", type=int, help="process at most N documents")
    parser.add_argument("--source", help="only this source family key, e.g. snb")
    args = parser.parse_args(argv)
    setup_logging()
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one INFO line per request is noise
    settings = get_settings()
    result = run(settings, limit=args.limit, source=args.source)
    print(format_stats(result))
    if not args.no_index and result.chunks:
        index_chunks(settings.processed_dir / CHUNKS_FILE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
