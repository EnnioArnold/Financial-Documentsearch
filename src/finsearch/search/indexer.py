"""Bulk indexing: chunks.jsonl -> embeddings -> Elasticsearch. Idempotent via _id = chunk_id."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterable, Iterator
from datetime import date
from pathlib import Path
from typing import Any

from elasticsearch import Elasticsearch, helpers
from tqdm import tqdm

from finsearch.config import Settings, get_settings
from finsearch.contracts import Chunk, Embedder
from finsearch.search.embedder import get_embedder
from finsearch.search.es_client import get_es
from finsearch.search.mapping import chunk_to_doc, ensure_index

log = logging.getLogger(__name__)

VALID_LANGS = {"de", "en", "other"}


def _parse_chunk(raw: dict[str, Any]) -> Chunk:
    raw_date = raw.get("date")
    return Chunk(
        chunk_id=str(raw["chunk_id"]),
        doc_id=str(raw["doc_id"]),
        ord=int(raw["ord"]),
        text=str(raw["text"]),
        page=int(raw.get("page") or 1),
        source=str(raw["source"]),
        title=str(raw.get("title") or ""),
        url=str(raw.get("url") or ""),
        lang=raw["lang"] if raw.get("lang") in VALID_LANGS else "other",
        doc_type=str(raw.get("doc_type") or "other"),
        date=date.fromisoformat(str(raw_date)[:10]) if raw_date else None,
    )


def load_chunks(path: Path) -> Iterator[Chunk]:
    """Read chunks.jsonl tolerantly: blank/broken/incomplete lines are skipped with a warning."""
    with Path(path).open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                chunk = _parse_chunk(json.loads(line))
            except (ValueError, KeyError, TypeError) as exc:
                log.warning("%s:%d skipped (%s: %s)", path, line_no, type(exc).__name__, exc)
                continue
            if chunk.text.strip():
                yield chunk


def _batches(items: Iterable[Chunk], size: int) -> Iterator[list[Chunk]]:
    batch: list[Chunk] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def index_chunks(
    chunks: Iterable[Chunk],
    settings: Settings | None = None,
    embedder: Embedder | None = None,
    index_name: str | None = None,
    es: Elasticsearch | None = None,
) -> dict[str, Any]:
    """Embed and bulk-index chunks. Re-running with the same chunk_ids overwrites, never duplicates.

    Without `index_name` the default versioned index is created (if needed) and the alias is pointed
    at it when the alias does not exist yet. Returns {"indexed", "failed", "took_s"}.
    """
    s = settings or get_settings()
    client = es or get_es(s)
    emb = embedder or get_embedder(s)
    target = ensure_index(client, s, index_name)
    started = time.perf_counter()
    indexed = failed = 0
    materialised = list(chunks)
    with tqdm(total=len(materialised), desc="indexing", unit="chunk", disable=None) as bar:
        for batch in _batches(materialised, s.embedding_batch_size):
            vectors = emb.embed_passages([c.text for c in batch])
            actions = [
                {"_index": target, "_id": c.chunk_id, "_source": chunk_to_doc(c, v)}
                for c, v in zip(batch, vectors, strict=True)
            ]
            ok, errors = helpers.bulk(client, actions, raise_on_error=False)
            indexed += ok
            failed += len(errors) if isinstance(errors, list) else int(errors)
            for err in errors[:3] if isinstance(errors, list) else []:
                log.error("bulk error: %s", err)
            bar.update(len(batch))
    client.indices.refresh(index=target)
    stats = {
        "indexed": indexed,
        "failed": failed,
        "took_s": round(time.perf_counter() - started, 3),
    }
    log.info("indexed into %s: %s", target, stats)
    return stats


def index_chunks_jsonl(
    path: Path,
    settings: Settings | None = None,
    embedder: Embedder | None = None,
    index_name: str | None = None,
    es: Elasticsearch | None = None,
) -> dict[str, Any]:
    """Index a chunks.jsonl file (one Chunk per line). See `index_chunks`."""
    return index_chunks(load_chunks(path), settings, embedder, index_name, es)
