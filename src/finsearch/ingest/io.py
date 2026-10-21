"""JSONL persistence for documents and chunks (one JSON object per line, dataclass fields)."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict
from datetime import date
from pathlib import Path

from finsearch.contracts import Chunk, DocumentMeta


def _to_json(record: DocumentMeta | Chunk) -> str:
    data = asdict(record)
    data["date"] = record.date.isoformat() if record.date else None
    return json.dumps(data, ensure_ascii=False)


def write_jsonl(path: Path, records: Iterable[DocumentMeta | Chunk]) -> int:
    """Overwrite path atomically; returns the number of lines. Same input => same bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    count = 0
    with tmp.open("w", encoding="utf-8", newline="\n") as fh:
        for record in records:
            fh.write(_to_json(record) + "\n")
            count += 1
    tmp.replace(path)
    return count


def _read_lines(path: Path) -> Iterator[dict[str, object]]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def _parse_date(value: object) -> date | None:
    return date.fromisoformat(str(value)) if value else None


def load_chunks(path: Path) -> Iterator[Chunk]:
    """Stream chunks back from chunks.jsonl (used by the indexer)."""
    for row in _read_lines(path):
        row["date"] = _parse_date(row.get("date"))
        yield Chunk(**row)  # type: ignore[arg-type]


def load_documents(path: Path) -> Iterator[DocumentMeta]:
    for row in _read_lines(path):
        row["date"] = _parse_date(row.get("date"))
        yield DocumentMeta(**row)  # type: ignore[arg-type]
