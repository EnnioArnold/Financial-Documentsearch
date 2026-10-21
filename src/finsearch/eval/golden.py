"""Golden-set format and loader (JSONL, one question per line). See eval/README.md."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

QUESTION_TYPES = ("fact", "paraphrase", "number_date", "comparison", "unanswerable")
LANGS = ("de", "en")
PROVENANCES = ("human", "llm_suggested_verified")


class GoldenSetError(ValueError):
    """The golden set file is malformed (message names file, line and problem)."""


@dataclass(frozen=True)
class GoldenQuestion:
    id: str
    question: str
    lang: str
    type: str
    relevant_chunk_ids: tuple[str, ...]
    relevant_doc_ids: tuple[str, ...]
    answerable: bool
    reference_answer: str | None
    provenance: str

    @property
    def doc_level(self) -> bool:
        """True if relevance is judged on doc_id (no chunk ids given)."""
        return not self.relevant_chunk_ids

    @property
    def relevant_keys(self) -> frozenset[str]:
        return frozenset(self.relevant_doc_ids if self.doc_level else self.relevant_chunk_ids)


def _str_list(obj: dict[str, object], key: str) -> tuple[str, ...]:
    value = obj.get(key)
    if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
        raise ValueError(f"'{key}' must be a list of strings")
    return tuple(value)


def parse_question(obj: object) -> GoldenQuestion:
    """Validate one decoded JSON object. Raises ValueError with a readable message."""
    if not isinstance(obj, dict):
        raise ValueError("line must be a JSON object")
    for key in ("id", "question"):
        if not isinstance(obj.get(key), str) or not obj[key].strip():
            raise ValueError(f"'{key}' must be a non-empty string")
    for key, allowed in (("lang", LANGS), ("type", QUESTION_TYPES), ("provenance", PROVENANCES)):
        if obj.get(key) not in allowed:
            raise ValueError(f"'{key}' must be one of {list(allowed)}, got {obj.get(key)!r}")
    if not isinstance(obj.get("answerable"), bool):
        raise ValueError("'answerable' must be true or false")
    ref = obj.get("reference_answer")
    if ref is not None and not isinstance(ref, str):
        raise ValueError("'reference_answer' must be a string or null")
    chunks, docs = _str_list(obj, "relevant_chunk_ids"), _str_list(obj, "relevant_doc_ids")
    answerable = obj["answerable"]
    if answerable and not (chunks or docs):
        raise ValueError("answerable question needs relevant_chunk_ids or relevant_doc_ids")
    if (obj["type"] == "unanswerable") == answerable:
        raise ValueError(
            "type 'unanswerable' must be paired with answerable=false (and vice versa)"
        )
    return GoldenQuestion(
        id=obj["id"], question=obj["question"], lang=obj["lang"], type=obj["type"],
        relevant_chunk_ids=chunks, relevant_doc_ids=docs, answerable=answerable,
        reference_answer=ref, provenance=obj["provenance"],
    )  # fmt: skip


def load_golden(path: Path) -> list[GoldenQuestion]:
    """Load and validate the golden set; blank lines are ignored, ids must be unique."""
    if not path.is_file():
        raise GoldenSetError(f"golden set not found: {path}")
    questions: list[GoldenQuestion] = []
    seen: set[str] = set()
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            q = parse_question(json.loads(line))
            if q.id in seen:
                raise ValueError(f"duplicate id {q.id!r}")
        except (ValueError, json.JSONDecodeError) as exc:
            raise GoldenSetError(f"{path}:{lineno}: {exc}") from exc
        seen.add(q.id)
        questions.append(q)
    if not questions:
        raise GoldenSetError(f"golden set is empty: {path}")
    return questions
