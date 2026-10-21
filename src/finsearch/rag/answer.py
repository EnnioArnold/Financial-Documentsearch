"""RAG orchestration: retrieve -> evidence gate -> (extractive | LLM) -> citation validation.

The evidence gate runs BEFORE any LLM call, so a question without support in the retrieved
chunks is answered with NOT_FOUND_TEXT no matter how chatty the model would have been.
"""

from __future__ import annotations

import logging
import time

from finsearch.config import Settings, get_settings
from finsearch.contracts import (
    NOT_FOUND_TEXT,
    Chunk,
    LLMClient,
    LLMError,
    RagAnswer,
    Searcher,
    SearchFilters,
)
from finsearch.rag.citations import build_citations, validate_citations
from finsearch.rag.prompt import build_messages
from finsearch.rag.providers.extractive import extractive_answer
from finsearch.rag.text import content_terms, detect_lang

log = logging.getLogger(__name__)

NOTE_NO_EVIDENCE = "Keine ausreichende Evidenz"


def evidence_coverage(question: str, chunks: list[Chunk]) -> float:
    """Share (0..1) of the question's content terms that occur anywhere in `chunks`.

    Terms are stopword-free and lightly stemmed (see text.py). The share is taken over the
    distinct terms of the question; 0.0 if the question has no content terms or no chunks.
    """
    terms = set(content_terms(question))
    if not terms or not chunks:
        return 0.0
    chunk_terms: set[str] = set()
    for c in chunks:
        chunk_terms.update(content_terms(f"{c.title} {c.text}", detect_lang(c.text)))
    return len(terms & chunk_terms) / len(terms)


def has_evidence(question: str, chunks: list[Chunk], min_coverage: float | None = None) -> bool:
    """Evidence gate: any chunks at all, and lexical coverage >= `min_coverage`.

    The threshold is the single setting `rag_min_coverage` (0 switches the lexical check off).
    Known limit: a German question about an English chunk matches only on shared terms
    (names, numbers), so cross-language questions may be rejected; tune on the golden set.
    """
    if not chunks:
        return False
    if min_coverage is None:
        min_coverage = get_settings().rag_min_coverage
    return min_coverage <= 0 or evidence_coverage(question, chunks) >= min_coverage


def _not_found(
    chunks: list[Chunk], note: str, provider: str, timings: dict[str, float]
) -> RagAnswer:
    return RagAnswer(NOT_FOUND_TEXT, [], chunks, note, False, provider, [], timings)


def _extractive(
    question: str, chunks: list[Chunk], note_prefix: str, timings: dict[str, float]
) -> RagAnswer:
    text = extractive_answer(question, chunks)
    check = validate_citations(text, len(chunks)) if text else None
    if check is None or not check.found:
        return _not_found(chunks, NOTE_NO_EVIDENCE, "extractive", timings)
    return RagAnswer(
        answer=check.text,
        citations=build_citations(check.valid, chunks),
        used_chunks=chunks,
        confidence_note=f"{note_prefix}Extraktive Antwort aus {len(check.valid)} Quellen",
        found=True,
        provider="extractive",
        dropped_citations=check.dropped,
        timings_ms=timings,
    )


def answer_question(
    question: str,
    searcher: Searcher,
    llm: LLMClient,
    *,
    method: str = "hybrid",
    k: int | None = None,
    filters: SearchFilters | None = None,
    settings: Settings | None = None,
) -> RagAnswer:
    """Answer `question` from retrieved chunks; never raises for LLM problems.

    `used_chunks[n - 1]` is source [n] in the answer (same order as the prompt).
    """
    settings = settings or get_settings()
    t0 = time.perf_counter()
    result = searcher.search(question, method=method, k=k or settings.rag_top_k, filters=filters)
    chunks = [h.chunk for h in result.hits if h.score >= settings.rag_min_score]
    timings = {"retrieve": (time.perf_counter() - t0) * 1000, "llm": 0.0}

    if not has_evidence(question, chunks, settings.rag_min_coverage):
        return _not_found(chunks, NOTE_NO_EVIDENCE, llm.name, timings)
    if llm.name == "extractive":
        return _extractive(question, chunks, "", timings)

    t1 = time.perf_counter()
    try:
        response = llm.generate(build_messages(question, chunks))
    except LLMError as exc:  # message is key-free by construction (llm_base)
        log.warning("LLM %s failed, using extractive fallback: %s", llm.name, exc)
        timings["llm"] = (time.perf_counter() - t1) * 1000
        return _extractive(question, chunks, f"LLM '{llm.name}' nicht verfügbar. ", timings)
    timings["llm"] = (time.perf_counter() - t1) * 1000

    if NOT_FOUND_TEXT.lower() in response.text.lower():
        return _not_found(chunks, "Das Modell fand in den Quellen keine Antwort", llm.name, timings)
    check = validate_citations(response.text, len(chunks))
    if not check.found:
        return _not_found(chunks, "Antwort ohne gültige Quellenangabe verworfen", llm.name, timings)
    note = f"Antwort von {llm.name} aus {len(check.valid)} Quellen"
    if check.dropped:
        note += f"; {len(check.dropped)} ungültige Quellenangabe(n) entfernt"
    if check.unsupported:
        note += f"; {check.unsupported} Aussage(n) ohne Beleg markiert"
    return RagAnswer(
        answer=check.text,
        citations=build_citations(check.valid, chunks),
        used_chunks=chunks,
        confidence_note=note,
        found=True,
        provider=llm.name,
        dropped_citations=check.dropped,
        timings_ms=timings,
    )
