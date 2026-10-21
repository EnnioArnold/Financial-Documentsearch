"""RAG orchestration: retrieve -> evidence gate -> (extractive | LLM) -> citation validation.

The evidence gate runs BEFORE any LLM call, so a question without support in the retrieved
chunks is answered with NOT_FOUND_TEXT no matter how chatty the model would have been.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

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
    """Best single-chunk share (0..1) of the question's content terms found in a chunk.

    The question is stemmed in the language of each chunk, so a one-word English question is
    not stemmed as German. Taking the best chunk (not the union of all chunks) stops generic
    terms spread over several chunks from counting as evidence. 0.0 without terms or chunks.
    """
    best = 0.0
    for c in chunks:
        lang = detect_lang(c.text)
        terms = set(content_terms(question, lang))
        if not terms:
            continue
        chunk_terms = set(content_terms(f"{c.title} {c.text}", lang))
        best = max(best, len(terms & chunk_terms) / len(terms))
    return best


def has_evidence(
    question: str,
    chunks: list[Chunk],
    min_coverage: float | None = None,
    bm25_top: Callable[[], float] | None = None,
    min_bm25: float | None = None,
) -> bool:
    """Evidence gate: chunks exist and (lexical coverage is high OR the BM25 top score is high).

    Thresholds (`rag_min_coverage`, `rag_min_bm25`) were tuned on the golden set, see
    docs/DECISIONS.md ADR-007. `bm25_top` is evaluated lazily, only if coverage is too low.
    min_coverage <= 0 switches the gate off.
    """
    if not chunks:
        return False
    settings = get_settings()
    if min_coverage is None:
        min_coverage = settings.rag_min_coverage
    if min_coverage <= 0 or evidence_coverage(question, chunks) >= min_coverage:
        return True
    if bm25_top is None:
        return False
    return bm25_top() >= (settings.rag_min_bm25 if min_bm25 is None else min_bm25)


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
    chunks = [h.chunk for h in result.hits]
    timings = {"retrieve": (time.perf_counter() - t0) * 1000, "llm": 0.0}

    def bm25_top() -> float:
        hits = (
            result.hits
            if method == "bm25"
            else searcher.search(question, method="bm25", k=1, filters=filters).hits
        )
        return hits[0].score if hits else 0.0

    if not has_evidence(
        question, chunks, settings.rag_min_coverage, bm25_top, settings.rag_min_bm25
    ):
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
