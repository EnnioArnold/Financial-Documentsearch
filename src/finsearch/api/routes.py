"""HTTP endpoints. Handlers stay thin: validate -> call SearchService / RAG -> shape the response."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Annotated, Any

from elasticsearch import ApiError as EsApiError
from elasticsearch import NotFoundError, TransportError
from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import HTMLResponse

from finsearch.api.deps import ApiError, Services, enforce_rate_limit
from finsearch.api.schemas import (
    AskRequest,
    AskResponse,
    CompareParams,
    CompareResponse,
    DocumentOut,
    DocumentsParams,
    DocumentsResponse,
    SearchParams,
    SearchResponse,
)
from finsearch.contracts import SearchFilters, SearchResult
from finsearch.rag.answer import answer_question
from finsearch.rag.text import detect_lang

log = logging.getLogger(__name__)

HTTP_UNAVAILABLE = 503
HEALTH_TIMEOUT_S = 3
COMPARE_METHODS = ("bm25", "knn", "hybrid")
EMPTY_INDEX_HINT = "Index leer – zuerst `make ingest`"
EXAMPLES = [
    "Welche Risiken für die Finanzstabilität nennt die SNB?",
    "Welche Rolle spielt der Franken für die Geldpolitik?",
    "Was sagt die Fed zur Inflation?",
    "What did the Federal Reserve say about the labor market?",
    "How does the SNB assess the Swiss franc?",
]

router = APIRouter()
RateLimited = Depends(enforce_rate_limit)


def get_services(request: Request) -> Services:
    services: Services = request.app.state.services
    return services


Svc = Annotated[Services, Depends(get_services)]


@contextmanager
def reranker_guard(method: str) -> Iterator[None]:
    """A model that cannot be loaded (offline, no disk) must become a clean 503, not a traceback."""
    try:
        yield
    except (TransportError, EsApiError, ValueError, ApiError):
        raise  # handled elsewhere (Elasticsearch down / bad arguments)
    except Exception as exc:
        if method != "hybrid_rerank":
            raise
        log.exception("reranker unavailable")
        raise ApiError(
            HTTP_UNAVAILABLE,
            "reranker_unavailable",
            "Das Reranker-Modell konnte nicht geladen werden.",
            hint="Methode 'Hybrid' verwenden oder das Modell vorab herunterladen.",
        ) from exc


def _begin(request: Request, endpoint: str, method: str, query: str) -> None:
    """Label the request for the query log / runtime stats before any work, so failures are labelled too."""
    request.state.obs = {
        "endpoint": endpoint,
        "method": method,
        "query": query,
        "lang": detect_lang(query),
    }


def _finish(
    request: Request, n_hits: int, timings: dict[str, float], found: bool | None = None
) -> None:
    """Add the outcome (see api/middleware.py for the fields the query log reads)."""
    request.state.obs.update(n_hits=n_hits, timings_ms=timings, found=found)


def _search(svc: Services, q: str, method: str, k: int, filters: SearchFilters) -> SearchResult:
    with reranker_guard(method):
        return svc.search().search(q, method=method, k=k, filters=filters)


# ---- JSON API ----------------------------------------------------------------------------------


@router.get("/health")
def health(svc: Svc) -> dict[str, Any]:
    """Always 200: a degraded service is still a running service (the UI explains the rest)."""
    llm, llm_note = svc.llm()
    reachable, chunks, hint = False, 0, None
    index = svc.settings.index_name
    try:
        search = svc.search()
        index = search.index
        chunks = search.es.options(request_timeout=HEALTH_TIMEOUT_S, max_retries=0).count(
            index=index
        )["count"]
        reachable = True
    except NotFoundError:
        reachable, hint = True, EMPTY_INDEX_HINT
    except Exception as exc:  # health must never fail
        log.warning("health: Elasticsearch check failed: %s", exc)
        hint = "Elasticsearch nicht erreichbar – läuft `make up`?"
    if reachable and chunks == 0:
        hint = EMPTY_INDEX_HINT
    return {
        "status": "ok" if reachable and chunks > 0 else "degraded",
        "api": "ok",
        "elasticsearch": {"reachable": reachable, "index": index, "chunks": chunks},
        "llm": {
            "provider": llm.name,
            "requested": svc.settings.llm_provider,
            "fallback": bool(llm_note),
            "note": llm_note or None,
        },
        "hint": hint,
    }


@router.get("/search", response_model=SearchResponse, dependencies=[RateLimited])
def search(request: Request, params: Annotated[SearchParams, Query()], svc: Svc) -> SearchResponse:
    _begin(request, "search", params.method, params.q)
    result = _search(svc, params.q, params.method, params.k, params.to_search_filters())
    _finish(request, len(result.hits), result.timings_ms)
    return SearchResponse.from_result(result, params.k)


@router.get("/compare", response_model=CompareResponse, dependencies=[RateLimited])
def compare(
    request: Request, params: Annotated[CompareParams, Query()], svc: Svc
) -> CompareResponse:
    _begin(request, "compare", "compare", params.q)
    filters = params.to_search_filters()
    results = {m: _search(svc, params.q, m, params.k, filters) for m in COMPARE_METHODS}
    _finish(request, len(results["hybrid"].hits), results["hybrid"].timings_ms)
    return CompareResponse(
        query=params.q,
        k=params.k,
        results={m: SearchResponse.from_result(r, params.k) for m, r in results.items()},
    )


@router.post("/ask", response_model=AskResponse, dependencies=[RateLimited])
def ask(request: Request, body: AskRequest, svc: Svc) -> AskResponse:
    _begin(request, "ask", body.method, body.question)
    llm, llm_note = svc.llm()
    with reranker_guard(body.method):
        answer = answer_question(
            body.question,
            svc.search(),
            llm,
            method=body.method,
            k=body.k,
            filters=body.filters.to_search_filters(),
            settings=svc.settings,
        )
    if llm_note:
        answer.confidence_note = f"{llm_note} {answer.confidence_note}"
    _finish(
        request, len(answer.used_chunks), {"llm": answer.timings_ms.get("llm", 0.0)}, answer.found
    )
    return AskResponse.from_answer(body.question, answer)


@router.get("/documents", response_model=DocumentsResponse)
def documents(params: Annotated[DocumentsParams, Query()], svc: Svc) -> DocumentsResponse:
    rows = svc.search().list_documents(params.source, params.limit, params.offset)
    return DocumentsResponse(
        limit=params.limit, offset=params.offset, items=[DocumentOut(**r) for r in rows]
    )


@router.get("/stats")
def stats(svc: Svc) -> dict[str, Any]:
    return svc.search().stats()


@router.get("/stats/runtime")
def stats_runtime(request: Request) -> dict[str, Any]:
    snapshot: dict[str, Any] = request.app.state.runtime_stats.snapshot()
    return snapshot


# ---- UI pages (the data comes from the JSON API via fetch) -------------------------------------


def _page(request: Request, template: str, active: str) -> HTMLResponse:
    return request.app.state.templates.TemplateResponse(
        request, template, {"active": active, "examples": EXAMPLES}
    )


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
def page_search(request: Request) -> HTMLResponse:
    return _page(request, "search.html", "search")


@router.get("/frage", response_class=HTMLResponse, include_in_schema=False)
def page_ask(request: Request) -> HTMLResponse:
    return _page(request, "ask.html", "ask")


@router.get("/vergleich", response_class=HTMLResponse, include_in_schema=False)
def page_compare(request: Request) -> HTMLResponse:
    return _page(request, "compare.html", "compare")
