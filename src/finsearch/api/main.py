"""FastAPI app: `uvicorn finsearch.api.main:app`.

Startup is deliberately cheap: no model is loaded and Elasticsearch may be down. Services are
built on first use (see deps.Services); /health reports what is missing.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Any

from elasticsearch import ApiError as EsApiError
from elasticsearch import NotFoundError, TransportError
from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.datastructures import MutableHeaders
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.cors import CORSMiddleware
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from finsearch.api.deps import ApiError, RateLimiter, RuntimeStats, Services
from finsearch.api.middleware import QueryLogMiddleware
from finsearch.api.querylog import QueryLogger
from finsearch.api.routes import EMPTY_INDEX_HINT, router
from finsearch.config import ROOT, Settings, get_settings

log = logging.getLogger(__name__)

WEB_DIR = ROOT / "web"
MS_PER_S = 1000.0
HTTP_UNAVAILABLE = 503
# No inline script/style anywhere: everything comes from /static (JS sets styles via CSSOM only).
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    "Content-Security-Policy": (
        "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
        "connect-src 'self'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
    ),
}
STATUS_TITLES = {
    404: ("Seite nicht gefunden", "Diese Adresse gibt es nicht."),
    405: ("Methode nicht erlaubt", "Diese Adresse akzeptiert die Methode nicht."),
}


def parse_origins(raw: str) -> list[str]:
    """Comma-separated CORS origins; a wildcard is never accepted."""
    return [o.strip() for o in raw.split(",") if o.strip() and o.strip() != "*"]


class SecurityHeadersMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                for key, value in SECURITY_HEADERS.items():
                    headers.setdefault(key, value)  # error handlers may have set them already
            await send(message)

        await self.app(scope, receive, send_with_headers)


class RuntimeStatsMiddleware:
    """Feeds RuntimeStats; the endpoint/method labels come from `request.state.obs` if a route set it."""

    def __init__(self, app: ASGIApp, stats: RuntimeStats) -> None:
        self.app = app
        self.stats = stats

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        path = scope.get("path", "")
        if scope["type"] != "http" or path.startswith("/static/"):
            await self.app(scope, receive, send)
            return
        scope.setdefault("state", {})
        status = 500

        async def send_wrapper(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            await send(message)

        start = time.perf_counter()
        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            obs = scope["state"].get("obs") or {}
            self.stats.record(
                str(obs.get("endpoint") or path),
                str(obs.get("method") or scope["method"]),
                status,
                (time.perf_counter() - start) * MS_PER_S,
            )


def _error_response(
    request: Request,
    status: int,
    error: str,
    message: str,
    hint: str | None = None,
    headers: dict[str, str] | None = None,
    **extra: Any,
) -> Response:
    """HTML page for browsers (Accept: text/html), JSON for everything else (fetch, curl)."""
    if "text/html" in request.headers.get("accept", ""):
        context = {"status": status, "message": message, "hint": hint, "active": ""}
        response = request.app.state.templates.TemplateResponse(
            request, "error.html", context, status_code=status, headers=headers
        )
    else:
        body = {"error": error, "message": message, "hint": hint, **extra}
        response = JSONResponse(body, status_code=status, headers=headers)
    for key, value in SECURITY_HEADERS.items():  # also set when the middleware did not see it
        response.headers.setdefault(key, value)
    return response


def _describe_validation(errors: Sequence[Any]) -> str:
    """One German sentence for the UI; `detail` carries the raw pydantic errors for programs."""
    parts = []
    for err in errors:
        field = ".".join(str(x) for x in err["loc"][1:]) or str(err["loc"][0])
        empty = err["type"] == "string_too_short" and err.get("ctx", {}).get("min_length") == 1
        empty = empty or (err["type"] == "string_type" and err.get("input") is None)  # "?q="
        parts.append(f"{field}: {'darf nicht leer sein' if empty else err['msg']}")
    return "Ungültige Eingabe – " + "; ".join(parts)


def _register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def api_error(request: Request, exc: ApiError) -> Response:
        return _error_response(
            request, exc.status, exc.error, exc.message, exc.hint, headers=exc.headers
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException) -> Response:
        title, text = STATUS_TITLES.get(exc.status_code, ("Fehler", str(exc.detail)))
        return _error_response(request, exc.status_code, "http_error", f"{title}. {text}")

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> Response:
        raw = exc.errors()
        detail = [{k: e[k] for k in ("loc", "msg", "type")} for e in raw]  # ctx may not be JSON
        return _error_response(
            request, 422, "validation_error", _describe_validation(raw), detail=detail
        )

    @app.exception_handler(NotFoundError)
    async def index_missing(request: Request, exc: NotFoundError) -> Response:
        return _error_response(
            request, HTTP_UNAVAILABLE, "index_missing", "Der Suchindex existiert noch nicht.",
            EMPTY_INDEX_HINT,
        )  # fmt: skip

    @app.exception_handler(TransportError)
    @app.exception_handler(EsApiError)
    async def es_unavailable(request: Request, exc: Exception) -> Response:
        log.warning("Elasticsearch problem: %s", type(exc).__name__)
        return _error_response(
            request, HTTP_UNAVAILABLE, "search_backend_unavailable",
            "Elasticsearch ist nicht erreichbar.", "Läuft `make up`? Status: /health",
        )  # fmt: skip

    @app.exception_handler(Exception)
    async def unexpected(request: Request, exc: Exception) -> Response:
        log.error("unhandled %s on %s", type(exc).__name__, request.url.path, exc_info=exc)
        return _error_response(
            request, 500, "internal_error", "Interner Fehler. Bitte später erneut."
        )


def create_app(
    settings: Settings | None = None,
    services: Services | None = None,
    query_logger: QueryLogger | None = None,
) -> FastAPI:
    """Build the app. Arguments exist for tests (fake services / query logger)."""
    settings = settings or get_settings()
    ql = query_logger or QueryLogger.from_settings(settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        await run_in_threadpool(ql.ensure_index)
        yield
        await run_in_threadpool(ql.close)

    # /docs and /redoc load scripts from a CDN, which our CSP (and the no-CDN rule) forbids
    app = FastAPI(title="FinSearch", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.state.services = services or Services(settings)
    app.state.rate_limiter = RateLimiter(settings.api_rate_limit, settings.api_rate_window_s)
    app.state.runtime_stats = RuntimeStats()
    app.state.templates = Jinja2Templates(directory=WEB_DIR / "templates")  # autoescape is on

    # last added = outermost
    app.add_middleware(RuntimeStatsMiddleware, stats=app.state.runtime_stats)
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=parse_origins(settings.api_cors_origins),
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    app.add_middleware(QueryLogMiddleware, query_logger=ql)

    _register_error_handlers(app)
    app.include_router(router)
    app.mount("/static", StaticFiles(directory=WEB_DIR / "static"), name="static")
    return app


app = create_app()
