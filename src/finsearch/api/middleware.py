"""Pure-ASGI middleware: time every request, log it, and write one query-log document.

Route handlers add details via `request.state.obs = {...}` (see QueryLogger/build_doc for fields).
"""

from __future__ import annotations

import logging
import time

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from finsearch.api.querylog import QueryLogger

log = logging.getLogger(__name__)

SKIP_PATHS = frozenset({"/health", "/favicon.ico"})
SKIP_PREFIXES = ("/static/",)
MS_PER_S = 1000.0
HTTP_INTERNAL_ERROR = 500


class QueryLogMiddleware:
    def __init__(self, app: ASGIApp, query_logger: QueryLogger) -> None:
        self.app = app
        self.query_logger = query_logger

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        path = scope.get("path", "")
        if scope["type"] != "http" or path in SKIP_PATHS or path.startswith(SKIP_PREFIXES):
            await self.app(scope, receive, send)
            return

        scope.setdefault("state", {})  # same dict that backs request.state in the handler
        status = HTTP_INTERNAL_ERROR
        error_type: str | None = None

        async def send_wrapper(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            await send(message)

        start = time.perf_counter()
        try:
            await self.app(scope, receive, send_wrapper)
        except Exception as exc:
            error_type = type(exc).__name__
            log.exception("unhandled error in %s %s", scope["method"], path)
            raise  # Starlette's outermost handler turns this into the 500 response
        finally:
            try:
                self.query_logger.record(
                    path=path,
                    http_method=scope["method"],
                    status=status,
                    total_ms=(time.perf_counter() - start) * MS_PER_S,
                    error_type=error_type,
                    obs=scope["state"].get("obs"),
                )
            except Exception:  # logging must never break or mask the real outcome
                log.exception("query logging failed")
