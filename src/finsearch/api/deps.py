"""Shared API plumbing: lazy services, per-IP rate limiter, in-process runtime counters, errors."""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from fastapi import Request

from finsearch.config import Settings
from finsearch.contracts import LLMClient, LLMError
from finsearch.rag.llm_base import get_llm_client
from finsearch.rag.providers.extractive import ExtractiveClient
from finsearch.search.service import SearchService, get_search_service

log = logging.getLogger(__name__)

HTTP_TOO_MANY = 429
LATENCY_WINDOW = 1000  # last N durations kept per (endpoint, method) for the percentiles
MAX_TRACKED_CLIENTS = 10_000  # bound the rate-limiter memory
MS_PER_S = 1000.0


class ApiError(Exception):
    """An expected failure with a stable JSON shape: {"error", "message", "hint"}."""

    def __init__(
        self,
        status: int,
        error: str,
        message: str,
        hint: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status, self.error, self.message, self.hint = status, error, message, hint
        self.headers = headers or {}


# ---- lazy services -----------------------------------------------------------------------------


class Services:
    """Builds the SearchService and the LLM client on first use, so the app starts instantly."""

    def __init__(
        self,
        settings: Settings,
        search_factory: Callable[[Settings], SearchService] = get_search_service,
        llm_factory: Callable[[Settings], LLMClient] = get_llm_client,
    ) -> None:
        self.settings = settings
        self._search_factory = search_factory
        self._llm_factory = llm_factory
        self._search: SearchService | None = None
        self._llm: tuple[LLMClient, str] | None = None
        self._lock = threading.Lock()

    def search(self) -> SearchService:
        with self._lock:
            if self._search is None:
                self._search = self._search_factory(self.settings)
            return self._search

    def llm(self) -> tuple[LLMClient, str]:
        """(client, note). The note is non-empty when we fell back to the extractive provider."""
        with self._lock:
            if self._llm is None:
                self._llm = self._build_llm()
            return self._llm

    def _build_llm(self) -> tuple[LLMClient, str]:
        wanted = self.settings.llm_provider
        try:
            return self._llm_factory(self.settings), ""
        except (LLMError, ValueError) as exc:  # message is key-free by construction
            log.warning("LLM provider %r unusable, falling back to extractive: %s", wanted, exc)
            return ExtractiveClient(), (
                f"LLM-Anbieter '{wanted}' nicht verfügbar ({exc}); extraktive Antwort ohne LLM."
            )


# ---- rate limit --------------------------------------------------------------------------------


class RateLimiter:
    """Fixed window per client key. In memory, so per process (fine for a single-node demo)."""

    def __init__(
        self, limit: int, window_s: float, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.limit, self.window_s, self._clock = limit, window_s, clock
        self._windows: dict[str, tuple[float, int]] = {}  # key -> (window start, count)
        self._lock = threading.Lock()

    def check(self, key: str) -> int | None:
        """Count one request; return None if allowed, else the seconds until the window resets."""
        if self.limit <= 0:
            return None
        now = self._clock()
        with self._lock:
            start, count = self._windows.get(key, (now, 0))
            if now - start >= self.window_s:
                start, count = now, 0
            if count >= self.limit:
                return max(1, math.ceil(start + self.window_s - now))
            if len(self._windows) >= MAX_TRACKED_CLIENTS:
                self._windows = {
                    k: v for k, v in self._windows.items() if now - v[0] < self.window_s
                }
            self._windows[key] = (start, count + 1)
            return None


def client_key(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def enforce_rate_limit(request: Request) -> None:
    """FastAPI dependency for /search, /ask, /compare."""
    limiter: RateLimiter = request.app.state.rate_limiter
    retry_after = limiter.check(client_key(request))
    if retry_after is not None:
        raise ApiError(
            HTTP_TOO_MANY,
            "rate_limited",
            f"Zu viele Anfragen. Bitte in {retry_after} s erneut versuchen.",
            headers={"Retry-After": str(retry_after)},
        )


# ---- runtime counters --------------------------------------------------------------------------


@dataclass
class _Counter:
    count: int = 0
    errors: int = 0
    durations_ms: deque[float] = field(default_factory=lambda: deque(maxlen=LATENCY_WINDOW))


def percentile(sorted_values: list[float], p: float) -> float:
    """Nearest-rank percentile of an ascending list (0.0 for an empty list)."""
    if not sorted_values:
        return 0.0
    rank = max(1, math.ceil(p / 100 * len(sorted_values)))
    return sorted_values[rank - 1]


class RuntimeStats:
    """Thread-safe request counters since process start, keyed by (endpoint, method)."""

    def __init__(self) -> None:
        self.started = datetime.now(UTC)
        self._counters: defaultdict[tuple[str, str], _Counter] = defaultdict(_Counter)
        self._lock = threading.Lock()

    def record(self, endpoint: str, method: str, status: int, duration_ms: float) -> None:
        with self._lock:
            c = self._counters[(endpoint, method)]
            c.count += 1
            c.errors += status >= 500
            c.durations_ms.append(duration_ms)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            rows = [
                {
                    "endpoint": ep,
                    "method": m,
                    "count": c.count,
                    "errors": c.errors,
                    "p50_ms": round(percentile(sorted(c.durations_ms), 50), 2),
                    "p95_ms": round(percentile(sorted(c.durations_ms), 95), 2),
                }
                for (ep, m), c in sorted(self._counters.items())
            ]
        return {"since": self.started.isoformat(), "endpoints": rows}
