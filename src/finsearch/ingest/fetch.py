"""Polite, SSRF-safe downloader with an on-disk cache.

Rules enforced here (see CLAUDE.md): only https URLs that are listed in sources.yaml, honest
User-Agent, >= ingest_delay_s between requests to the same host, robots.txt respected,
retries with backoff, size cap, no private/loopback/link-local targets, bounded + re-validated
redirects, and files are only ever written to raw_dir/<doc_id>.<ext>.
"""

from __future__ import annotations

import hashlib
import ipaddress
import logging
import socket
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import httpx

from finsearch.config import Settings
from finsearch.ingest.sources import doc_id_for

log = logging.getLogger(__name__)

MAX_BYTES = 50 * 1024 * 1024
MAX_REDIRECTS = 5
MAX_ATTEMPTS = 3
BACKOFF_BASE_S = 2.0
TIMEOUT_S = 30.0
RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
REDIRECT_STATUS = frozenset({301, 302, 303, 307, 308})
PDF_MAGIC = b"%PDF-"
DISALLOW_ALL = ["User-agent: *", "Disallow: /"]

Resolver = Callable[[str], list[str]]


class FetchError(RuntimeError):
    """Download refused or failed (the pipeline logs it and skips the document)."""


@dataclass(frozen=True)
class Fetched:
    doc_id: str
    path: Path
    kind: str  # "pdf" | "html"
    content_hash: str  # sha256 of the raw bytes
    from_cache: bool

    def read(self) -> bytes:
        return self.path.read_bytes()


def resolve_host(host: str) -> list[str]:
    """All IP addresses of a host (system DNS)."""
    try:
        infos = socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise FetchError(f"DNS lookup failed for {host}: {exc}") from exc
    return sorted({str(info[4][0]) for info in infos})


def assert_public_https(url: str, resolver: Resolver = resolve_host) -> None:
    """SSRF guard: https only and every resolved address must be a public one."""
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise FetchError(f"only https URLs are allowed: {url}")
    for ip in resolver(parsed.hostname):
        addr = ipaddress.ip_address(ip)
        if not addr.is_global or addr.is_multicast:
            raise FetchError(f"refusing non-public address {ip} for host {parsed.hostname}")
    # shortcut: httpx resolves the name again when connecting (DNS rebinding window);
    # upgrade by connecting to the vetted IP if this ever fetches untrusted URLs.


def detect_kind(data: bytes, content_type: str) -> str:
    """ "pdf" or "html" - by magic bytes first, header second."""
    if data.startswith(PDF_MAGIC) or "pdf" in content_type.lower():
        return "pdf"
    return "html"


class Fetcher:
    def __init__(
        self,
        settings: Settings,
        allowed_urls: Iterable[str],
        client: httpx.Client | None = None,
        resolver: Resolver = resolve_host,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.raw_dir = settings.raw_dir
        self.delay_s = settings.ingest_delay_s
        self.allowed_urls = set(allowed_urls)
        self.allowed_hosts = {urlparse(u).hostname for u in self.allowed_urls}
        self.client = client or httpx.Client(
            timeout=TIMEOUT_S,
            follow_redirects=False,  # redirects are followed manually so each hop is validated
        )
        self.user_agent = settings.ingest_user_agent
        self.resolver = resolver
        self.sleep = sleep
        self.clock = clock
        self._last_request: dict[str, float] = {}
        self._robots: dict[str, RobotFileParser | None] = {}

    # ---- public API -----------------------------------------------------------------------

    def fetch(self, url: str) -> Fetched:
        if url not in self.allowed_urls:
            raise FetchError(f"URL is not listed in the sources file: {url}")
        doc_id = doc_id_for(url)
        cached = self._from_cache(doc_id)
        if cached:
            return cached
        if not self._robots_allows(url):
            raise FetchError(f"robots.txt disallows {url}")
        data, content_type = self._download(url)
        kind = detect_kind(data, content_type)
        path = self._store(doc_id, kind, data)
        return Fetched(doc_id, path, kind, hashlib.sha256(data).hexdigest(), from_cache=False)

    # ---- cache ------------------------------------------------------------------------------

    def _from_cache(self, doc_id: str) -> Fetched | None:
        for kind in ("pdf", "html"):
            path = self.raw_dir / f"{doc_id}.{kind}"
            if path.is_file():
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                return Fetched(doc_id, path, kind, digest, from_cache=True)
        return None

    def _store(self, doc_id: str, kind: str, data: bytes) -> Path:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        path = self.raw_dir / f"{doc_id}.{kind}"  # doc_id is 16 hex chars => no traversal
        tmp = path.with_suffix(".part")
        tmp.write_bytes(data)
        tmp.replace(path)
        return path

    # ---- network ----------------------------------------------------------------------------

    def _wait_turn(self, host: str) -> None:
        last = self._last_request.get(host)
        if last is not None:
            remaining = self.delay_s - (self.clock() - last)
            if remaining > 0:
                self.sleep(remaining)
        self._last_request[host] = self.clock()

    def _get_once(self, url: str) -> tuple[httpx.Response, bytes]:
        """One validated request (no redirect following), body capped at MAX_BYTES."""
        assert_public_https(url, self.resolver)
        self._wait_turn(urlparse(url).hostname or "")
        with self.client.stream("GET", url, headers={"User-Agent": self.user_agent}) as response:
            length = response.headers.get("content-length", "")
            declared = int(length) if length.isdigit() else 0
            if declared > MAX_BYTES:
                raise FetchError(f"content-length {declared} exceeds {MAX_BYTES} bytes: {url}")
            body = bytearray()
            for part in response.iter_bytes():
                body.extend(part)
                if len(body) > MAX_BYTES:
                    raise FetchError(f"download exceeds {MAX_BYTES} bytes: {url}")
        return response, bytes(body)

    def _get_with_retries(self, url: str) -> tuple[httpx.Response, bytes]:
        for attempt in range(MAX_ATTEMPTS):
            last = attempt == MAX_ATTEMPTS - 1
            try:
                response, body = self._get_once(url)
            except httpx.TransportError as exc:
                if last:
                    raise FetchError(f"network error for {url}: {exc}") from exc
            else:
                if response.status_code not in RETRY_STATUS or last:
                    return response, body
            self.sleep(BACKOFF_BASE_S * 2**attempt)
        raise FetchError(f"unreachable: {url}")  # pragma: no cover

    def _download(self, url: str) -> tuple[bytes, str]:
        current = url
        for _ in range(MAX_REDIRECTS + 1):
            response, body = self._get_with_retries(current)
            if response.status_code in REDIRECT_STATUS:
                current = urljoin(current, response.headers.get("location", ""))
                if urlparse(current).hostname not in self.allowed_hosts:
                    raise FetchError(f"redirect to unlisted host refused: {current}")
                continue
            if response.status_code != httpx.codes.OK:
                raise FetchError(f"HTTP {response.status_code} for {current}")
            return body, response.headers.get("content-type", "")
        raise FetchError(f"more than {MAX_REDIRECTS} redirects: {url}")

    # ---- robots.txt -------------------------------------------------------------------------

    def _robots_allows(self, url: str) -> bool:
        parsed = urlparse(url)
        origin = f"https://{parsed.netloc}"
        if origin not in self._robots:
            self._robots[origin] = self._load_robots(origin)
        parser = self._robots[origin]
        return True if parser is None else parser.can_fetch(self.user_agent, url)

    def _load_robots(self, origin: str) -> RobotFileParser | None:
        """None = no robots.txt (4xx other than 401/403) => everything allowed."""
        try:
            response, body = self._get_with_retries(f"{origin}/robots.txt")
        except FetchError as exc:
            log.warning("robots.txt unreachable for %s (%s); treating as disallow-all", origin, exc)
            return _parser(DISALLOW_ALL)
        status = response.status_code
        if status == httpx.codes.OK:
            return _parser(body.decode("utf-8", errors="replace").splitlines())
        if status not in (401, 403) and 400 <= status < 500:
            return None
        return _parser(DISALLOW_ALL)  # 401/403 and 3xx/5xx: cannot tell what the site wants


def _parser(lines: list[str]) -> RobotFileParser:
    parser = RobotFileParser()
    parser.parse(lines)
    return parser
