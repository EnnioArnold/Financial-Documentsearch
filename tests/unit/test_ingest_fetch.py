"""Fetcher: SSRF guard, redirects, retries, size cap, rate limit, robots.txt, cache.

No network: httpx.MockTransport plays the web server, resolver/sleep/clock are fakes.
"""

from __future__ import annotations

import hashlib
import socket
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from finsearch.config import Settings
from finsearch.ingest import fetch as fetch_module
from finsearch.ingest.fetch import (
    BACKOFF_BASE_S,
    MAX_ATTEMPTS,
    MAX_REDIRECTS,
    Fetcher,
    FetchError,
    assert_public_https,
    detect_kind,
    resolve_host,
)
from finsearch.ingest.sources import doc_id_for

URL = "https://docs.example.org/report.pdf"
OTHER = "https://docs.example.org/other.pdf"
ELSEWHERE = "https://elsewhere.example.net/x.pdf"
PDF = b"%PDF-1.7 synthetic test bytes"
PUBLIC_IP = "93.184.216.34"


class FakeWeb:
    """Mock server + recording of every request."""

    def __init__(self, routes: dict[str, Callable[[httpx.Request], httpx.Response]]) -> None:
        self.routes = routes
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        handler = self.routes.get(str(request.url))
        if handler is None:
            return httpx.Response(404)
        return handler(request)

    def paths(self) -> list[str]:
        return [r.url.path for r in self.requests]


def ok_pdf(_: httpx.Request) -> httpx.Response:
    return httpx.Response(200, content=PDF, headers={"content-type": "application/pdf"})


def make_fetcher(
    tmp_path: Path,
    web: FakeWeb,
    urls: list[str] | None = None,
    resolver: Callable[[str], list[str]] = lambda host: [PUBLIC_IP],
    delay: float = 1.0,
    clock: Callable[[], float] | None = None,
    sleeps: list[float] | None = None,
) -> Fetcher:
    settings = Settings(
        raw_dir=tmp_path / "raw", ingest_delay_s=delay, ingest_user_agent="Test-UA/1.0"
    )
    client = httpx.Client(transport=httpx.MockTransport(web))
    recorded = sleeps if sleeps is not None else []
    return Fetcher(
        settings,
        urls or [URL, OTHER, ELSEWHERE],
        client=client,
        resolver=resolver,
        sleep=recorded.append,
        clock=clock or (lambda: 0.0),
    )


def web_with(**routes: Callable[[httpx.Request], httpx.Response]) -> FakeWeb:
    mapping = {u: routes.get(k, ok_pdf) for k, u in {"url": URL, "other": OTHER}.items()}
    mapping["https://docs.example.org/robots.txt"] = routes.get(
        "robots", lambda _: httpx.Response(404)
    )
    return FakeWeb(mapping)


# ---- happy path, cache, hash ----------------------------------------------------------------


def test_download_is_cached_hashed_and_sends_honest_user_agent(tmp_path: Path) -> None:
    web = web_with()
    fetcher = make_fetcher(tmp_path, web)
    result = fetcher.fetch(URL)
    assert result.kind == "pdf"
    assert not result.from_cache
    assert result.content_hash == hashlib.sha256(PDF).hexdigest()
    assert result.path == tmp_path / "raw" / f"{doc_id_for(URL)}.pdf"
    assert result.read() == PDF
    assert {r.headers["user-agent"] for r in web.requests} == {"Test-UA/1.0"}


def test_second_fetch_comes_from_cache_without_network(tmp_path: Path) -> None:
    web = web_with()
    fetcher = make_fetcher(tmp_path, web)
    first = fetcher.fetch(URL)
    n_requests = len(web.requests)
    second = fetcher.fetch(URL)
    assert second.from_cache
    assert second.content_hash == first.content_hash
    assert len(web.requests) == n_requests


def test_cache_survives_a_new_fetcher_instance(tmp_path: Path) -> None:
    make_fetcher(tmp_path, web_with()).fetch(URL)
    web = web_with()
    assert make_fetcher(tmp_path, web).fetch(URL).from_cache
    assert web.requests == []


def test_html_is_stored_with_html_extension(tmp_path: Path) -> None:
    def html(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"<html>x</html>", headers={"content-type": "text/html"})

    result = make_fetcher(tmp_path, web_with(url=html)).fetch(URL)
    assert result.kind == "html"
    assert result.path.suffix == ".html"


@pytest.mark.parametrize(
    ("data", "content_type", "kind"),
    [
        (b"%PDF-1.4", "application/octet-stream", "pdf"),
        (b"junk", "application/pdf", "pdf"),
        (b"<html>", "text/html", "html"),
    ],
)
def test_detect_kind(data: bytes, content_type: str, kind: str) -> None:
    assert detect_kind(data, content_type) == kind


# ---- allow-list and SSRF ---------------------------------------------------------------------


def test_url_not_in_sources_is_refused(tmp_path: Path) -> None:
    web = web_with()
    with pytest.raises(FetchError, match="not listed"):
        make_fetcher(tmp_path, web).fetch("https://docs.example.org/secret.pdf")
    assert web.requests == []


def test_http_url_is_refused() -> None:
    with pytest.raises(FetchError, match="https"):
        assert_public_https("http://docs.example.org/a.pdf", lambda h: [PUBLIC_IP])


@pytest.mark.parametrize(
    "ip",
    [
        "127.0.0.1",
        "10.0.0.5",
        "172.16.3.4",
        "192.168.1.1",
        "169.254.169.254",
        "0.0.0.0",
        "::1",
        "fe80::1",
        "fc00::1",
        "100.64.0.1",
    ],  # noqa: S104
)
def test_private_loopback_and_link_local_addresses_are_refused(ip: str) -> None:
    with pytest.raises(FetchError, match="non-public"):
        assert_public_https("https://docs.example.org/a.pdf", lambda h: [ip])


def test_one_private_address_among_public_ones_is_enough_to_refuse() -> None:
    with pytest.raises(FetchError, match="non-public"):
        assert_public_https("https://docs.example.org/a.pdf", lambda h: [PUBLIC_IP, "10.0.0.1"])


def test_public_address_is_accepted() -> None:
    assert_public_https(
        "https://docs.example.org/a.pdf", lambda h: [PUBLIC_IP, "2606:2800:220:1::1"]
    )


def test_fetch_never_sends_a_request_when_dns_points_to_private_ip(tmp_path: Path) -> None:
    web = web_with()
    fetcher = make_fetcher(tmp_path, web, resolver=lambda host: ["127.0.0.1"])
    with pytest.raises(FetchError, match="robots"):  # robots.txt lookup is refused first
        fetcher.fetch(URL)
    assert web.requests == []
    assert not (tmp_path / "raw").exists()


def test_resolve_host_returns_sorted_unique_ips(monkeypatch: pytest.MonkeyPatch) -> None:
    infos = [
        (0, 0, 0, "", ("1.2.3.4", 443)),
        (0, 0, 0, "", ("1.2.3.4", 443)),
        (0, 0, 0, "", ("0.0.0.1", 443)),
    ]
    monkeypatch.setattr(fetch_module.socket, "getaddrinfo", lambda *a, **k: infos)
    assert resolve_host("example.org") == ["0.0.0.1", "1.2.3.4"]


def test_resolve_host_failure_becomes_fetch_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*a: object, **k: object) -> None:
        raise socket.gaierror("nope")

    monkeypatch.setattr(fetch_module.socket, "getaddrinfo", boom)
    with pytest.raises(FetchError, match="DNS"):
        resolve_host("example.invalid")


# ---- redirects -------------------------------------------------------------------------------


def redirect(target: str) -> Callable[[httpx.Request], httpx.Response]:
    return lambda _: httpx.Response(302, headers={"location": target})


def test_redirect_to_listed_host_is_followed(tmp_path: Path) -> None:
    web = web_with(url=redirect(ELSEWHERE))
    web.routes[ELSEWHERE] = ok_pdf
    web.routes["https://elsewhere.example.net/robots.txt"] = lambda _: httpx.Response(404)
    assert make_fetcher(tmp_path, web).fetch(URL).read() == PDF


def test_relative_redirect_is_followed(tmp_path: Path) -> None:
    web = web_with(url=redirect("/other.pdf"))
    assert make_fetcher(tmp_path, web).fetch(URL).read() == PDF


def test_redirect_to_unlisted_host_is_refused(tmp_path: Path) -> None:
    web = web_with(url=redirect("https://evil.example.com/x.pdf"))
    with pytest.raises(FetchError, match="unlisted host"):
        make_fetcher(tmp_path, web).fetch(URL)


def test_redirect_to_http_is_refused(tmp_path: Path) -> None:
    web = web_with(url=redirect("http://docs.example.org/other.pdf"))
    with pytest.raises(FetchError, match="https"):
        make_fetcher(tmp_path, web).fetch(URL)


def test_redirect_to_host_resolving_to_private_ip_is_refused(tmp_path: Path) -> None:
    web = web_with(url=redirect(ELSEWHERE))

    def resolver(host: str) -> list[str]:
        return ["10.1.1.1"] if host.startswith("elsewhere") else [PUBLIC_IP]

    with pytest.raises(FetchError, match="non-public"):
        make_fetcher(tmp_path, web, resolver=resolver).fetch(URL)


def test_redirect_loop_is_cut_off(tmp_path: Path) -> None:
    web = web_with(url=redirect(URL))
    with pytest.raises(FetchError, match="redirects"):
        make_fetcher(tmp_path, web).fetch(URL)
    assert len([r for r in web.requests if r.url.path == "/report.pdf"]) == MAX_REDIRECTS + 1


# ---- retries, errors, size cap ---------------------------------------------------------------


def test_transient_errors_are_retried_with_exponential_backoff(tmp_path: Path) -> None:
    calls = {"n": 0}

    def flaky(_: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(503) if calls["n"] < MAX_ATTEMPTS else ok_pdf(_)

    sleeps: list[float] = []
    result = make_fetcher(tmp_path, web_with(url=flaky), delay=0.0, sleeps=sleeps).fetch(URL)
    assert result.read() == PDF
    assert calls["n"] == MAX_ATTEMPTS
    assert sleeps == [BACKOFF_BASE_S, BACKOFF_BASE_S * 2]


def test_connection_errors_are_retried_then_reported(tmp_path: Path) -> None:
    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom", request=request)

    sleeps: list[float] = []
    web = web_with(url=down)
    with pytest.raises(FetchError, match="network error"):
        make_fetcher(tmp_path, web, delay=0.0, sleeps=sleeps).fetch(URL)
    assert len([r for r in web.requests if r.url.path == "/report.pdf"]) == MAX_ATTEMPTS


def test_persistent_server_error_gives_fetch_error(tmp_path: Path) -> None:
    with pytest.raises(FetchError, match="HTTP 500"):
        make_fetcher(tmp_path, web_with(url=lambda _: httpx.Response(500))).fetch(URL)


def test_client_errors_are_not_retried(tmp_path: Path) -> None:
    web = web_with(url=lambda _: httpx.Response(404))
    with pytest.raises(FetchError, match="HTTP 404"):
        make_fetcher(tmp_path, web).fetch(URL)
    assert len([r for r in web.requests if r.url.path == "/report.pdf"]) == 1


def test_declared_content_length_over_cap_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(fetch_module, "MAX_BYTES", 10)

    def big(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"x" * 100, headers={"content-length": "100"})

    with pytest.raises(FetchError, match="content-length"):
        make_fetcher(tmp_path, web_with(url=big)).fetch(URL)


def test_streamed_body_over_cap_is_refused_and_not_stored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(fetch_module, "MAX_BYTES", 10)

    def chunked(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=iter([b"x" * 8, b"y" * 8]))  # no content-length

    with pytest.raises(FetchError, match="exceeds"):
        make_fetcher(tmp_path, web_with(url=chunked)).fetch(URL)
    assert not (tmp_path / "raw").exists()


# ---- politeness ------------------------------------------------------------------------------


def test_requests_to_the_same_host_are_spaced_by_the_delay(tmp_path: Path) -> None:
    now = {"t": 100.0}
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now["t"] += seconds

    fetcher = make_fetcher(tmp_path, web_with(), delay=1.0, clock=lambda: now["t"])
    fetcher.sleep = sleep
    fetcher.fetch(URL)  # robots.txt + pdf: the second request must wait
    fetcher.fetch(OTHER)
    assert len(sleeps) == 2
    assert all(s == pytest.approx(1.0) for s in sleeps)


def test_different_hosts_do_not_wait_for_each_other(tmp_path: Path) -> None:
    web = web_with()
    web.routes[ELSEWHERE] = ok_pdf
    web.routes["https://elsewhere.example.net/robots.txt"] = lambda _: httpx.Response(404)
    sleeps: list[float] = []
    fetcher = make_fetcher(tmp_path, web, delay=1.0, clock=lambda: 50.0, sleeps=sleeps)
    fetcher.fetch(ELSEWHERE)
    assert sleeps == [
        1.0
    ]  # only elsewhere's own robots.txt -> pdf gap, nothing for docs.example.org


# ---- robots.txt ------------------------------------------------------------------------------


def robots(text: str, status: int = 200) -> Callable[[httpx.Request], httpx.Response]:
    return lambda _: httpx.Response(status, text=text)


def test_robots_disallow_blocks_the_download(tmp_path: Path) -> None:
    web = web_with(robots=robots("User-agent: *\nDisallow: /report"))
    with pytest.raises(FetchError, match="robots"):
        make_fetcher(tmp_path, web).fetch(URL)
    assert "/report.pdf" not in web.paths()


def test_robots_allows_other_paths_and_is_fetched_once(tmp_path: Path) -> None:
    web = web_with(robots=robots("User-agent: *\nDisallow: /private/"))
    fetcher = make_fetcher(tmp_path, web)
    fetcher.fetch(URL)
    fetcher.fetch(OTHER)
    assert web.paths().count("/robots.txt") == 1


def test_missing_robots_means_allowed(tmp_path: Path) -> None:
    assert make_fetcher(tmp_path, web_with(robots=robots("", 404))).fetch(URL).read() == PDF


@pytest.mark.parametrize("status", [401, 403, 500, 301])
def test_forbidden_or_broken_robots_means_stay_away(tmp_path: Path, status: int) -> None:
    web = web_with(robots=robots("", status))
    with pytest.raises(FetchError, match="robots"):
        make_fetcher(tmp_path, web, delay=0.0).fetch(URL)


def test_unreachable_robots_means_stay_away(tmp_path: Path) -> None:
    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom", request=request)

    with pytest.raises(FetchError, match="robots"):
        make_fetcher(tmp_path, web_with(robots=down), delay=0.0).fetch(URL)
