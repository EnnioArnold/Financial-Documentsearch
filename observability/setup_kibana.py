"""Set up Kibana for FinSearch: data views + the "FinSearch – Betrieb" dashboard.

Idempotent, stdlib only (runs with plain `python`, also as docker entrypoint).
Env: KIBANA_URL (default http://127.0.0.1:5601), KIBANA_WAIT_S (default 300).
Exit code 0 on success, 1 with a message on stderr otherwise.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

KIBANA_URL = os.environ.get("KIBANA_URL", "http://127.0.0.1:5601").rstrip("/")
WAIT_S = int(os.environ.get("KIBANA_WAIT_S", "300"))
POLL_S = 3
HTTP_TIMEOUT_S = 60
NDJSON = Path(__file__).with_name("kibana_dashboard.ndjson")

# Stable IDs: the dashboard NDJSON references these.
# allowNoIndex lets Kibana create the data views before any data exists.
DATA_VIEWS = [
    {"id": "finsearch-queries", "title": "finsearch-queries*", "name": "FinSearch Query-Log",
     "timeFieldName": "@timestamp", "allowNoIndex": True},
    {"id": "finsearch-content", "title": "finsearch-v*", "name": "FinSearch Inhalt (Chunks)",
     "allowNoIndex": True},
]  # fmt: skip


class SetupError(Exception):
    pass


def _request(
    method: str, path: str, body: bytes | None = None, headers: dict[str, str] | None = None
) -> tuple[int, dict]:
    req = urllib.request.Request(  # noqa: S310 - URL comes from our own env var
        KIBANA_URL + path, data=body, method=method, headers={"kbn-xsrf": "true", **(headers or {})}
    )
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_S) as resp:  # noqa: S310
            return resp.status, json.load(resp)
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, json.load(exc)
        except ValueError:
            return exc.code, {}


def wait_for_kibana() -> None:
    deadline = time.monotonic() + WAIT_S
    while True:
        try:
            status, body = _request("GET", "/api/status")
            if (
                status == 200
                and body.get("status", {}).get("overall", {}).get("level") == "available"
            ):
                return
        except (urllib.error.URLError, OSError, ValueError):
            pass
        if time.monotonic() > deadline:
            raise SetupError(f"Kibana at {KIBANA_URL} not available after {WAIT_S}s")
        time.sleep(POLL_S)


def ensure_data_view(view: dict) -> str:
    status, _ = _request("GET", f"/api/data_views/data_view/{view['id']}")
    if status == 200:
        return "exists"
    if status != 404:
        raise SetupError(f"checking data view {view['id']} failed: HTTP {status}")
    payload = json.dumps({"data_view": view}).encode()
    status, body = _request(
        "POST", "/api/data_views/data_view", payload, {"content-type": "application/json"}
    )
    if status != 200:
        raise SetupError(f"creating data view {view['id']} failed: HTTP {status} {body}")
    return "created"


def import_dashboard() -> int:
    if not NDJSON.exists():
        raise SetupError(f"missing {NDJSON}")
    boundary = uuid.uuid4().hex
    head = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
        f'filename="{NDJSON.name}"\r\nContent-Type: application/ndjson\r\n\r\n'
    ).encode()
    body = head + NDJSON.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    status, resp = _request(
        "POST",
        "/api/saved_objects/_import?overwrite=true",
        body,
        {"content-type": f"multipart/form-data; boundary={boundary}"},
    )
    if status != 200 or not resp.get("success"):
        raise SetupError(f"dashboard import failed: HTTP {status} {json.dumps(resp)[:1000]}")
    return int(resp.get("successCount", 0))


def main() -> int:
    try:
        print(f"waiting for Kibana at {KIBANA_URL} ...")
        wait_for_kibana()
        for view in DATA_VIEWS:
            print(f"data view {view['title']}: {ensure_data_view(view)}")
        print(f"imported {import_dashboard()} saved objects from {NDJSON.name}")
    except (SetupError, urllib.error.URLError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"done: {KIBANA_URL}/app/dashboards#/view/finsearch-betrieb")
    return 0


if __name__ == "__main__":
    sys.exit(main())
