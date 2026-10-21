"""Serve the real app on a THROWAWAY index with the synthetic corpus (never touches `finsearch`).

Usage: python tests/e2e/serve_synthetic.py --port 8765
Fake embedder/reranker keep it fast and offline; the index and the query-log index are deleted on exit.
"""

from __future__ import annotations

import argparse
import signal
import sys
import uuid
from pathlib import Path

TESTS = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(TESTS), str(TESTS / "integration")]  # fakes.py, synthetic_corpus.py

import uvicorn  # noqa: E402
from fakes import FakeReranker, HashEmbedder  # noqa: E402
from synthetic_corpus import CORPUS  # noqa: E402

from finsearch.api.deps import Services  # noqa: E402
from finsearch.api.main import create_app  # noqa: E402
from finsearch.config import Settings  # noqa: E402
from finsearch.search.es_client import get_es  # noqa: E402
from finsearch.search.indexer import index_chunks  # noqa: E402
from finsearch.search.service import SearchService  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--name", default=f"finsearch-test-e2e-{uuid.uuid4().hex[:8]}")
    args = parser.parse_args()
    port, name = args.port, args.name
    # uvicorn re-raises SIGTERM after its shutdown; turn that into a normal exit so `finally` runs
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    settings = Settings(es_index_alias=name, es_queries_index=f"{name}-queries", api_rate_limit=0)
    es = get_es(settings)
    index_chunks(CORPUS, settings, HashEmbedder(), index_name=name, es=es)
    services = Services(
        settings,
        search_factory=lambda s: SearchService(es, HashEmbedder(), FakeReranker(), s),
    )
    try:
        uvicorn.run(
            create_app(settings, services), host="127.0.0.1", port=port, log_level="warning"
        )
    finally:
        for index in (name, settings.es_queries_index):
            es.indices.delete(index=index, ignore_unavailable=True)


if __name__ == "__main__":
    main()
