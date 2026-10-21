"""Seed SYNTHETIC query-log documents so the Kibana dashboard has data to show.

Every document carries `synthetic: true`. Fixed random seed + fixed _ids, so re-running is idempotent.

    python observability/seed_demo_queries.py            # write ~60 docs spread over the last hours
    python observability/seed_demo_queries.py --delete   # remove all synthetic docs again
"""

from __future__ import annotations

import argparse
import logging
import random
from datetime import UTC, datetime, timedelta

from elasticsearch import Elasticsearch, helpers

from finsearch.api.querylog import QueryLogger, build_doc
from finsearch.config import get_settings

log = logging.getLogger("seed_demo_queries")

N_DOCS = 60
SPAN_HOURS = 6
RNG_SEED = 42
ERROR_RATE = 0.08
ZERO_HIT_RATE = 0.08
SEARCH_METHODS = ["bm25", "knn", "hybrid", "hybrid", "hybrid_rerank"]
ASK_PROVIDERS = ["extractive", "extractive", "gemini", "groq"]
QUERIES = [
    ("SNB Leitzins Entscheid", "de"), ("Inflation Schweiz Prognose", "de"),
    ("Fed rate decision inflation outlook", "en"), ("Eigenkapitalquote UBS", "de"),
    ("Swiss franc intervention", "en"), ("Geldpolitische Lagebeurteilung", "de"),
    ("quantitative tightening balance sheet", "en"), ("Negativzinsen Banken", "de"),
    ("Zinsen", "de"), ("Hypothekarzinsen Entwicklung", "de"), ("labor market unemployment", "en"),
    ("Wechselkurs Franken Euro", "de"),
]  # fmt: skip
ZERO_HIT_QUERIES = [("bitcoin kurs prognose", "de"), ("zxqv", "other"), ("pizza rezept", "de")]
ERROR_TYPES = ["ConnectionError", "LLMError", "TimeoutError"]


def _timings(rng: random.Random, method: str, endpoint: str, provider: str) -> dict[str, float]:
    timings: dict[str, float] = {}
    if method != "bm25":
        timings["embed"] = rng.uniform(12, 45)
        timings["knn"] = rng.uniform(8, 40)
    if method != "knn":
        timings["bm25"] = rng.uniform(4, 30)
    if method in ("hybrid", "hybrid_rerank"):
        timings["fusion"] = rng.uniform(0.3, 2.0)
    if method == "hybrid_rerank":
        timings["rerank"] = rng.uniform(250, 900)
    if endpoint == "ask":
        timings["llm"] = rng.uniform(2, 15) if provider == "extractive" else rng.uniform(700, 2800)
    return timings


def make_docs(now: datetime) -> list[dict]:
    rng = random.Random(RNG_SEED)  # noqa: S311 - demo data, not security
    docs: list[dict] = []
    for i in range(N_DOCS):
        endpoint = "ask" if rng.random() < 0.3 else "search"
        provider = rng.choice(ASK_PROVIDERS)
        method = provider if endpoint == "ask" else rng.choice(SEARCH_METHODS)
        # /ask runs hybrid retrieval first; its stage timings come from there
        stages_method = "hybrid_rerank" if endpoint == "ask" else method
        zero = rng.random() < ZERO_HIT_RATE
        query, lang = rng.choice(ZERO_HIT_QUERIES if zero else QUERIES)
        failed = rng.random() < ERROR_RATE
        timings = _timings(rng, stages_method, endpoint, provider)
        obs = {
            "endpoint": endpoint, "method": method, "query": query, "lang": lang,
            "n_hits": 0 if zero else rng.randint(3, 10), "timings_ms": timings,
            "found": (not zero) if endpoint == "ask" else None,
        }  # fmt: skip
        doc = build_doc(
            path=f"/{endpoint}", http_method="POST" if endpoint == "ask" else "GET",
            status=500 if failed else 200, total_ms=sum(timings.values()) * rng.uniform(1.05, 1.3),
            error_type=rng.choice(ERROR_TYPES) if failed else None, obs=obs, log_raw_queries=True,
        )  # fmt: skip
        doc["@timestamp"] = (now - timedelta(minutes=rng.uniform(0, SPAN_HOURS * 60))).isoformat()
        doc["synthetic"] = True
        docs.append({"_index": "", "_id": f"synthetic-{i:03d}", "_source": doc})
    return docs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--delete", action="store_true", help="delete synthetic docs and exit")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    settings = get_settings()
    es = Elasticsearch(settings.es_url)
    index = settings.es_queries_index
    if args.delete:
        if es.indices.exists(index=index):
            resp = es.delete_by_query(
                index=index, query={"term": {"synthetic": True}}, refresh=True
            )
            log.info("deleted %s synthetic docs from %s", resp["deleted"], index)
        return
    QueryLogger(es, index).ensure_index()
    docs = make_docs(datetime.now(UTC))
    for d in docs:
        d["_index"] = index
    ok, _ = helpers.bulk(es, docs, refresh=True)
    log.info("indexed %s synthetic docs into %s", ok, index)


if __name__ == "__main__":
    main()
