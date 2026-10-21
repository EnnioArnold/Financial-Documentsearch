"""Reciprocal Rank Fusion (ADR-001). Pure functions, no Elasticsearch."""

from __future__ import annotations

import math


def rrf_fuse(rankings: dict[str, list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Fuse several ranked id lists: score(d) = sum over lists of 1 / (k + rank), rank 1-based.

    Only ranks matter, so BM25 and cosine scores never need normalising. A list may contain an id
    twice; only its first (best) position counts. Order: higher score, then better best-rank in any
    list, then id (so results are fully deterministic).
    """
    contributions: dict[str, list[float]] = {}
    best_rank: dict[str, int] = {}
    for ids in rankings.values():
        seen: set[str] = set()
        rank = 0
        for doc_id in ids:
            if doc_id in seen:
                continue
            seen.add(doc_id)
            rank += 1
            contributions.setdefault(doc_id, []).append(1.0 / (k + rank))
            best_rank[doc_id] = min(best_rank.get(doc_id, rank), rank)
    # fsum is exact, so equal multisets of terms give bit-identical scores (ties stay ties)
    scored = [(doc_id, math.fsum(terms)) for doc_id, terms in contributions.items()]
    scored.sort(key=lambda t: (-t[1], best_rank[t[0]], t[0]))
    return scored
