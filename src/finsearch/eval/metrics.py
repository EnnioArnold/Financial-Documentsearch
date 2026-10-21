"""Pure retrieval metrics (no Elasticsearch, no I/O).

Conventions
-----------
* ``ranked`` is the list of *relevance keys* of the returned hits in rank order (best first).
  The runner uses ``chunk_id`` (or ``doc_id`` in the doc-level fallback) as key.
* ``relevant`` is the set of keys that count as relevant for the query.
* Relevance is binary. A key that appears several times in ``ranked`` counts only at its
  first position; later duplicates still occupy a rank but are non-relevant.
"""

from __future__ import annotations

import math
import random
from collections.abc import Collection, Sequence

DEFAULT_BOOTSTRAP_RESAMPLES = 1000
DEFAULT_CONFIDENCE = 0.95
DEFAULT_SEED = 42


def _relevance_flags(ranked: Sequence[str], relevant: Collection[str], k: int) -> list[bool]:
    """Binary relevance of the top-k positions; repeated keys are only credited once."""
    seen: set[str] = set()
    flags: list[bool] = []
    for key in ranked[:k]:
        flags.append(key in relevant and key not in seen)
        seen.add(key)
    return flags


def first_relevant_rank(ranked: Sequence[str], relevant: Collection[str]) -> int | None:
    """1-based rank of the first relevant key, or None if there is none."""
    for i, key in enumerate(ranked, start=1):
        if key in relevant:
            return i
    return None


def recall_at_k(ranked: Sequence[str], relevant: Collection[str], k: int) -> float:
    """|distinct relevant keys in top-k| / |relevant|. 0.0 if ``relevant`` is empty."""
    if not relevant:
        return 0.0
    return sum(_relevance_flags(ranked, relevant, k)) / len(set(relevant))


def reciprocal_rank(
    ranked: Sequence[str], relevant: Collection[str], k: int | None = None
) -> float:
    """1 / rank of the first relevant key (0.0 if none within the top-k). Mean over queries = MRR."""
    rank = first_relevant_rank(ranked if k is None else ranked[:k], relevant)
    return 0.0 if rank is None else 1.0 / rank


def ndcg_at_k(ranked: Sequence[str], relevant: Collection[str], k: int) -> float:
    """Binary nDCG@k.

    DCG@k  = sum_{i=1..k} rel_i / log2(i + 1)
    IDCG@k = sum_{i=1..min(k, |relevant|)} 1 / log2(i + 1)   (all relevant keys ranked first)
    nDCG@k = DCG@k / IDCG@k, and 0.0 if there are no relevant keys.
    """
    n_rel = len(set(relevant))
    if n_rel == 0:
        return 0.0
    dcg = sum(
        1.0 / math.log2(i + 1)
        for i, hit in enumerate(_relevance_flags(ranked, relevant, k), 1)
        if hit
    )
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(k, n_rel) + 1))
    return dcg / idcg


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(values: Sequence[float], q: float) -> float:
    """q-th percentile (0..100) with linear interpolation between closest ranks (numpy default)."""
    if not values:
        return 0.0
    if not 0.0 <= q <= 100.0:
        raise ValueError(f"q must be in [0, 100], got {q}")
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def median(values: Sequence[float]) -> float:
    return percentile(values, 50.0)


def bootstrap_ci(
    values: Sequence[float],
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = DEFAULT_SEED,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of per-query scores (seeded, deterministic)."""
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)  # noqa: S311 - statistics, not security
    n = len(values)
    means = sorted(mean([values[rng.randrange(n)] for _ in range(n)]) for _ in range(n_resamples))
    alpha = (1.0 - confidence) / 2.0 * 100.0
    return (percentile(means, alpha), percentile(means, 100.0 - alpha))
