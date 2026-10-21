"""Result containers and aggregation (pure, no I/O)."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date

from finsearch.eval import metrics
from finsearch.eval.golden import GoldenQuestion

RECALL_KS = (1, 3, 5, 10)
NDCG_K = 10
CI_METRICS = ("recall@5", "mrr", "ndcg@10")


@dataclass(frozen=True)
class QueryOutcome:
    """Per-question result of one method."""

    qid: str
    qtype: str
    lang: str
    first_rank: int | None  # 1-based rank of the first relevant hit, None = not retrieved
    scores: dict[str, float]  # recall@1/3/5/10, mrr, ndcg@10
    total_ms: float
    timings_ms: dict[str, float]


@dataclass
class MethodSummary:
    n: int
    scores: dict[str, float]  # mean per metric
    ci: dict[str, tuple[float, float]]  # 95 % bootstrap CI for CI_METRICS
    latency_median_ms: float
    latency_p95_ms: float
    stage_means_ms: dict[str, float]


@dataclass
class MethodResult:
    method: str
    outcomes: list[QueryOutcome]
    summary: MethodSummary


@dataclass
class EvalRun:
    run_date: date
    label: str
    k: int
    n_total: int
    n_answerable: int
    n_unanswerable: int
    provenance_counts: dict[str, int]
    results: dict[str, MethodResult]  # only methods that actually ran, in METHODS order
    skipped: dict[str, str]  # method -> reason
    questions: dict[str, GoldenQuestion] = field(default_factory=dict)
    index_stats: dict[str, object] = field(default_factory=dict)
    experiments: list[ExperimentResult] = field(default_factory=list)


@dataclass
class ExperimentResult:
    label: str
    method: str
    overrides: dict[str, object]
    summary: MethodSummary | None  # None = could not run
    note: str = ""


def score_query(ranked: list[str], relevant: frozenset[str]) -> tuple[int | None, dict[str, float]]:
    scores = {f"recall@{k}": metrics.recall_at_k(ranked, relevant, k) for k in RECALL_KS}
    scores["mrr"] = metrics.reciprocal_rank(ranked, relevant)
    scores[f"ndcg@{NDCG_K}"] = metrics.ndcg_at_k(ranked, relevant, NDCG_K)
    return metrics.first_relevant_rank(ranked, relevant), scores


def summarize(outcomes: list[QueryOutcome]) -> MethodSummary:
    names = list(outcomes[0].scores) if outcomes else []
    per_metric = {m: [o.scores[m] for o in outcomes] for m in names}
    latencies = [o.total_ms for o in outcomes]
    stages: dict[str, list[float]] = defaultdict(list)
    for o in outcomes:
        for stage, ms in o.timings_ms.items():
            stages[stage].append(ms)
    return MethodSummary(
        n=len(outcomes),
        scores={m: metrics.mean(v) for m, v in per_metric.items()},
        ci={m: metrics.bootstrap_ci(per_metric[m]) for m in CI_METRICS if m in per_metric},
        latency_median_ms=metrics.median(latencies),
        latency_p95_ms=metrics.percentile(latencies, 95.0),
        stage_means_ms={s: metrics.mean(v) for s, v in sorted(stages.items())},
    )


def breakdown(outcomes: list[QueryOutcome], attr: str) -> dict[str, tuple[int, float, float]]:
    """Group by 'qtype' or 'lang' -> (n, mean Recall@5, MRR), keys sorted."""
    groups: dict[str, list[QueryOutcome]] = defaultdict(list)
    for o in outcomes:
        groups[getattr(o, attr)].append(o)
    return {
        key: (
            len(g),
            metrics.mean([o.scores["recall@5"] for o in g]),
            metrics.mean([o.scores["mrr"] for o in g]),
        )
        for key, g in sorted(groups.items())
    }
