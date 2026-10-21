"""Run the golden set against a Searcher and write the report.

python -m finsearch.eval.runner [--golden eval/golden_set.jsonl] [--methods bm25,knn,hybrid,hybrid_rerank]
                                [--k 10] [--out reports/eval_results.md] [--experiments] [--label NAME]
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from finsearch.config import ROOT, Settings, get_settings
from finsearch.contracts import METHODS, Searcher, SearchHit
from finsearch.eval.golden import GoldenQuestion, load_golden
from finsearch.eval.results import (
    EvalRun,
    ExperimentResult,
    MethodResult,
    QueryOutcome,
    score_query,
    summarize,
)

log = logging.getLogger(__name__)

DEFAULT_K = 10
DEFAULT_GOLDEN = ROOT / "eval" / "golden_set.jsonl"
DEFAULT_OUT = ROOT / "reports" / "eval_results.md"

SearcherFactory = Callable[[Settings], Searcher]


class MethodUnavailableError(RuntimeError):
    """A method cannot run (e.g. reranker model failed to load)."""


@dataclass(frozen=True)
class Experiment:
    """One row of the sweep table: run ``method`` with Settings fields overridden."""

    label: str
    method: str
    overrides: dict[str, object] = field(default_factory=dict)


EXPERIMENT_GRID: tuple[Experiment, ...] = (
    Experiment("rrf_k=20", "hybrid", {"rrf_k": 20}),
    Experiment("rrf_k=60 (default)", "hybrid", {"rrf_k": 60}),
    Experiment("rrf_k=100", "hybrid", {"rrf_k": 100}),
    Experiment("boost_text=0.0", "hybrid", {"boost_text": 0.0}),
    Experiment("boost_text=1.0", "hybrid", {"boost_text": 1.0}),
    Experiment("boost_de=1.5, boost_en=1.0", "hybrid", {"boost_de": 1.5, "boost_en": 1.0}),
    Experiment("boost_de=1.0, boost_en=1.5", "hybrid", {"boost_de": 1.0, "boost_en": 1.5}),
    Experiment("rerank off", "hybrid", {}),
    Experiment("rerank on", "hybrid_rerank", {}),
)


def chunk_config_label(settings: Settings) -> str:
    """Label for chunk-size experiments (lead re-ingests, then passes this as --label)."""
    return f"chunk{settings.chunk_target_tokens}_overlap{settings.chunk_overlap_ratio:g}"


def _keys(hits: list[SearchHit], question: GoldenQuestion) -> list[str]:
    return [h.chunk.doc_id if question.doc_level else h.chunk.chunk_id for h in hits]


def run_method(
    searcher: Searcher, method: str, questions: Sequence[GoldenQuestion], k: int
) -> MethodResult:
    """Evaluate one method on the answerable questions. Raises MethodUnavailableError."""
    if not questions:
        raise ValueError("no answerable questions to evaluate")
    try:  # warm-up (model load, caches); excluded from latency
        searcher.search(questions[0].question, method=method, k=k)
    except Exception as exc:  # noqa: BLE001 - any failure here means "method unavailable"
        raise MethodUnavailableError(f"{type(exc).__name__}: {exc}") from exc
    outcomes: list[QueryOutcome] = []
    for q in questions:
        res = searcher.search(q.question, method=method, k=k)
        first_rank, scores = score_query(_keys(res.hits, q), q.relevant_keys)
        outcomes.append(
            QueryOutcome(
                q.id, q.type, q.lang, first_rank, scores, res.total_ms, dict(res.timings_ms)
            )
        )
    return MethodResult(method, outcomes, summarize(outcomes))


def evaluate(
    searcher: Searcher,
    golden: Sequence[GoldenQuestion],
    methods: Sequence[str],
    k: int = DEFAULT_K,
    label: str = "",
    run_date: date | None = None,
) -> EvalRun:
    answerable = [q for q in golden if q.answerable]
    results: dict[str, MethodResult] = {}
    skipped: dict[str, str] = {}
    for method in [m for m in METHODS if m in methods]:  # deterministic order
        try:
            results[method] = run_method(searcher, method, answerable, k)
        except MethodUnavailableError as exc:
            log.warning("skipping method %s: %s", method, exc)
            skipped[method] = str(exc)
    stats_fn = getattr(searcher, "stats", None)  # optional: doc/chunk counts of the index
    stats: dict[str, object] = {}
    if callable(stats_fn):
        try:
            stats = dict(stats_fn())
        except Exception as exc:  # noqa: BLE001 - stats are decoration only
            log.warning("searcher.stats() failed: %s", exc)
    return EvalRun(
        run_date=run_date or date.today(),
        label=label,
        k=k,
        n_total=len(golden),
        n_answerable=len(answerable),
        n_unanswerable=len(golden) - len(answerable),
        provenance_counts=dict(sorted(Counter(q.provenance for q in golden).items())),
        results=results,
        skipped=skipped,
        questions={q.id: q for q in golden},
        index_stats=stats,
    )


def run_experiments(
    golden: Sequence[GoldenQuestion],
    searcher_factory: SearcherFactory,
    base_settings: Settings,
    k: int = DEFAULT_K,
    grid: Sequence[Experiment] = EXPERIMENT_GRID,
) -> list[ExperimentResult]:
    """Table-driven sweep: build a searcher per row from modified Settings and evaluate it."""
    answerable = [q for q in golden if q.answerable]
    out: list[ExperimentResult] = []
    for exp in grid:
        try:
            searcher = searcher_factory(base_settings.model_copy(update=exp.overrides))
            res = run_method(searcher, exp.method, answerable, k)
            out.append(ExperimentResult(exp.label, exp.method, exp.overrides, res.summary))
        except Exception as exc:  # noqa: BLE001 - a broken row must not kill the sweep
            log.warning("experiment %r failed: %s", exp.label, exc)
            out.append(ExperimentResult(exp.label, exp.method, exp.overrides, None, str(exc)))
    return out


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate FinSearch retrieval on the golden set.")
    p.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    p.add_argument("--methods", default=",".join(METHODS))
    p.add_argument("--k", type=int, default=DEFAULT_K)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--experiments", action="store_true")
    p.add_argument("--label", default="", help="free-text run label, e.g. chunk200_overlap0.12")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    from finsearch.eval.report import write_report
    from finsearch.logging import setup_logging
    from finsearch.search.service import get_search_service  # lazy: needs ES + models

    setup_logging()
    args = parse_args(argv)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = [m for m in methods if m not in METHODS]
    if unknown:
        raise SystemExit(f"unknown methods {unknown}; choose from {list(METHODS)}")
    golden = load_golden(args.golden)
    settings = get_settings()
    run = evaluate(get_search_service(settings), golden, methods, args.k, label=args.label)
    if args.experiments:
        run.experiments = run_experiments(golden, get_search_service, settings, args.k)
    write_report(run, args.out)
    log.info("report written to %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
