"""Runner + report end to end with a deterministic FakeSearcher."""

import json
from pathlib import Path

import pytest

from finsearch.config import Settings
from finsearch.contracts import Chunk, Searcher, SearchFilters, SearchHit, SearchResult
from finsearch.eval import report, runner
from finsearch.eval.golden import load_golden
from finsearch.eval.results import EvalRun

SAMPLE = Path(__file__).parents[1] / "fixtures" / "eval_golden_sample.jsonl"
WARMUP_MS = 9999.0

# method -> question text fragment -> ranked chunk ids (doc id = chunk id minus the ordinal)
RANKINGS: dict[str, dict[str, list[str]]] = {
    "bm25": {
        "Leitzins": ["syn-x-0001", "syn-a-0001"],  # relevant at rank 2
        "inflation": ["syn-x-0001"],  # miss
        "Quartalsbericht": ["syn-c-0003"],  # rank 1
        "policy": ["syn-d-0001", "syn-d-0002"],  # doc-level: rank 1 (second chunk is a duplicate)
        "lowest fee": ["syn-x-0001"],  # miss
    },
    "knn": {
        "Leitzins": ["syn-x-0001"],  # miss
        "inflation": ["syn-b-0002"],  # rank 1
        "Quartalsbericht": ["syn-x-0001", "syn-y-0001", "syn-c-0003"],  # rank 3
        "policy": ["syn-z-0001", "syn-d-0002"],  # rank 2
        "lowest fee": ["syn-x-0001"],  # miss
    },
    "hybrid": {
        "Leitzins": ["syn-a-0001"],  # rank 1 (helps)
        "inflation": ["syn-x-0001", "syn-b-0002"],  # rank 2 (hurts vs knn)
        "Quartalsbericht": ["syn-c-0003"],
        "policy": ["syn-d-0001"],
        "lowest fee": ["syn-x-0001"],  # miss
    },
}


class FakeSearcher:
    def __init__(
        self, unavailable: tuple[str, ...] = ("hybrid_rerank",), stats: bool = False
    ) -> None:
        self.unavailable = unavailable
        self.calls: list[tuple[str, str]] = []
        self._seen: set[str] = set()
        if stats:
            self.stats = lambda: {"docs": 5, "chunks": 50}  # type: ignore[method-assign]

    def search(
        self, query: str, method: str = "hybrid", k: int = 10, filters: SearchFilters | None = None
    ) -> SearchResult:
        self.calls.append((method, query))
        if method in self.unavailable:
            raise RuntimeError("model could not be loaded")
        first_call = method not in self._seen
        self._seen.add(method)
        ids = next(v for frag, v in RANKINGS[method].items() if frag in query)
        hits = [
            SearchHit(_chunk(cid), 1.0 / r, r, method) for r, cid in enumerate(ids[:k], start=1)
        ]
        base = {"bm25": 10.0, "knn": 20.0, "hybrid": 30.0}[method]
        return SearchResult(query, method, hits, WARMUP_MS if first_call else base, {"bm25": 5.0})


def _chunk(chunk_id: str) -> Chunk:
    doc_id, ordinal = chunk_id.rsplit("-", 1)
    return Chunk(chunk_id, doc_id, int(ordinal), "t", 1, "syn", "T", "u", "de", "other")


@pytest.fixture
def run() -> EvalRun:
    return runner.evaluate(FakeSearcher(stats=True), load_golden(SAMPLE), list(runner.METHODS), 10)


def test_fake_is_a_searcher() -> None:
    assert isinstance(FakeSearcher(), Searcher)


def test_metrics_hand_calculated(run: EvalRun) -> None:
    # hybrid ranks: 1, 2, 1, 1, miss -> Recall@5 = 4/5, MRR = (1 + 1/2 + 1 + 1 + 0) / 5 = 0.7
    hyb = run.results["hybrid"].summary
    assert hyb.scores["recall@5"] == pytest.approx(0.8)
    assert hyb.scores["recall@1"] == pytest.approx(0.6)
    assert hyb.scores["mrr"] == pytest.approx(0.7)
    # bm25 ranks: 2, miss, 1, 1, miss -> MRR = (0.5 + 1 + 1) / 5 = 0.5
    assert run.results["bm25"].summary.scores["mrr"] == pytest.approx(0.5)
    # knn ranks: miss, 1, 3, 2, miss -> MRR = (1 + 1/3 + 1/2) / 5
    assert run.results["knn"].summary.scores["mrr"] == pytest.approx((1 + 1 / 3 + 0.5) / 5)


def test_unanswerable_skipped_and_warmup_excluded(run: EvalRun) -> None:
    assert (run.n_total, run.n_answerable, run.n_unanswerable) == (6, 5, 1)
    assert all(o.qid != "syn006" for r in run.results.values() for o in r.outcomes)
    s = run.results["bm25"].summary
    assert (s.n, s.latency_median_ms, s.latency_p95_ms) == (5, 10.0, 10.0)  # no 9999 warm-up
    assert s.stage_means_ms == {"bm25": 5.0}
    assert run.index_stats == {"docs": 5, "chunks": 50}
    assert run.provenance_counts == {"human": 5, "llm_suggested_verified": 1}


def test_unavailable_method_is_skipped_not_fatal(run: EvalRun) -> None:
    assert list(run.results) == ["bm25", "knn", "hybrid"]
    assert "model could not be loaded" in run.skipped["hybrid_rerank"]
    assert "Übersprungen: Hybrid+Rerank" in report.render_markdown(run)


def test_method_order_is_canonical() -> None:
    r = runner.evaluate(FakeSearcher(), load_golden(SAMPLE), ["hybrid", "bm25"], 10)
    assert list(r.results) == ["bm25", "hybrid"]


def test_no_answerable_questions_raises() -> None:
    only_unanswerable = [q for q in load_golden(SAMPLE) if not q.answerable]
    with pytest.raises(ValueError, match="no answerable"):
        runner.evaluate(FakeSearcher(), only_unanswerable, ["bm25"], 10)


def test_stats_failure_is_tolerated() -> None:
    s = FakeSearcher()
    s.stats = lambda: 1 / 0  # type: ignore[attr-defined]
    assert runner.evaluate(s, load_golden(SAMPLE), ["bm25"], 10).index_stats == {}


def test_verdict_and_help_hurt(run: EvalRun) -> None:
    answer, why = report.verdict(run)
    assert answer == "yes" and "0.800" in why
    md = report.render_markdown(run)
    assert "Hybrid >= best single method on Recall@5 and MRR: yes" in md
    helps, hurts = md.split("Hybrid besser als die beste Einzelmethode")[1].split("schlechter als")
    assert "syn001" in helps and "syn002" not in helps
    assert "syn002" in hurts and "syn001" not in hurts


def test_verdict_no_and_not_available(run: EvalRun) -> None:
    run.results["hybrid"] = run.results["knn"]  # MRR 0.367 < bm25 0.5 -> "no"
    assert report.verdict(run)[0] == "no"
    del run.results["hybrid"]
    assert report.verdict(run)[0] == "n/a"
    assert "Nicht auswertbar" in report.render_markdown(run)


def test_failure_analysis_lists_misses(run: EvalRun) -> None:
    md = report.render_markdown(run)
    section = md.split("### Fragen ohne relevanten Treffer")[1].split("### Wo hilft")[0]
    assert "**Hybrid**: 1 von 5" in section and "`syn005`" in section
    assert "**BM25**: 2 von 5" in section


def test_write_report_end_to_end(run: EvalRun, tmp_path: Path) -> None:
    out = tmp_path / "reports" / "eval_results.md"
    report.write_report(run, out)
    md = out.read_text(encoding="utf-8")
    assert "![eval_quality](figures/eval_quality.png)" in md
    assert "Ehrliche Einschränkungen" in md and "llm_suggested_verified: 1" in md
    assert (tmp_path / "reports" / "figures" / "eval_quality.png").stat().st_size > 0
    assert (tmp_path / "reports" / "figures" / "eval_latency.png").stat().st_size > 0
    data = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert data["verdict_hybrid_ge_best_single"] == "yes"
    assert data["methods"]["hybrid"]["summary"]["scores"]["mrr"] == pytest.approx(0.7)
    assert data["skipped"].keys() == {"hybrid_rerank"}


def test_report_without_any_result(tmp_path: Path) -> None:
    r = runner.evaluate(FakeSearcher(unavailable=("bm25",)), load_golden(SAMPLE), ["bm25"], 10)
    out = tmp_path / "r.md"
    report.write_report(r, out)
    assert "Keine Methode" in out.read_text(encoding="utf-8")
    assert not (tmp_path / "figures").exists()


def test_experiments_table_driven() -> None:
    seen: list[Settings] = []

    def factory(settings: Settings) -> Searcher:
        seen.append(settings)
        if settings.rrf_k == 100:
            raise RuntimeError("boom")
        return FakeSearcher()

    grid = (
        runner.Experiment("k20", "hybrid", {"rrf_k": 20}),
        runner.Experiment("k100", "hybrid", {"rrf_k": 100}),
        runner.Experiment("rerank", "hybrid_rerank"),
    )
    base = Settings()
    res = runner.run_experiments(load_golden(SAMPLE), factory, base, 10, grid)
    assert [r.label for r in res] == ["k20", "k100", "rerank"]
    assert res[0].summary is not None and res[0].summary.scores["mrr"] == pytest.approx(0.7)
    assert res[1].summary is None and "boom" in res[1].note
    assert res[2].summary is None  # unavailable method inside a sweep
    assert [s.rrf_k for s in seen] == [20, 100, base.rrf_k]
    assert base.rrf_k == 60  # base settings untouched

    r = runner.evaluate(FakeSearcher(), load_golden(SAMPLE), ["hybrid"], 10)
    r.experiments = res
    md = report.render_markdown(r)
    assert "## Experimente" in md and "nicht ausführbar: boom" in md
    assert "experiments" in report.to_json(r)


def test_default_grid_covers_required_sweeps() -> None:
    labels = " ".join(e.label for e in runner.EXPERIMENT_GRID)
    for needle in ("rrf_k=20", "rrf_k=100", "boost_text", "boost_de", "rerank off", "rerank on"):
        assert needle in labels


def test_chunk_config_label() -> None:
    assert runner.chunk_config_label(Settings(chunk_target_tokens=200)) == "chunk200_overlap0.12"


def test_main_wires_everything(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import types

    fake_mod = types.ModuleType("finsearch.search.service")
    fake_mod.get_search_service = lambda settings: FakeSearcher()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "finsearch.search.service", fake_mod)
    out = tmp_path / "out" / "eval.md"
    rc = runner.main(
        ["--golden", str(SAMPLE), "--methods", "bm25,hybrid", "--out", str(out), "--label", "t"]
    )
    assert rc == 0 and "Lauf: t" in out.read_text(encoding="utf-8")
    with pytest.raises(SystemExit, match="unknown methods"):
        runner.main(["--golden", str(SAMPLE), "--methods", "nope", "--out", str(out)])
    rc = runner.main(
        ["--golden", str(SAMPLE), "--methods", "hybrid", "--out", str(out), "--experiments"]
    )
    assert "## Experimente" in out.read_text(encoding="utf-8")
