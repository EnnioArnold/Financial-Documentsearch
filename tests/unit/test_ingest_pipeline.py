"""sources.yaml loading, JSONL io and the end-to-end pipeline on SYNTHETIC fixtures (no network)."""

from __future__ import annotations

import json
import sys
import types
from datetime import date
from pathlib import Path

import httpx
import pytest

from finsearch.config import ROOT, Settings
from finsearch.contracts import Chunk, DocumentMeta
from finsearch.ingest import pipeline
from finsearch.ingest.io import load_chunks, load_documents, write_jsonl
from finsearch.ingest.sources import DOC_TYPES, doc_id_for, load_sources

FIXTURES = Path(__file__).parents[1] / "fixtures"
PDF_URL = "https://synthetic.example.org/report_de.pdf"
HTML_URL = "https://synthetic.example.org/speech_en.html"
DUP_URL = "https://synthetic.example.org/report_copy.pdf"
MISSING_URL = "https://synthetic.example.org/missing.pdf"
PUBLIC_IP = "93.184.216.34"

SOURCES_YAML = f"""
sources:
  bank:
    name: Synthetic bank
    robots_checked: "synthetic"
    license_note: "synthetic test data"
    documents:
      - url: {PDF_URL}
        title: Synthetic report
        doc_type: annual_report
        lang: de
        date: 2099-03-01
      - url: {MISSING_URL}
        title: Missing
        doc_type: other
        lang: de
  central:
    name: Synthetic central bank
    robots_checked: "synthetic"
    license_note: "synthetic test data"
    documents:
      - url: {HTML_URL}
        title: Synthetic speech
        doc_type: speech
        lang: en
      - url: {DUP_URL}
        title: Same bytes as the report
        doc_type: annual_report
        lang: de
        max_pages: 3
"""


def make_settings(tmp_path: Path) -> Settings:
    sources = tmp_path / "sources.yaml"
    sources.write_text(SOURCES_YAML, encoding="utf-8")
    return Settings(
        sources_file=sources,
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
        ingest_delay_s=0.0,
    )


def fake_web(request: httpx.Request) -> httpx.Response:
    url = str(request.url)
    pdf = (FIXTURES / "synthetic_report_de.pdf").read_bytes()
    if url.endswith("/robots.txt"):
        return httpx.Response(404)
    if url in (PDF_URL, DUP_URL):
        return httpx.Response(200, content=pdf, headers={"content-type": "application/pdf"})
    if url == HTML_URL:
        html = (FIXTURES / "synthetic_speech_en.html").read_bytes()
        return httpx.Response(200, content=html, headers={"content-type": "text/html"})
    return httpx.Response(404)


def run_pipeline(settings: Settings, **kwargs: object) -> pipeline.RunResult:
    client = httpx.Client(transport=httpx.MockTransport(fake_web))
    return pipeline.run(settings, client=client, resolver=lambda host: [PUBLIC_IP], **kwargs)  # type: ignore[arg-type]


# ---- sources.yaml ----------------------------------------------------------------------------


def test_doc_id_is_first_16_hex_chars_of_sha256_of_the_url() -> None:
    doc_id = doc_id_for("https://example.org/a.pdf")
    assert len(doc_id) == 16
    int(doc_id, 16)
    assert doc_id == doc_id_for("https://example.org/a.pdf")
    assert doc_id != doc_id_for("https://example.org/b.pdf")


def test_load_sources_reads_fields_and_family_notes(tmp_path: Path) -> None:
    docs = load_sources(make_settings(tmp_path).sources_file)
    assert [d.source for d in docs] == ["bank", "bank", "central", "central"]
    first = docs[0]
    assert (first.title, first.doc_type, first.lang_hint) == (
        "Synthetic report",
        "annual_report",
        "de",
    )
    assert first.date == date(2099, 3, 1)
    assert first.license_note == "synthetic test data"
    assert first.robots_checked == "synthetic"
    assert docs[1].date is None
    assert docs[3].max_pages == 3


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        ("url: http://x.org/a.pdf\n        title: t\n        doc_type: speech", "https"),
        ("url: https://x.org/a.pdf\n        title: t\n        doc_type: poem", "doc_type"),
    ],
)
def test_load_sources_rejects_bad_entries(tmp_path: Path, entry: str, message: str) -> None:
    path = tmp_path / "s.yaml"
    path.write_text(f"sources:\n  a:\n    documents:\n      - {entry}\n", encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_sources(path)


def test_load_sources_rejects_duplicate_urls(tmp_path: Path) -> None:
    entry = "- url: https://x.org/a.pdf\n        title: t\n        doc_type: speech\n      "
    path = tmp_path / "s.yaml"
    path.write_text(f"sources:\n  a:\n    documents:\n      {entry}{entry}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_sources(path)


def test_shipped_sources_yaml_meets_the_corpus_requirements() -> None:
    docs = load_sources(ROOT / "sources.yaml")
    assert len(docs) >= 30
    assert len({d.source for d in docs}) >= 3
    assert len({d.url for d in docs}) == len(docs)
    assert all(d.doc_type in DOC_TYPES and d.url.startswith("https://") for d in docs)
    assert all(d.license_note and d.robots_checked for d in docs)


# ---- JSONL io --------------------------------------------------------------------------------


def make_doc(day: date | None) -> DocumentMeta:
    return DocumentMeta(
        "0123456789abcdef", "s", "Tïtle ü", "https://x.org/a", "de", "speech", day, 2, "h"
    )


def make_chunk(day: date | None) -> Chunk:
    return Chunk(
        "0123456789abcdef-0000",
        "0123456789abcdef",
        0,
        "Text ü",
        1,
        "s",
        "T",
        "u",
        "de",
        "speech",
        day,
    )


def test_chunks_roundtrip_through_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "out" / "chunks.jsonl"
    chunks = [make_chunk(date(2099, 1, 2)), make_chunk(None)]
    assert write_jsonl(path, chunks) == 2
    assert list(load_chunks(path)) == chunks
    first = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert first["date"] == "2099-01-02"
    assert "Text ü" in path.read_text(encoding="utf-8")  # not \u-escaped


def test_documents_roundtrip_through_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "documents.jsonl"
    docs = [make_doc(date(2099, 5, 6)), make_doc(None)]
    write_jsonl(path, docs)
    assert list(load_documents(path)) == docs
    assert json.loads(path.read_text(encoding="utf-8").splitlines()[1])["date"] is None


def test_write_jsonl_overwrites_and_leaves_no_temp_file(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl"
    write_jsonl(path, [make_chunk(None)] * 3)
    write_jsonl(path, [make_chunk(None)])
    assert len(path.read_text(encoding="utf-8").splitlines()) == 1
    assert [p.name for p in tmp_path.iterdir()] == ["c.jsonl"]


# ---- end-to-end ------------------------------------------------------------------------------


def test_pipeline_processes_documents_and_skips_failures_and_duplicates(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    result = run_pipeline(settings)
    assert [d.url for d in result.documents] == [PDF_URL, HTML_URL]
    assert {url for url, _ in result.failures} == {MISSING_URL, DUP_URL}
    reasons = dict(result.failures)
    assert "HTTP 404" in reasons[MISSING_URL]
    assert "duplicate content" in reasons[DUP_URL]

    report, speech = result.documents
    assert (report.lang, report.n_pages, report.source, report.date) == (
        "de",
        4,
        "bank",
        date(2099, 3, 1),
    )
    assert speech.lang == "en"
    assert len(report.content_hash) == 64
    ids = [c.chunk_id for c in result.chunks]
    assert len(ids) == len(set(ids))
    assert {c.doc_id for c in result.chunks} == {report.doc_id, speech.doc_id}
    assert all(c.lang == "de" for c in result.chunks if c.doc_id == report.doc_id)


def test_pipeline_cleans_the_pdf_text(tmp_path: Path) -> None:
    result = run_pipeline(make_settings(tmp_path))
    text = " ".join(c.text for c in result.chunks if c.source == "bank")
    assert "Zinssatz stieg" in text  # hyphenation repaired
    assert "Zins- und Kreditmarkt" in text  # compound hyphen kept
    assert "SYNTHETISCH" not in text  # running header removed
    assert "Seite 2" not in text  # page numbers removed
    assert "1 234 567" not in text  # table remnant removed
    assert "2.5%" in text  # numbers inside sentences kept


def test_pipeline_output_files_are_valid_and_idempotent(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    run_pipeline(settings)
    chunks_file = settings.processed_dir / "chunks.jsonl"
    docs_file = settings.processed_dir / "documents.jsonl"
    first = (chunks_file.read_bytes(), docs_file.read_bytes())

    run_pipeline(settings)  # second run: served from data/raw cache, output fully rewritten
    assert (chunks_file.read_bytes(), docs_file.read_bytes()) == first

    # a cold cache gives the same bytes too (same input => byte-identical output)
    for cached in settings.raw_dir.iterdir():
        cached.unlink()
    run_pipeline(settings)
    assert (chunks_file.read_bytes(), docs_file.read_bytes()) == first

    chunks = list(load_chunks(chunks_file))
    assert len({c.chunk_id for c in chunks}) == len(chunks) > 0
    assert len(list(load_documents(docs_file))) == 2


def test_cache_is_only_written_inside_raw_dir(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    run_pipeline(settings)
    names = sorted(p.name for p in settings.raw_dir.iterdir())
    expected = [
        f"{doc_id_for(PDF_URL)}.pdf",
        f"{doc_id_for(HTML_URL)}.html",
        f"{doc_id_for(DUP_URL)}.pdf",
    ]
    assert names == sorted(expected)  # the missing URL (404) left nothing behind


def test_source_filter_and_limit(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    only_central = run_pipeline(settings, source="central")
    assert {d.source for d in only_central.documents} == {"central"}
    limited = run_pipeline(settings, limit=1)
    assert [d.url for d in limited.documents] == [PDF_URL]


def test_documents_with_too_little_text_are_skipped(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)

    def tiny(request: httpx.Request) -> httpx.Response:
        if str(request.url).endswith("robots.txt"):
            return httpx.Response(404)
        return httpx.Response(200, content=b"<html><body><p>Hi.</p></body></html>")

    client = httpx.Client(transport=httpx.MockTransport(tiny))
    result = pipeline.run(
        settings, client=client, resolver=lambda host: [PUBLIC_IP], source="central"
    )
    assert result.documents == []
    assert all(
        "characters of text" in reason or "duplicate" in reason for _, reason in result.failures
    )


def test_wrong_language_hint_is_logged_but_detected_language_wins(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    settings = make_settings(tmp_path)
    settings.sources_file.write_text(SOURCES_YAML.replace("lang: en", "lang: de"), encoding="utf-8")
    with caplog.at_level("WARNING"):
        result = run_pipeline(settings, source="central", limit=1)
    assert result.documents[0].lang == "en"
    assert "language hint de != detected en" in caplog.text


def test_stats_table_mentions_every_figure(tmp_path: Path) -> None:
    stats = pipeline.format_stats(run_pipeline(make_settings(tmp_path)))
    assert "documents: 2" in stats
    assert "bank=1" in stats
    assert "central=1" in stats
    assert "de=1" in stats
    assert "median" in stats
    assert "p95" in stats
    assert f"FAILED {MISSING_URL}" in stats


def test_stats_of_an_empty_run() -> None:
    assert "documents: 0" in pipeline.format_stats(pipeline.RunResult())


# ---- command line ----------------------------------------------------------------------------


def seed_cache(settings: Settings) -> None:
    """Pre-populate data/raw so main() needs no network at all."""
    settings.raw_dir.mkdir(parents=True)
    pdf = (FIXTURES / "synthetic_report_de.pdf").read_bytes()
    html = (FIXTURES / "synthetic_speech_en.html").read_bytes()
    (settings.raw_dir / f"{doc_id_for(PDF_URL)}.pdf").write_bytes(pdf)
    (settings.raw_dir / f"{doc_id_for(HTML_URL)}.html").write_bytes(html)


def test_main_no_index_writes_files_and_prints_stats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    settings = make_settings(tmp_path)
    seed_cache(settings)
    monkeypatch.setattr(pipeline, "get_settings", lambda: settings)
    assert pipeline.main(["--no-index", "--source", "central"]) == 0
    assert "documents: 1" in capsys.readouterr().out
    assert (settings.processed_dir / "chunks.jsonl").exists()


def test_main_calls_the_indexer_with_the_chunks_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = make_settings(tmp_path)
    seed_cache(settings)
    monkeypatch.setattr(pipeline, "get_settings", lambda: settings)
    calls: list[Path] = []
    fake = types.ModuleType("finsearch.search.indexer")
    fake.index_chunks_jsonl = lambda path: calls.append(path) or {"indexed": 1}  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "finsearch.search.indexer", fake)
    pipeline.main(["--source", "central"])
    assert calls == [settings.processed_dir / "chunks.jsonl"]


def test_main_explains_a_missing_indexer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = make_settings(tmp_path)
    seed_cache(settings)
    monkeypatch.setattr(pipeline, "get_settings", lambda: settings)
    monkeypatch.setitem(sys.modules, "finsearch.search.indexer", None)  # import raises ImportError
    with pytest.raises(SystemExit, match="--no-index"):
        pipeline.main(["--source", "central"])
