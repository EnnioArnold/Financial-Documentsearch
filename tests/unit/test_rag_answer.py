import pytest
from rag_fakes import CHUNKS_DE, CHUNKS_EN, FakeLLM, FakeSearcher, make_chunk, settings

from finsearch.contracts import NOT_FOUND_TEXT
from finsearch.rag.answer import answer_question, evidence_coverage, has_evidence
from finsearch.rag.providers.extractive import ExtractiveClient

Q_DE = "Wie hoch ist der Leitzins im Juni 2024?"
EXTRACTIVE = ExtractiveClient()


def ask(question, chunks, llm, **kw):
    cfg = kw.pop("cfg", settings())
    return answer_question(question, FakeSearcher(chunks), llm, settings=cfg, **kw)


def test_extractive_answer_de():
    a = ask(Q_DE, CHUNKS_DE, EXTRACTIVE)
    assert a.found and a.provider == "extractive"
    assert a.citations and a.citations[0].n == 1
    assert a.citations[0].chunk_id == CHUNKS_DE[0].chunk_id
    assert a.confidence_note == "Extraktive Antwort aus 1 Quellen"
    assert a.used_chunks == CHUNKS_DE
    assert set(a.timings_ms) == {"retrieve", "llm"}


def test_extractive_answer_en():
    a = ask("What did the Federal Reserve do with the federal funds rate?", CHUNKS_EN, EXTRACTIVE)
    assert a.found and "[1]" in a.answer


def test_llm_answer_is_validated_and_cited():
    llm = FakeLLM(
        "Die SNB senkte den Leitzins auf 1,25 Prozent [1][2]. Im Jahr [2024] war das neu [9].",
        name="groq",
    )
    a = ask(Q_DE, CHUNKS_DE, llm)
    assert a.found and a.provider == "groq"
    assert [c.n for c in a.citations] == [1, 2]
    assert a.dropped_citations == [2024, 9]
    assert "ungültige Quellenangabe" in a.confidence_note
    assert "[9]" not in a.answer
    assert '<source n="1"' in llm.calls[0][1].content


def test_unsupported_sentence_marked_in_note():
    llm = FakeLLM("Die SNB senkte den Leitzins deutlich [1]. Der Franken wird bald stärker werden.")
    a = ask(Q_DE, CHUNKS_DE, llm)
    assert a.answer.endswith("[unbelegt]") and "ohne Beleg" in a.confidence_note


def test_llm_error_falls_back_to_extractive():
    llm = FakeLLM(error="groq: HTTP 429 after 3 attempts", name="groq")
    a = ask(Q_DE, CHUNKS_DE, llm)
    assert a.found and a.provider == "extractive"
    assert "'groq' nicht verfügbar" in a.confidence_note
    assert "429" not in a.confidence_note
    assert a.citations


def test_llm_says_not_found():
    a = ask(Q_DE, CHUNKS_DE, FakeLLM("In den Quellen nicht gefunden. Siehe [1]."))
    assert not a.found and a.answer == NOT_FOUND_TEXT and a.citations == []


def test_llm_without_valid_citation_is_discarded():
    a = ask(Q_DE, CHUNKS_DE, FakeLLM("Der Leitzins beträgt 1,25 Prozent."))
    assert not a.found and a.answer == NOT_FOUND_TEXT and a.citations == []
    a = ask(Q_DE, CHUNKS_DE, FakeLLM("Der Leitzins beträgt 1,25 Prozent [42]."))
    assert not a.found and a.dropped_citations == [] and "verworfen" in a.confidence_note


def test_no_hits_is_not_found_and_llm_not_called():
    llm = FakeLLM("Erfunden [1].")
    a = ask(Q_DE, [], llm)
    assert not a.found and a.answer == NOT_FOUND_TEXT and llm.calls == []


def test_hallucination_gate_runs_before_llm():
    llm = FakeLLM("Die Marssonde-Kreditquote beträgt 42 Prozent [1].", name="chatty")
    a = ask("Wie hoch ist die Marssonde-Kreditquote?", CHUNKS_DE, llm)
    assert a.answer == NOT_FOUND_TEXT and not a.found and a.citations == []
    assert a.confidence_note == "Keine ausreichende Evidenz"
    assert llm.calls == []


def test_extractive_without_matching_sentence_is_not_found():
    # gate passes (terms occur) but no sentence is long enough / matching
    chunks = [make_chunk(1, "Leitzins Juni.")]
    a = ask("Leitzins Juni", chunks, EXTRACTIVE)
    assert not a.found and a.answer == NOT_FOUND_TEXT


def test_min_score_filters_hits():
    searcher = FakeSearcher(CHUNKS_DE, score=0.1)
    a = answer_question(Q_DE, searcher, EXTRACTIVE, settings=settings(rag_min_score=0.5))
    assert not a.found and a.used_chunks == []


def test_k_and_method_are_passed_to_searcher():
    searcher = FakeSearcher(CHUNKS_DE)
    answer_question(Q_DE, searcher, EXTRACTIVE, method="bm25", k=1, settings=settings())
    answer_question(Q_DE, searcher, EXTRACTIVE, settings=settings(rag_top_k=2))
    assert searcher.calls == [(Q_DE, "bm25", 1), (Q_DE, "hybrid", 2)]


def test_coverage_and_threshold():
    assert evidence_coverage(Q_DE, CHUNKS_DE) > 0.8
    assert evidence_coverage("Marssonde Kreditquote", CHUNKS_DE) == 0.0
    assert evidence_coverage("Wie ist das?", CHUNKS_DE) == 0.0  # only stopwords
    assert evidence_coverage(Q_DE, []) == 0.0
    assert not has_evidence("Marssonde Kreditquote", CHUNKS_DE, 0.4)
    assert has_evidence("Marssonde Kreditquote", CHUNKS_DE, 0.0)  # gate switched off
    assert not has_evidence(Q_DE, [], 0.0)
    assert has_evidence(Q_DE, CHUNKS_DE, 0.9) is (evidence_coverage(Q_DE, CHUNKS_DE) >= 0.9)


@pytest.mark.parametrize("q", ["Wie hoch ist der Leitzins?", "What is the federal funds rate?"])
def test_gate_default_threshold_from_settings(q):
    assert has_evidence(q, CHUNKS_DE + CHUNKS_EN)
