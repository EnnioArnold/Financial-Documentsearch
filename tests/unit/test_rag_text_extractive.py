from rag_fakes import CHUNKS_DE, CHUNKS_EN, make_chunk

from finsearch.contracts import ChatMessage, LLMError
from finsearch.rag.providers.extractive import ExtractiveClient, extractive_answer
from finsearch.rag.text import content_terms, detect_lang, split_sentences, stem


def test_language_detection_and_terms():
    assert detect_lang("Wie hoch ist der Leitzins in der Schweiz?") == "de"
    assert detect_lang("What is the policy rate in the United States?") == "en"
    assert "wie" not in content_terms("Wie hoch ist der Leitzins?")
    assert content_terms("Zinsen") == content_terms("Zins") == ["zins"]
    assert content_terms("rates", "en") == content_terms("rate", "en")
    assert content_terms("Inflationsprognosen") == content_terms("Inflationsprognose")
    assert stem("ab", "de") == "ab"  # short words are never stemmed


def test_umlaut_folding():
    assert content_terms("Währung") == content_terms("Wahrung")


def test_split_sentences():
    parts = split_sentences("Erster Satz hier. Zweiter Satz [1]. [2] Dritter Satz.")
    assert parts == ["Erster Satz hier.", "Zweiter Satz [1]. [2]", "Dritter Satz."]
    assert split_sentences("Der Zins beträgt 1.25 Prozent.") == ["Der Zins beträgt 1.25 Prozent."]


def test_extractive_picks_matching_sentence_with_source_number():
    out = extractive_answer("Wie hoch ist der Leitzins im Juni 2024?", CHUNKS_DE)
    assert out.startswith("Die Schweizerische Nationalbank senkt den Leitzins")
    assert out.splitlines()[0].endswith("[1]")
    assert len(out.splitlines()) <= 3


def test_extractive_second_source():
    out = extractive_answer("Wie hoch ist die Inflationsprognose für 2025?", CHUNKS_DE)
    assert out.endswith("[2]") and "Inflationsprognose" in out


def test_extractive_is_verbatim_and_english():
    out = extractive_answer(
        "What did the Federal Reserve do with the federal funds rate?", CHUNKS_EN
    )
    assert "held the federal funds rate steady" in out and out.endswith("[1]")


def test_extractive_strips_foreign_footnote_markers():
    chunk = make_chunk(
        1, "Der Leitzins wurde im Juni deutlich gesenkt [7]. Weitere Details folgen später."
    )
    out = extractive_answer("Leitzins im Juni", [chunk])
    assert "[7]" not in out and out.endswith("[1]")


def test_extractive_nothing_matches():
    assert extractive_answer("Marssonde Kreditquote", CHUNKS_DE) == ""
    assert extractive_answer("", CHUNKS_DE) == ""
    assert extractive_answer("Leitzins", []) == ""


def test_extractive_client_does_not_generate():
    client = ExtractiveClient()
    assert client.name == "extractive"
    try:
        client.generate([ChatMessage("user", "x")])
    except LLMError:
        pass
    else:
        raise AssertionError("expected LLMError")
