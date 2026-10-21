"""Chunking: sentence splitting, token cap, overlap, page tracking, edge cases (no I/O)."""

from __future__ import annotations

from datetime import date

import pytest

from finsearch.contracts import DocumentMeta
from finsearch.ingest.chunk import (
    HARD_CAP_TOKENS,
    MIN_CHUNK_TOKENS,
    chunk_document,
    estimate_tokens,
    split_sentences,
)

DOC = DocumentMeta(
    doc_id="0123456789abcdef",
    source="test",
    title="Synthetic title",
    url="https://example.org/a.pdf",
    lang="en",
    doc_type="speech",
    date=date(2099, 1, 2),
)


def sentences(n: int, prefix: str = "S") -> str:
    """n distinct sentences of ~17 words each."""
    return " ".join(
        f"{prefix}{i} the quick brown fox number {i} jumps over the lazy dog again and again today."
        for i in range(n)
    )


def test_estimate_tokens_is_characters_over_four() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("a" * 400) == 100
    assert estimate_tokens("a" * 401) == 101  # rounds up (pessimistic)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("One. Two. Three.", ["One.", "Two.", "Three."]),
        ("Was ist das? Das ist gut! Ja.", ["Was ist das?", "Das ist gut!", "Ja."]),
        ("Am 5. Mai stieg der Zins. Dann sank er.", ["Am 5. Mai stieg der Zins.", "Dann sank er."]),
        (
            "Growth was 2.5% in 2025. The next year was weaker.",
            ["Growth was 2.5% in 2025.", "The next year was weaker."],
        ),
        (
            "Siehe z.B. Tabelle drei. Danach folgt Text.",
            ["Siehe z.B. Tabelle drei.", "Danach folgt Text."],
        ),
        # known limit: a one/two-digit number + "." is read as a German ordinal => no boundary
        ("See table 3. Then more text.", ["See table 3. Then more text."]),
        ("Dr. Meier sprach. Es war kurz.", ["Dr. Meier sprach.", "Es war kurz."]),
        ('He said "stop." Then he left.', ['He said "stop."', "Then he left."]),
        ("no capital after this. so it stays one", ["no capital after this. so it stays one"]),
        ("", []),
    ],
)
def test_split_sentences(text: str, expected: list[str]) -> None:
    assert split_sentences(text) == expected


def test_empty_text_gives_no_chunks() -> None:
    assert chunk_document(DOC, []) == []
    assert chunk_document(DOC, [(1, "")]) == []
    assert chunk_document(DOC, [(1, "   ")]) == []


def test_tiny_document_is_one_chunk_even_below_minimum() -> None:
    chunks = chunk_document(DOC, [(1, "Short text.")])
    assert len(chunks) == 1
    assert estimate_tokens(chunks[0].text) < MIN_CHUNK_TOKENS
    assert chunks[0].chunk_id == "0123456789abcdef-0000"
    assert chunks[0].ord == 0


def test_chunk_carries_document_metadata() -> None:
    chunk = chunk_document(DOC, [(3, "Hello world. Second sentence.")])[0]
    assert (chunk.doc_id, chunk.source, chunk.title, chunk.url) == (
        DOC.doc_id,
        "test",
        "Synthetic title",
        DOC.url,
    )
    assert (chunk.lang, chunk.doc_type, chunk.date, chunk.page) == ("en", "speech", DOC.date, 3)


def test_chunk_ids_are_sequential_and_unique() -> None:
    chunks = chunk_document(DOC, [(1, sentences(120))])
    assert len(chunks) > 3
    assert [c.chunk_id for c in chunks] == [f"{DOC.doc_id}-{i:04d}" for i in range(len(chunks))]
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_token_cap_is_never_exceeded() -> None:
    chunks = chunk_document(DOC, [(1, sentences(300))])
    assert all(estimate_tokens(c.text) <= HARD_CAP_TOKENS for c in chunks)
    # chunks are reasonably full (target 400): the median is near the target, not tiny
    sizes = sorted(estimate_tokens(c.text) for c in chunks)
    assert sizes[len(sizes) // 2] > 300


def test_single_very_long_sentence_is_hard_split() -> None:
    long_sentence = " ".join(f"word{i}" for i in range(3000)) + "."
    chunks = chunk_document(DOC, [(1, long_sentence)])
    assert len(chunks) > 5
    assert all(estimate_tokens(c.text) <= HARD_CAP_TOKENS for c in chunks)
    # nothing lost: every word appears (overlap may duplicate, never drop)
    words = " ".join(c.text for c in chunks).split()
    assert {f"word{i}" for i in range(2999)} <= set(words)


def test_single_word_longer_than_budget_is_cut() -> None:
    chunks = chunk_document(DOC, [(1, "x" * 10_000)])
    assert len(chunks) >= 6
    assert all(estimate_tokens(c.text) <= HARD_CAP_TOKENS for c in chunks)


def test_overlong_word_after_normal_words_is_cut_without_losing_the_words() -> None:
    chunks = chunk_document(DOC, [(1, "alpha beta gamma " + "y" * 5000 + " omega")])
    joined = " ".join(c.text for c in chunks)
    assert all(estimate_tokens(c.text) <= HARD_CAP_TOKENS for c in chunks)
    assert "alpha beta gamma" in joined
    assert "omega" in joined


def test_table_remnant_without_sentence_punctuation_is_capped() -> None:
    table = " ".join(f"{i * 7 % 1000}.{i % 10}" for i in range(2000))
    chunks = chunk_document(DOC, [(1, table)])
    assert all(estimate_tokens(c.text) <= HARD_CAP_TOKENS for c in chunks)


def _overlap_chars(prev_text: str, next_text: str) -> int:
    """Length of the leading sentences of next_text that were already in prev_text."""
    shared = 0
    for sentence in split_sentences(next_text):
        if sentence not in prev_text:
            break
        shared += len(sentence) + 1
    return shared


def test_overlap_repeats_last_sentences_of_previous_chunk() -> None:
    chunks = chunk_document(DOC, [(1, sentences(150))])
    assert len(chunks) > 3
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        assert prev.text.endswith(nxt.text[: _overlap_chars(prev.text, nxt.text) - 1])
        assert _overlap_chars(prev.text, nxt.text) > 0


def test_overlap_is_about_twelve_percent_and_zero_when_disabled() -> None:
    text = sentences(150)
    with_overlap = chunk_document(DOC, [(1, text)])
    without = chunk_document(DOC, [(1, text)], overlap_ratio=0.0)
    assert all(
        _overlap_chars(p.text, n.text) == 0 for p, n in zip(without, without[1:], strict=False)
    )
    target_chars = 400 * 4
    for prev, nxt in zip(with_overlap, with_overlap[1:], strict=False):
        assert 0.05 < _overlap_chars(prev.text, nxt.text) / target_chars < 0.13


def test_page_is_the_page_where_the_chunk_starts() -> None:
    pages = [(5, sentences(40, "A")), (6, sentences(40, "B")), (9, sentences(40, "C"))]
    chunks = chunk_document(DOC, pages)
    for chunk in chunks:
        first_marker = chunk.text[0]  # sentences are labelled A/B/C per page
        assert chunk.page == {"A": 5, "B": 6, "C": 9}[first_marker]
    assert chunks[0].page == 5
    assert chunks[-1].page == 9
    assert [c.page for c in chunks] == sorted(c.page for c in chunks)


def test_no_tiny_trailing_chunk() -> None:
    body = sentences(40)
    for extra in (0, 1, 2, 3):
        text = body + " Tail one." * extra
        chunks = chunk_document(DOC, [(1, text)])
        if len(chunks) > 1:
            assert all(estimate_tokens(c.text) >= MIN_CHUNK_TOKENS for c in chunks)


def test_deterministic() -> None:
    pages = [(1, sentences(100)), (2, sentences(100, "T"))]
    assert chunk_document(DOC, pages) == chunk_document(DOC, pages)


def test_custom_target_changes_chunk_size() -> None:
    small = chunk_document(DOC, [(1, sentences(100))], target_tokens=100)
    big = chunk_document(DOC, [(1, sentences(100))], target_tokens=400)
    assert len(small) > len(big)
    assert all(estimate_tokens(c.text) <= 100 + 12 + 2 for c in small)
