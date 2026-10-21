import pytest
from rag_fakes import CHUNKS_DE

from finsearch.contracts import NOT_FOUND_TEXT
from finsearch.rag.citations import UNSUPPORTED_TAG, build_citations, validate_citations


def check(text, n=3, **kw):
    return validate_citations(text, n, **kw)


def test_single_and_adjacent_markers():
    r = check("Der Zins sinkt [1][2]. Die Inflation ist tief [3].")
    assert r.valid == [1, 2, 3] and r.dropped == [] and r.found
    assert r.text == "Der Zins sinkt [1][2]. Die Inflation ist tief [3]."


@pytest.mark.parametrize(
    ("text", "valid"),
    [
        ("Der Zins sank deutlich im Juni [1, 3].", [1, 3]),
        ("Der Zins sank deutlich im Juni [1,3].", [1, 3]),
        ("Der Zins sank deutlich im Juni [1-3].", [1, 2, 3]),
        ("Der Zins sank deutlich im Juni [1–2, 3].", [1, 2, 3]),
        ("Der Zins sank deutlich im Juni [[2]].", [2]),
        ("Der Zins sank deutlich im Juni [2][2].", [2]),
    ],
)
def test_marker_formats(text, valid):
    r = check(text)
    assert r.valid == valid
    assert r.dropped == []
    assert "[unbelegt]" not in r.text


def test_normalises_to_individual_markers():
    assert check("Der Zins sank deutlich im Juni [1, 3].").text == (
        "Der Zins sank deutlich im Juni [1][3]."
    )


def test_invalid_numbers_are_dropped_and_reported():
    r = check("Der Zins sinkt laut Bericht [0] und [99] sowie [1].")
    assert r.valid == [1]
    assert r.dropped == [0, 99]
    assert "[0]" not in r.text and "[99]" not in r.text
    assert r.text == "Der Zins sinkt laut Bericht und sowie [1]."


def test_years_are_not_citations():
    r = check("Im Jahr [2024] sank der Zins deutlich [1].")
    assert r.valid == [1] and r.dropped == [2024]
    assert "2024" not in r.text


def test_year_range_is_rejected_as_a_whole():
    r = check("Zwischen den Jahren [2010-2020] sank der Zins [2].")
    assert r.valid == [2] and r.dropped == [2010, 2020]


def test_out_of_range_only_answer_becomes_not_found():
    r = check("Der Zins sinkt deutlich im Jahr [2024] und [7].")
    assert r.text == NOT_FOUND_TEXT and not r.found
    assert r.valid == [] and r.dropped == [2024, 7]


def test_literal_chunk_ids_are_not_citations():
    r = check("Der Zins sinkt deutlich laut [0000000000000001-0000].")
    assert not r.found and r.valid == []
    r = check("Der Zins sinkt deutlich [0000000000000001-0000] [1].")
    assert r.valid == [1] and "0000000000000001" not in r.text


def test_no_markers_and_empty_answers_are_not_found():
    assert check("Der Zins sinkt deutlich im Juni.").text == NOT_FOUND_TEXT
    assert check("").text == NOT_FOUND_TEXT
    assert check("[1]", n=0).text == NOT_FOUND_TEXT


def test_unsupported_sentence_is_tagged():
    r = check("Der Zins sinkt deutlich [1]. Der Franken wird stärker werden.")
    assert r.valid == [1] and r.unsupported == 1
    assert r.text.endswith(f"werden. {UNSUPPORTED_TAG}")


def test_unsupported_sentence_can_be_dropped():
    r = check(
        "Der Zins sinkt deutlich [1]. Der Franken wird stärker werden.", drop_unsupported=True
    )
    assert r.text == "Der Zins sinkt deutlich [1]." and r.unsupported == 1


def test_marker_after_full_stop_belongs_to_previous_sentence():
    r = check("Der Zins sinkt deutlich im Juni. [1] Die Inflation bleibt tief. [2]")
    assert r.unsupported == 0 and r.valid == [1, 2]


def test_short_fragments_and_multiline():
    r = check("Ja.\n- Der Zins sinkt deutlich im Juni [1].\n- Unbelegte Aussage ohne jede Quelle.")
    assert "Ja." in r.text and r.text.count(UNSUPPORTED_TAG) == 1
    assert r.text.count("\n") == 2


def test_build_citations_maps_numbers_to_chunks():
    cites = build_citations([2, 1, 1, 9], CHUNKS_DE)
    assert [c.n for c in cites] == [1, 2]
    assert cites[0].chunk_id == CHUNKS_DE[0].chunk_id
    assert cites[1].title == "SNB Bericht" and cites[1].page == 2
    assert cites[1].url == CHUNKS_DE[1].url
