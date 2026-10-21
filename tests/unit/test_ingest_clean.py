"""Cleaning (headers/footers, hyphenation, garbage lines) and language detection."""

from __future__ import annotations

import pytest

from finsearch.ingest.clean import (
    clean_pages,
    detect_language,
    is_garbage,
    is_page_number,
    join_hyphenation,
    normalize,
)

DE = (
    "Die Schweizerische Nationalbank betreibt eine Geldpolitik, die der Preisstabilität dient "
    "und die Konjunktur berücksichtigt. Der Leitzins wurde im Berichtsjahr unverändert belassen."
)
EN = (
    "The central bank conducts a monetary policy that serves price stability and takes the "
    "economic situation into account. The policy rate was left unchanged during the year."
)
FR = (
    "La banque nationale mène une politique monétaire qui sert la stabilité des prix et tient "
    "compte de la situation économique. Le taux directeur est resté inchangé pendant l'année."
)


def test_hyphenation_across_line_break_is_repaired() -> None:
    assert join_hyphenation(["Der Zins-", "satz stieg."]) == "Der Zinssatz stieg."


def test_hyphen_before_und_oder_is_kept() -> None:
    assert join_hyphenation(["der Zins-", "und Kreditmarkt"]) == "der Zins- und Kreditmarkt"
    assert join_hyphenation(["stocks and bonds-", "or cash"]) == "stocks and bonds- or cash"


def test_hyphen_before_capital_or_digit_is_kept() -> None:
    assert join_hyphenation(["Basel-", "III Regeln"]) == "Basel- III Regeln"
    assert join_hyphenation(["Phase-", "2 beginnt"]) == "Phase- 2 beginnt"


def test_plain_lines_are_joined_with_one_space() -> None:
    assert join_hyphenation(["first   line", "second line"]) == "first line second line"


def test_normalize_handles_ligatures_soft_hyphens_and_control_chars() -> None:
    assert normalize("ﬁnance­ rate\x00") == "finance rate"
    assert normalize("a b") == "a b"


def test_dot_leaders_of_a_table_of_contents_are_removed() -> None:
    pages = [(1, "Risks and outlook . . . . . . . . . . 12\nMore text follows here.")]
    assert clean_pages(pages) == [(1, "Risks and outlook 12 More text follows here.")]


@pytest.mark.parametrize("line", ["12", "Seite 3", "Page 4 of 10", "p. 7", "5 / 40", "2025"])
def test_page_number_lines(line: str) -> None:
    assert is_page_number(line)


@pytest.mark.parametrize("line", ["Chapter 3", "3 Inflation", "In 2025 growth slowed"])
def test_not_page_number_lines(line: str) -> None:
    assert not is_page_number(line)


@pytest.mark.parametrize(
    "line", ["1 234 567 8.9 -0.3", "12.5 13.1 (4.2) 0.0 %", "- - - -", "", "   "]
)
def test_garbage_lines(line: str) -> None:
    assert is_garbage(line)


@pytest.mark.parametrize(
    "line",
    [
        "In 2025 GDP grew by 1.3%, up from 0.9% in 2024.",
        "2.5% in 2026 and 1.8%",
        "Net interest income CHF 1 234 million",
    ],
)
def test_numbers_inside_sentences_are_kept(line: str) -> None:
    assert not is_garbage(line)


TOPICS = [
    "inflation",
    "mortgages",
    "capital ratios",
    "liquidity",
    "cyber risk",
    "climate",
    "payments",
]


def _report(
    n_pages: int, header: str = "Annual Report 2099 | Example Bank"
) -> list[tuple[int, str]]:
    return [
        (
            i,
            f"{header}\nThis page discusses {TOPICS[i - 1]} in some detail.\n"
            f"Seite {i}\nExample Bank AG - confidential draft - {i}/{n_pages}",
        )
        for i in range(1, n_pages + 1)
    ]


def test_repeated_headers_and_footers_are_removed() -> None:
    cleaned = clean_pages(_report(6))
    assert len(cleaned) == 6
    for page_no, text in cleaned:
        assert text == f"This page discusses {TOPICS[page_no - 1]} in some detail."


def test_repeated_footer_in_the_middle_of_the_page_is_removed() -> None:
    pages = [
        (i, f"Body {i} sentence here.\nFooter | Name | Bank\nMore chart label {i}")
        for i in range(1, 7)
    ]
    assert all("Footer" not in text for _, text in clean_pages(pages))


def test_nothing_is_removed_in_short_documents() -> None:
    pages = [(i, f"Same header\nBody text number {i}.") for i in range(1, 3)]
    assert all(text.startswith("Same header") for _, text in clean_pages(pages))


def test_long_repeated_sentences_are_not_treated_as_boilerplate() -> None:
    sentence = "This long sentence repeats on every page but it is real content. " * 3
    pages = [(i, f"{sentence.strip()}\nUnique {i}.") for i in range(1, 7)]
    assert all(text.startswith("This long sentence") for _, text in clean_pages(pages))


def test_pages_without_text_are_dropped_but_numbers_are_kept() -> None:
    pages = [(1, "Real text on page one."), (2, "12\n1 2 3\n"), (3, "Real text on page three.")]
    assert [n for n, _ in clean_pages(pages)] == [1, 3]


def test_clean_pages_is_deterministic() -> None:
    assert clean_pages(_report(6)) == clean_pages(_report(6))


def test_german_text_is_detected() -> None:
    assert detect_language(DE) == "de"


def test_english_text_is_detected() -> None:
    assert detect_language(EN) == "en"


def test_other_languages_and_empty_text_map_to_other() -> None:
    assert detect_language(FR) == "other"
    assert detect_language("") == "other"
    assert detect_language("1234 5678 ...") == "other"


def test_language_detection_is_reproducible() -> None:
    mixed = DE + " " + EN
    assert len({detect_language(mixed) for _ in range(20)}) == 1
