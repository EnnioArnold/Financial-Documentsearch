"""Text cleaning and language detection.

Pipeline per document (clean_pages): drop recurring page headers/footers (short lines that
repeat on many pages), drop page-number-only and "mostly digits" lines, repair hyphenation
across line breaks, collapse whitespace.
Each page becomes ONE line of text (paragraph structure is not needed for sentence chunking).
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

from langdetect import DetectorFactory, detect
from langdetect.lang_detect_exception import LangDetectException

from finsearch.contracts import Lang
from finsearch.ingest.extract import Page

DetectorFactory.seed = 0  # langdetect is randomised otherwise => non-reproducible ingest

MAX_BOILERPLATE_CHARS = 150  # longer lines are never treated as running headers/footers
MIN_PAGES_FOR_REPEATS = 4  # below this "recurring" is meaningless
REPEAT_MIN_PAGES = 3  # a header must appear on at least this many pages ...
REPEAT_MIN_FRACTION = 0.4  # ... and on at least this share of all pages
MIN_ALPHA_RATIO = 0.25  # lines with a smaller share of letters are table/figure garbage
LANG_SAMPLE_CHARS = 10_000

# "Zins-\nund Kreditmarkt": the hyphen belongs to the compound, do not glue the words together
KEEP_HYPHEN_BEFORE = frozenset({"und", "oder", "bzw", "sowie", "and", "or", "et", "ou", "e", "o"})

_PAGE_NUMBER = re.compile(
    r"^(?:page|seite|p\.|s\.)?\s*\d{1,4}(?:\s*(?:/|of|von)\s*\d{1,4})?$", re.IGNORECASE
)
_DIGITS = re.compile(r"\d+")
_SPACES = re.compile(r"\s+")
_DOT_LEADER = re.compile(r"(?:\s*\.){4,}")  # table-of-contents filler: "Risks . . . . . 12"
_SOFT_HYPHEN = "­"


def normalize(text: str) -> str:
    """NFKC (ligatures, nbsp), soft hyphens and dot leaders removed, control characters dropped."""
    text = unicodedata.normalize("NFKC", text).replace(_SOFT_HYPHEN, "")
    text = _DOT_LEADER.sub(" ", text)
    return "".join(ch for ch in text if ch in "\n\t" or unicodedata.category(ch)[0] != "C")


def _lines(text: str) -> list[str]:
    return [line.strip() for line in text.split("\n")]


def _key(line: str) -> str:
    """Header/footer identity: page numbers differ between pages, the rest must not."""
    return _DIGITS.sub("#", line.lower())


def find_repeated_lines(pages_lines: list[list[str]]) -> set[str]:
    """Keys of short lines that recur on many pages = running headers/footers.

    Position on the page is deliberately ignored: on slide decks the footer is often extracted
    in the middle of the page text. Only short lines qualify, so real sentences are never removed.
    """
    n = len(pages_lines)
    if n < MIN_PAGES_FOR_REPEATS:
        return set()
    counts: Counter[str] = Counter()
    for lines in pages_lines:
        counts.update({_key(line) for line in lines if 0 < len(line) <= MAX_BOILERPLATE_CHARS})
    threshold = max(REPEAT_MIN_PAGES, REPEAT_MIN_FRACTION * n)
    return {key for key, count in counts.items() if count >= threshold}


def is_page_number(line: str) -> bool:
    return bool(_PAGE_NUMBER.match(line))


def is_garbage(line: str) -> bool:
    """Table/figure remnants: (almost) no letters. Numbers inside real sentences survive."""
    chars = [c for c in line if not c.isspace()]
    if not chars:
        return True
    return sum(c.isalpha() for c in chars) / len(chars) < MIN_ALPHA_RATIO


def _ends_with_hyphenated_word(line: str) -> bool:
    return len(line) > 2 and line[-1] == "-" and line[-2].isalpha()


def join_hyphenation(lines: list[str]) -> str:
    """ "Zins-" + "satz" -> "Zinssatz"; "Zins-" + "und" -> "Zins- und"."""
    out = ""
    for line in lines:
        if out and _ends_with_hyphenated_word(out) and line[:1].islower():
            first_word = line.split(" ", 1)[0]
            out = out + " " + line if first_word in KEEP_HYPHEN_BEFORE else out[:-1] + line
        else:
            out = f"{out} {line}" if out else line
    return _SPACES.sub(" ", out).strip()


def clean_pages(pages: list[Page]) -> list[Page]:
    """Cleaned copy of pages; pages that end up empty are dropped (page numbers are kept)."""
    pages_lines = [_lines(normalize(text)) for _, text in pages]
    repeated = find_repeated_lines(pages_lines)
    cleaned: list[Page] = []
    for (page_no, _), lines in zip(pages, pages_lines, strict=True):
        kept = [
            line
            for line in lines
            if line
            and _key(line) not in repeated
            and not is_page_number(line)
            and not is_garbage(line)
        ]
        text = join_hyphenation(kept)
        if text:
            cleaned.append((page_no, text))
    return cleaned


def detect_language(text: str) -> Lang:
    """ "de" | "en" | "other" (French/Italian/undetectable all map to "other")."""
    sample = text[:LANG_SAMPLE_CHARS]
    if not sample.strip():
        return "other"
    try:
        code = detect(sample)
    except LangDetectException:
        return "other"
    if code == "de":
        return "de"
    if code == "en":
        return "en"
    return "other"
