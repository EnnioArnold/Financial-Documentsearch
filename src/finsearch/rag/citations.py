"""Citation parsing and validation. Provider-independent: every answer goes through here.

Only numeric markers ([1], [1][2], [1, 3], [1-3]) that point at a supplied source count.
Anything else ([0], [99], [2024], literal chunk ids) is removed and reported.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from finsearch.contracts import NOT_FOUND_TEXT, Chunk, Citation
from finsearch.rag.text import split_sentences

UNSUPPORTED_TAG = "[unbelegt]"
MIN_CLAIM_WORDS = 4  # shorter fragments ("Ja.", headings) are not treated as claims
# "[1]", "[1, 3]", "[1-3]", also "[[1]]"; group 1 = leading whitespace, group 2 = marker body
_MARKER_RE = re.compile(r"(\s*)\[+\s*(\d{1,9}(?:\s*[,\-–]\s*\d{1,9})*)\s*\]+")
_CHUNK_ID_RE = re.compile(r"\s*\[[0-9a-f]{16}-\d{4}\]")
_RANGE_RE = re.compile(r"(\d+)\s*[-–]\s*(\d+)")


@dataclass
class CitationCheck:
    text: str  # cleaned answer, or NOT_FOUND_TEXT if no valid citation remains
    valid: list[int] = field(default_factory=list)  # in order of first appearance
    dropped: list[int] = field(default_factory=list)  # markers pointing at no source
    unsupported: int = 0  # sentences without a valid citation (tagged or dropped)

    @property
    def found(self) -> bool:
        return self.text != NOT_FOUND_TEXT


def _numbers(body: str, n_sources: int) -> tuple[list[int], list[int]]:
    """Split one marker body into (valid, invalid) source numbers."""
    valid: list[int] = []
    invalid: list[int] = []
    for part in re.split(r"\s*,\s*", body):
        rng = _RANGE_RE.fullmatch(part)
        if rng:
            lo, hi = int(rng[1]), int(rng[2])
            # a range is valid only as a whole, "[2010-2020]" is a year span, not a citation
            if 1 <= lo <= hi <= n_sources:
                valid.extend(range(lo, hi + 1))
            else:
                invalid.extend((lo, hi))
        else:
            (valid if 1 <= int(part) <= n_sources else invalid).append(int(part))
    return valid, invalid


def validate_citations(
    answer_text: str, n_sources: int, *, drop_unsupported: bool = False
) -> CitationCheck:
    """Clean `answer_text` against `n_sources` numbered sources.

    1. Normalise valid markers to '[n][m]', delete invalid ones (reported in `dropped`).
    2. Sentences with a claim (>= MIN_CLAIM_WORDS words) but no valid marker are tagged
       '[unbelegt]' or, with `drop_unsupported`, removed.
    3. If no valid citation is left, the answer becomes NOT_FOUND_TEXT.
    """
    valid: list[int] = []
    dropped: list[int] = []

    def fix(match: re.Match[str]) -> str:
        ok, bad = _numbers(match[2], n_sources)
        valid.extend(ok)
        dropped.extend(bad)
        return (match[1] + "".join(f"[{n}]" for n in dict.fromkeys(ok))) if ok else ""

    text = _MARKER_RE.sub(fix, _CHUNK_ID_RE.sub("", answer_text))
    kept_lines: list[str] = []
    unsupported = 0
    for line in text.splitlines():
        kept: list[str] = []
        for sentence in split_sentences(line):
            is_claim = len(sentence.split()) >= MIN_CLAIM_WORDS
            if is_claim and not _MARKER_RE.search(sentence):
                unsupported += 1
                if drop_unsupported:
                    continue
                sentence += f" {UNSUPPORTED_TAG}"
            kept.append(sentence)
        if kept:
            kept_lines.append(" ".join(kept))
    final_valid = list(dict.fromkeys(valid))
    if not final_valid:
        return CitationCheck(NOT_FOUND_TEXT, [], list(dict.fromkeys(dropped)), unsupported)
    return CitationCheck(
        "\n".join(kept_lines), final_valid, list(dict.fromkeys(dropped)), unsupported
    )


def build_citations(valid: list[int], chunks: list[Chunk]) -> list[Citation]:
    """Citation objects for valid source numbers (1-based positions in `chunks`), sorted by n."""
    out = []
    for n in sorted(set(valid)):
        if 1 <= n <= len(chunks):
            c = chunks[n - 1]
            out.append(Citation(n=n, chunk_id=c.chunk_id, title=c.title, page=c.page, url=c.url))
    return out
