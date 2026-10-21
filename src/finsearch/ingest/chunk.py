"""Sentence-aware chunking with overlap.

Token budget: the embedding model (multilingual-e5-small) truncates at 512 real tokens and we
have no tokenizer at ingest time, so tokens are ESTIMATED from the character count (see
estimate_tokens). Measured on the real corpus with the model's own tokenizer: 0.225 tokens per
character (median, de and en alike; p95 0.26) - words are a worse unit because German compounds
make tokens/word range from 1.5 (en) to 1.8 (de). A chunk is built from whole sentences, never
exceeds HARD_CAP_TOKENS (estimated), starts with the last sentence(s) of the previous chunk as
overlap, and records the page on which it starts. A single sentence longer than the budget is
split at word boundaries.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from finsearch.config import get_settings
from finsearch.contracts import Chunk, DocumentMeta
from finsearch.ingest.extract import Page

TOKENS_PER_CHAR = 0.25  # ~ "4 characters per token"; slightly above the measured p95 of 0.26
HARD_CAP_TOKENS = 450  # estimated; real tokens stay below the model's 512 limit
MIN_CHUNK_TOKENS = 30  # a smaller trailing chunk is merged into its predecessor

# Words that end in "." without ending the sentence (compared lower-case, without the dot).
ABBREVIATIONS = frozenset(
    "dr prof mr mrs ms no nr vs bzw ca inkl ggf usw etc mio mrd fig abb tab approx cf ed vol "
    "st art abs ff sec inc corp ltd co jr sr resp evtl ev".split()
)
MAX_ORDINAL_DIGITS = 2
_SENTENCE_END = re.compile(r"[.!?][\"'”»)\]]*$")
_OPENERS = "\"'“„«([ "


def estimate_tokens(text: str) -> int:
    """Estimated subword tokens: ceil(characters * 0.25). Cheap, deterministic, pessimistic."""
    return math.ceil(len(text) * TOKENS_PER_CHAR)


def _max_chars(tokens: int) -> int:
    return max(1, int(tokens / TOKENS_PER_CHAR))


def _ends_sentence(word: str) -> bool:
    if not _SENTENCE_END.search(word):
        return False
    core = word.rstrip(".!?\"'”»)]")
    if not word.rstrip("\"'”»)]").endswith("."):
        return True  # "?" or "!"
    # "5." (German ordinal; 4-digit years do end sentences), "J." / "z." (initials), "z.B." /
    # "U.S." (inner dots), "Dr." etc.
    return not (
        (core.isdigit() and len(core) <= MAX_ORDINAL_DIGITS)
        or len(core) == 1
        or "." in core
        or core.lower() in ABBREVIATIONS
        or not core
    )


def _starts_sentence(word: str) -> bool:
    stripped = word.lstrip(_OPENERS)
    return bool(stripped) and stripped[0].isupper()


def split_sentences(text: str) -> list[str]:
    """Heuristic splitter: boundary = sentence-final punctuation + capitalised next word."""
    words = text.split()
    sentences: list[str] = []
    current: list[str] = []
    for i, word in enumerate(words):
        current.append(word)
        nxt = words[i + 1] if i + 1 < len(words) else None
        if nxt is not None and _ends_sentence(word) and _starts_sentence(nxt):
            sentences.append(" ".join(current))
            current = []
    if current:
        sentences.append(" ".join(current))
    return sentences


@dataclass(frozen=True)
class _Sentence:
    text: str
    page: int

    @property
    def cost(self) -> int:
        """Characters this sentence adds to a chunk (text + the joining space)."""
        return len(self.text) + 1


def _split_long(text: str, page: int, max_chars: int) -> list[_Sentence]:
    """Hard split of an over-long sentence into pieces of at most max_chars characters."""
    pieces: list[str] = []
    current = ""
    for word in text.split():
        while len(word) > max_chars:  # a single "word" (URL, junk) that is too long itself
            if current:
                pieces.append(current)
                current = ""
            pieces.append(word[:max_chars])
            word = word[max_chars:]
        if current and len(current) + 1 + len(word) > max_chars:
            pieces.append(current)
            current = word
        else:
            current = f"{current} {word}" if current else word
    if current:
        pieces.append(current)
    return [_Sentence(piece, page) for piece in pieces]


def _sentences(pages: list[Page], max_chars: int) -> list[_Sentence]:
    out: list[_Sentence] = []
    for page_no, text in pages:
        for sentence in split_sentences(text):
            if len(sentence) > max_chars:
                out.extend(_split_long(sentence, page_no, max_chars))
            else:
                out.append(_Sentence(sentence, page_no))
    return out


def _tail(sentences: list[_Sentence], budget_chars: int) -> list[_Sentence]:
    """Last sentences whose total size fits into budget_chars (possibly none)."""
    taken: list[_Sentence] = []
    total = 0
    for sentence in reversed(sentences):
        if total + sentence.cost > budget_chars:
            break
        taken.append(sentence)
        total += sentence.cost
    return taken[::-1]


def _group(
    sentences: list[_Sentence], body_chars: int, overlap_chars: int
) -> list[list[_Sentence]]:
    """Greedy packing; each group = overlap from the previous group + new sentences."""
    groups: list[list[_Sentence]] = []
    new: list[_Sentence] = []
    overlap: list[_Sentence] = []
    new_chars = 0
    for sentence in sentences:
        if new and new_chars + sentence.cost > body_chars:
            groups.append(overlap + new)
            overlap = _tail(overlap + new, overlap_chars)
            new, new_chars = [], 0
        new.append(sentence)
        new_chars += sentence.cost
    if new:
        tiny = new_chars * TOKENS_PER_CHAR < MIN_CHUNK_TOKENS
        fits = bool(groups) and (
            sum(s.cost for s in groups[-1]) + new_chars <= _max_chars(HARD_CAP_TOKENS)
        )
        if tiny and fits:
            groups[-1] = groups[-1] + new
        else:
            groups.append(overlap + new)
    return groups


def chunk_document(
    doc: DocumentMeta,
    pages: list[Page],
    target_tokens: int | None = None,
    overlap_ratio: float | None = None,
) -> list[Chunk]:
    """Chunks of one cleaned document. Defaults come from Settings (chunk_target_tokens, ...)."""
    settings = get_settings()
    target = settings.chunk_target_tokens if target_tokens is None else target_tokens
    ratio = settings.chunk_overlap_ratio if overlap_ratio is None else overlap_ratio
    overlap_tokens = round(target * ratio)
    body_chars = _max_chars(min(target, HARD_CAP_TOKENS - overlap_tokens))
    overlap_chars = _max_chars(overlap_tokens) if overlap_tokens else 0
    sentences = _sentences(pages, body_chars)
    return [
        Chunk(
            chunk_id=f"{doc.doc_id}-{ord_:04d}",
            doc_id=doc.doc_id,
            ord=ord_,
            text=" ".join(s.text for s in group),
            page=group[0].page,
            source=doc.source,
            title=doc.title,
            url=doc.url,
            lang=doc.lang,
            doc_type=doc.doc_type,
            date=doc.date,
        )
        for ord_, group in enumerate(_group(sentences, body_chars, overlap_chars))
    ]
