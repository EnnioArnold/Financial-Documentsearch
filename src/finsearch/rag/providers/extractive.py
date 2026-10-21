"""Extractive "provider": picks the best sentences from the retrieved chunks, generates nothing.

No generated text means no hallucination: every returned sentence is verbatim source text,
tagged with the number of the source it came from.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from finsearch.contracts import ChatMessage, Chunk, LLMError, LLMResponse
from finsearch.rag.text import content_terms, detect_lang, split_sentences

MAX_SENTENCES = 3
MIN_SENTENCE_WORDS = 5
MAX_SENTENCE_CHARS = 400
BM25_K1 = 1.5
BM25_B = 0.75
_SOURCE_FOOTNOTE_RE = re.compile(
    r"\s*\[\d+\]"
)  # footnote markers in the source must not look like ours


class ExtractiveClient:
    """Marker so `get_llm_client` can return something for LLM_PROVIDER=extractive."""

    name = "extractive"

    def generate(self, messages: list[ChatMessage], **opts: object) -> LLMResponse:
        raise LLMError("extractive: erzeugt keinen Text, answer.py nutzt extractive_answer()")


def extractive_answer(
    question: str, chunks: list[Chunk], max_sentences: int = MAX_SENTENCES
) -> str:
    """Return up to `max_sentences` best-matching source sentences as 'sentence [n]' lines.

    Scoring is BM25 over the pool of all candidate sentences (question terms vs sentence
    terms, idf from the pool). Returns "" when no sentence shares a term with the question.
    """
    lang = detect_lang(question)
    q_terms = set(content_terms(question, lang))
    pool: list[tuple[int, int, str, Counter[str]]] = []  # (n, position, sentence, term counts)
    seen: set[str] = set()
    for n, chunk in enumerate(chunks, start=1):
        for pos, sent in enumerate(split_sentences(_SOURCE_FOOTNOTE_RE.sub("", chunk.text))):
            if (
                len(sent.split()) < MIN_SENTENCE_WORDS
                or len(sent) > MAX_SENTENCE_CHARS
                or sent in seen
            ):
                continue
            seen.add(sent)
            pool.append((n, pos, sent, Counter(content_terms(sent, detect_lang(chunk.text)))))
    if not pool or not q_terms:
        return ""
    avg_len = sum(sum(c.values()) for *_, c in pool) / len(pool) or 1.0
    df = {t: sum(t in c for *_, c in pool) for t in q_terms}

    def score(counts: Counter[str]) -> float:
        length = sum(counts.values())
        total = 0.0
        for t in q_terms:
            tf = counts.get(t, 0)
            if tf:
                idf = math.log(1 + (len(pool) - df[t] + 0.5) / (df[t] + 0.5))
                total += (
                    idf
                    * tf
                    * (BM25_K1 + 1)
                    / (tf + BM25_K1 * (1 - BM25_B + BM25_B * length / avg_len))
                )
        return total

    ranked = sorted(((score(c), -i) for i, (*_, c) in enumerate(pool)), reverse=True)
    best = sorted(-i for s, i in ranked[:max_sentences] if s > 0)  # back to reading order
    return "\n".join(f"{pool[i][2]} [{pool[i][0]}]" for i in best)
