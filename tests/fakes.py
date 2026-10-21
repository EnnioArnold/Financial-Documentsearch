"""Deterministic test doubles shared by all test modules (import as `from fakes import HashEmbedder`)."""

from __future__ import annotations

import dataclasses
import hashlib
import math
import re

from finsearch.contracts import Chunk, SearchHit

DIM = 384
_TOKEN = re.compile(r"\w+", re.UNICODE)


def _bucket(token: str, dim: int) -> tuple[int, float]:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()  # stable across runs
    value = int.from_bytes(digest, "big")
    return value % dim, 1.0 if (value >> 40) & 1 else -1.0


class HashEmbedder:
    """Hashed bag-of-words, L2-normalised: texts sharing words get a high cosine similarity.

    Not semantic (no synonyms) - that is what the real model is for. Query and passage vectors use the
    same function, so a query is most similar to the passage with the most overlapping words.
    """

    def __init__(self, dim: int = DIM) -> None:
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in _TOKEN.findall(text.lower()):
            idx, sign = _bucket(token, self.dim)
            vec[idx] += sign
        norm = math.sqrt(sum(x * x for x in vec))
        if norm == 0.0:  # no tokens: return a fixed unit vector so the result is still normalised
            vec[0], norm = 1.0, 1.0
        return [x / norm for x in vec]

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]


class FakeReranker:
    """Ranks hits by the number of distinct query words they contain (ties keep the incoming order)."""

    def rerank(self, query: str, hits: list[SearchHit], top_k: int) -> list[SearchHit]:
        words = set(_TOKEN.findall(query.lower()))

        def overlap(h: SearchHit) -> int:
            return len(words & set(_TOKEN.findall(h.chunk.text.lower())))

        ranked = sorted(hits, key=lambda h: -overlap(h))[:top_k]
        return [
            dataclasses.replace(h, score=float(overlap(h)), rank=i, method="hybrid_rerank")
            for i, h in enumerate(ranked, start=1)
        ]


def make_chunk(
    chunk_id: str = "d1-0000", text: str = "synthetic text", **overrides: object
) -> Chunk:
    """A valid Chunk with sensible defaults; override any field by keyword."""
    base = {
        "chunk_id": chunk_id,
        "doc_id": chunk_id.rsplit("-", 1)[0],
        "ord": 0,
        "text": text,
        "page": 1,
        "source": "snb",
        "title": "Synthetic title",
        "url": "https://example.invalid/doc",
        "lang": "en",
        "doc_type": "speech",
        "date": None,
    }
    base.update(overrides)
    return Chunk(**base)  # type: ignore[arg-type]
