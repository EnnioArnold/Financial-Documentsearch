"""Cross-encoder reranking (optional last stage)."""

from __future__ import annotations

import dataclasses
import logging
from typing import Any

from finsearch.config import Settings, get_settings
from finsearch.contracts import SearchHit

log = logging.getLogger(__name__)

MAX_PAIR_TOKENS = 512


class CrossEncoderReranker:
    """Scores (query, chunk text) pairs jointly; slower but more precise than bi-encoder similarity."""

    def __init__(self, model_name: str | None = None, max_length: int = MAX_PAIR_TOKENS) -> None:
        self.model_name = model_name or get_settings().reranker_model
        self.max_length = max_length
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            log.info("loading reranker %s", self.model_name)
            self._model = CrossEncoder(self.model_name, device="cpu", max_length=self.max_length)
        return self._model

    def rerank(self, query: str, hits: list[SearchHit], top_k: int) -> list[SearchHit]:
        if not hits:
            return []
        scores = self._load().predict(
            [(query, h.chunk.text) for h in hits], show_progress_bar=False
        )
        # stable sort: equal scores keep the incoming (RRF) order
        order = sorted(range(len(hits)), key=lambda i: -float(scores[i]))[:top_k]
        return [
            dataclasses.replace(hits[i], score=float(scores[i]), rank=pos, method="hybrid_rerank")
            for pos, i in enumerate(order, start=1)
        ]


def get_reranker(settings: Settings | None = None) -> CrossEncoderReranker:
    return CrossEncoderReranker((settings or get_settings()).reranker_model)
