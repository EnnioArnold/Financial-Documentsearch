"""Embedding model wrapper (ADR-002): multilingual-e5-small, CPU, lazy load."""

from __future__ import annotations

import logging
import threading
from typing import Any

from finsearch.config import Settings, get_settings
from finsearch.contracts import Embedder

log = logging.getLogger(__name__)

QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "
SEED = 42
# Pinned HF commits (checked on huggingface.co on 2026-10-21). Models not listed here load "unpinned".
MODEL_REVISIONS: dict[str, str] = {
    "intfloat/multilingual-e5-small": "614241f622f53c4eeff9890bdc4f31cfecc418b3",
}


class SentenceTransformerEmbedder:
    """e5 models need 'query: ' / 'passage: ' prefixes; vectors are L2-normalised (cosine == dot)."""

    def __init__(self, model_name: str, dim: int, batch_size: int = 32) -> None:
        self.model_name = model_name
        self.dim = dim
        self.batch_size = batch_size
        self._model: Any = None
        self._lock = threading.Lock()

    def _load(self) -> Any:
        with self._lock:  # parallel first requests must not load the model several times
            return self._load_locked()

    def _load_locked(self) -> Any:
        if self._model is None:
            import torch
            from sentence_transformers import SentenceTransformer

            torch.manual_seed(SEED)
            revision = MODEL_REVISIONS.get(self.model_name)
            log.info(
                "loading embedding model %s (revision %s)", self.model_name, revision or "unpinned"
            )
            self._model = SentenceTransformer(self.model_name, device="cpu", revision=revision)
            get_dim = (
                getattr(self._model, "get_embedding_dimension", None)
                or self._model.get_sentence_embedding_dimension
            )
            actual = get_dim()
            if actual != self.dim:
                raise ValueError(f"{self.model_name} has dim {actual}, but settings say {self.dim}")
        return self._model

    def _encode(self, texts: list[str], prefix: str) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._load().encode(
            [prefix + t for t in texts],
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [[float(x) for x in v] for v in vectors]

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts, QUERY_PREFIX)

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts, PASSAGE_PREFIX)


def get_embedder(settings: Settings | None = None) -> Embedder:
    s = settings or get_settings()
    return SentenceTransformerEmbedder(s.embedding_model, s.embedding_dim, s.embedding_batch_size)
