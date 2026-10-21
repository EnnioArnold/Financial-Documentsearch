"""Offline fakes for the RAG tests."""

from __future__ import annotations

from datetime import date

from finsearch.config import Settings
from finsearch.contracts import (
    ChatMessage,
    Chunk,
    LLMError,
    LLMResponse,
    SearchFilters,
    SearchHit,
    SearchResult,
)


def make_chunk(n: int, text: str, *, title: str = "SNB Lagebeurteilung", lang: str = "de") -> Chunk:
    return Chunk(
        chunk_id=f"{n:016x}-0000",
        doc_id=f"{n:016x}",
        ord=0,
        text=text,
        page=n,
        source="snb",
        title=title,
        url=f"https://example.org/doc{n}.pdf",
        lang=lang,  # type: ignore[arg-type]
        doc_type="speech",
        date=date(2024, 6, 20),
    )


class FakeSearcher:
    def __init__(self, chunks: list[Chunk], score: float = 1.0) -> None:
        self.chunks = chunks
        self.score = score
        self.calls: list[tuple[str, str, int]] = []

    def search(
        self, query: str, method: str = "hybrid", k: int = 10, filters: SearchFilters | None = None
    ) -> SearchResult:
        self.calls.append((query, method, k))
        hits = [SearchHit(c, self.score, i, method) for i, c in enumerate(self.chunks[:k], start=1)]
        return SearchResult(query, method, hits, 1.0)


class FakeLLM:
    """Returns a canned text (or raises) and records the messages it was given."""

    def __init__(self, text: str = "", error: str | None = None, name: str = "fake") -> None:
        self.text, self.error, self.name = text, error, name
        self.calls: list[list[ChatMessage]] = []

    def generate(self, messages: list[ChatMessage], **opts: object) -> LLMResponse:
        self.calls.append(messages)
        if self.error:
            raise LLMError(self.error)
        return LLMResponse(self.text, "fake-model", 1.0)


def settings(**kw: object) -> Settings:
    return Settings(_env_file=None, **kw)  # type: ignore[call-arg, arg-type]


CHUNKS_DE = [
    make_chunk(
        1,
        "Die Schweizerische Nationalbank senkt den Leitzins im Juni 2024 auf 1,25 Prozent. "
        "Die Inflation liegt aktuell im Zielbereich der Nationalbank.",
    ),
    make_chunk(
        2,
        "Die Inflationsprognose für 2025 beträgt 0,4 Prozent. Der Franken bleibt hoch bewertet.",
        title="SNB Bericht",
    ),
]
CHUNKS_EN = [
    make_chunk(
        1,
        "The Federal Reserve held the federal funds rate steady at 5.25 percent in 2023. "
        "Inflation has eased but remains above the two percent target.",
        title="FOMC Statement",
        lang="en",
    ),
]
