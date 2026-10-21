"""Ollama (local). POST {OLLAMA_URL}/api/chat, non-streaming. Model: LLM_MODEL or OLLAMA_MODEL."""

from __future__ import annotations

from typing import Any

from finsearch.contracts import ChatMessage
from finsearch.rag.llm_base import HttpLLMClient, HttpRequest


class OllamaClient(HttpLLMClient):
    name = "ollama"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.model = self.settings.llm_model or self.settings.ollama_model

    def _request(
        self, messages: list[ChatMessage], max_tokens: int, temperature: float
    ) -> HttpRequest:
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        return f"{self.settings.ollama_url.rstrip('/')}/api/chat", {}, payload

    def _parse(self, data: Any) -> str:
        return str(data["message"]["content"])
