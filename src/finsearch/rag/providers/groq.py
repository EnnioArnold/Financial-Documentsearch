"""Groq (OpenAI-compatible chat completions). Model names may need updating via LLM_MODEL."""

from __future__ import annotations

from typing import Any

from finsearch.contracts import ChatMessage
from finsearch.rag.llm_base import HttpLLMClient, HttpRequest

URL = "https://api.groq.com/openai/v1/chat/completions"


class GroqClient(HttpLLMClient):
    name = "groq"
    default_model = "llama-3.3-70b-versatile"
    key_setting = "groq_api_key"

    def _request(
        self, messages: list[ChatMessage], max_tokens: int, temperature: float
    ) -> HttpRequest:
        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        return URL, {"Authorization": f"Bearer {self._api_key}"}, payload

    def _parse(self, data: Any) -> str:
        return str(data["choices"][0]["message"]["content"])
