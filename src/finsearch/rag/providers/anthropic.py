"""Anthropic Messages API via REST. Model names may need updating via LLM_MODEL."""

from __future__ import annotations

from typing import Any

from finsearch.contracts import ChatMessage
from finsearch.rag.llm_base import HttpLLMClient, HttpRequest, split_system

URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"


class AnthropicClient(HttpLLMClient):
    name = "anthropic"
    default_model = "claude-haiku-4-5"
    key_setting = "anthropic_api_key"

    def _request(
        self, messages: list[ChatMessage], max_tokens: int, temperature: float
    ) -> HttpRequest:
        system, rest = split_system(messages)
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": m.role, "content": m.content} for m in rest],
        }
        if system:
            payload["system"] = system
        headers = {"x-api-key": self._api_key, "anthropic-version": API_VERSION}
        return URL, headers, payload

    def _parse(self, data: Any) -> str:
        return "".join(b["text"] for b in data["content"] if b.get("type") == "text")
