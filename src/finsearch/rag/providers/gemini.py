"""Google Gemini via REST generateContent. Key goes in the x-goog-api-key header, never the URL.

Model names change often; override with LLM_MODEL if the default is retired.
"""

from __future__ import annotations

from typing import Any

from finsearch.contracts import ChatMessage
from finsearch.rag.llm_base import HttpLLMClient, HttpRequest, split_system

BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"


class GeminiClient(HttpLLMClient):
    name = "gemini"
    default_model = "gemini-2.5-flash"
    key_setting = "gemini_api_key"

    def _request(
        self, messages: list[ChatMessage], max_tokens: int, temperature: float
    ) -> HttpRequest:
        system, rest = split_system(messages)
        payload: dict[str, Any] = {
            "contents": [
                {
                    "role": "model" if m.role == "assistant" else "user",
                    "parts": [{"text": m.content}],
                }
                for m in rest
            ],
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        headers = {"x-goog-api-key": self._api_key}
        return f"{BASE_URL}/{self.model}:generateContent", headers, payload

    def _parse(self, data: Any) -> str:
        return "".join(p["text"] for p in data["candidates"][0]["content"]["parts"])
