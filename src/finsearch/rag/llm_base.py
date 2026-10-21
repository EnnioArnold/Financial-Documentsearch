"""LLM client factory plus the shared HTTP plumbing for all REST providers.

Providers talk plain REST via httpx (no vendor SDKs). Every failure becomes an `LLMError` whose
message contains only the provider name and a status/kind, never headers, URLs or bodies, so API
keys cannot leak into logs or the UI.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from finsearch.config import Settings
from finsearch.contracts import ChatMessage, LLMClient, LLMError, LLMResponse

log = logging.getLogger(__name__)

RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
BACKOFF_BASE_S = 1.0  # sleeps 1 s, 2 s, 4 s ... between attempts
DEFAULT_MAX_TOKENS = 800
HttpRequest = tuple[str, dict[str, str], dict[str, Any]]  # (url, headers, json payload)
PROVIDER_NAMES = ("extractive", "ollama", "gemini", "groq", "anthropic")


def post_json(
    url: str,
    *,
    provider: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout_s: float,
    retries: int,
    backoff_s: float = BACKOFF_BASE_S,
    transport: httpx.BaseTransport | None = None,
) -> Any:
    """POST JSON; retry timeouts, network errors, 429 and 5xx with exponential backoff."""
    reason = "unknown error"
    for attempt in range(retries + 1):
        if attempt:
            time.sleep(backoff_s * 2 ** (attempt - 1))
        try:
            with httpx.Client(timeout=timeout_s, transport=transport) as client:
                resp = client.post(url, headers=headers, json=payload)
        except httpx.TimeoutException:
            reason = "timeout"
            continue
        except httpx.HTTPError as exc:
            reason = f"network error ({type(exc).__name__})"
            continue
        if resp.status_code in RETRY_STATUS:
            reason = f"HTTP {resp.status_code}"
            log.warning("%s: %s, attempt %d/%d", provider, reason, attempt + 1, retries + 1)
            continue
        if resp.status_code >= 400:  # 4xx (bad key, bad request): retrying cannot help
            raise LLMError(f"{provider}: HTTP {resp.status_code}")
        try:
            return resp.json()
        except ValueError:
            raise LLMError(f"{provider}: response is not JSON") from None
    raise LLMError(f"{provider}: {reason} after {retries + 1} attempts")


def split_system(messages: list[ChatMessage]) -> tuple[str, list[ChatMessage]]:
    """Providers like Gemini/Anthropic take the system prompt separately."""
    system = "\n\n".join(m.content for m in messages if m.role == "system")
    return system, [m for m in messages if m.role != "system"]


class HttpLLMClient:
    """Common behaviour: model/key resolution, timing, error mapping. Subclasses build + parse."""

    name = ""
    default_model = ""
    key_setting = ""  # Settings attribute holding the API key ("" = no key needed)

    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.BaseTransport | None = None,
        backoff_s: float = BACKOFF_BASE_S,
    ) -> None:
        self.settings = settings
        self.model = settings.llm_model or self.default_model
        self._api_key: str = getattr(settings, self.key_setting) if self.key_setting else ""
        if self.key_setting and not self._api_key:
            raise LLMError(f"{self.name}: API-Key fehlt (setze {self.key_setting.upper()})")
        self._transport = transport
        self._backoff_s = backoff_s

    def _request(
        self, messages: list[ChatMessage], max_tokens: int, temperature: float
    ) -> HttpRequest:
        raise NotImplementedError

    def _parse(self, data: Any) -> str:
        """Extract the answer text from the decoded JSON response."""
        raise NotImplementedError

    def generate(self, messages: list[ChatMessage], **opts: object) -> LLMResponse:
        max_tokens = int(opts.get("max_tokens", DEFAULT_MAX_TOKENS))  # type: ignore[call-overload]
        temperature = float(opts.get("temperature", 0.0))  # type: ignore[arg-type]
        url, headers, payload = self._request(messages, max_tokens, temperature)
        start = time.perf_counter()
        data = post_json(
            url,
            provider=self.name,
            headers=headers,
            payload=payload,
            timeout_s=self.settings.llm_timeout_s,
            retries=self.settings.llm_max_retries,
            backoff_s=self._backoff_s,
            transport=self._transport,
        )
        try:
            text = self._parse(data)
        except (KeyError, IndexError, TypeError, AttributeError):
            raise LLMError(f"{self.name}: unexpected response format") from None
        if not text.strip():
            raise LLMError(f"{self.name}: empty response")
        return LLMResponse(
            text=text, model=self.model, latency_ms=(time.perf_counter() - start) * 1000
        )


def get_llm_client(settings: Settings) -> LLMClient:
    """Build the client for `settings.llm_provider`.

    Raises ValueError for an unknown provider name and LLMError when a keyed provider has no
    key. Callers (answer.py / the API) catch LLMError and fall back to the extractive provider.
    """
    # imported here: providers import this module for HttpLLMClient
    from finsearch.rag.providers.anthropic import AnthropicClient
    from finsearch.rag.providers.extractive import ExtractiveClient
    from finsearch.rag.providers.gemini import GeminiClient
    from finsearch.rag.providers.groq import GroqClient
    from finsearch.rag.providers.ollama import OllamaClient

    registry: dict[str, Any] = {
        "extractive": ExtractiveClient,
        "ollama": OllamaClient,
        "gemini": GeminiClient,
        "groq": GroqClient,
        "anthropic": AnthropicClient,
    }
    provider = settings.llm_provider.strip().lower()
    if provider not in registry:
        raise ValueError(
            f"Unbekannter LLM_PROVIDER {settings.llm_provider!r}; erlaubt: {', '.join(PROVIDER_NAMES)}"
        )
    client: LLMClient = (
        registry[provider](settings) if provider != "extractive" else registry[provider]()
    )
    return client
