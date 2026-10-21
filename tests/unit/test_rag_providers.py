import json

import httpx
import pytest
from rag_fakes import settings

from finsearch.contracts import ChatMessage, LLMError
from finsearch.rag.llm_base import get_llm_client, post_json
from finsearch.rag.providers.anthropic import AnthropicClient
from finsearch.rag.providers.extractive import ExtractiveClient
from finsearch.rag.providers.gemini import GeminiClient
from finsearch.rag.providers.groq import GroqClient
from finsearch.rag.providers.ollama import OllamaClient

KEY = "sk-secret-123"
MSGS = [ChatMessage("system", "SYS"), ChatMessage("user", "USER")]


def client_for(cls, handler, **kw):
    cfg = settings(
        gemini_api_key=KEY, groq_api_key=KEY, anthropic_api_key=KEY, llm_max_retries=2, **kw
    )
    return cls(cfg, transport=httpx.MockTransport(handler), backoff_s=0)


def recorder(body, status=200):
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(status, json=body)

    return handler, seen


def test_ollama_request_and_parse():
    handler, seen = recorder({"message": {"content": "Hallo [1]"}})
    llm = client_for(OllamaClient, handler, ollama_model="qwen")
    out = llm.generate(MSGS)
    req = seen[0]
    body = json.loads(req.content)
    assert str(req.url) == "http://127.0.0.1:11434/api/chat"
    assert body["stream"] is False and body["model"] == "qwen"
    assert body["messages"][0] == {"role": "system", "content": "SYS"}
    assert out.text == "Hallo [1]" and out.model == "qwen" and out.latency_ms >= 0


def test_llm_model_overrides_ollama_model():
    handler, _ = recorder({})
    assert client_for(OllamaClient, handler, llm_model="x").model == "x"


def test_gemini_key_in_header_not_url():
    handler, seen = recorder(
        {"candidates": [{"content": {"parts": [{"text": "A"}, {"text": "B"}]}}]}
    )
    out = client_for(GeminiClient, handler).generate(MSGS)
    req = seen[0]
    body = json.loads(req.content)
    assert out.text == "AB"
    assert req.headers["x-goog-api-key"] == KEY and KEY not in str(req.url)
    assert str(req.url).endswith(":generateContent")
    assert body["systemInstruction"]["parts"][0]["text"] == "SYS"
    assert body["contents"] == [{"role": "user", "parts": [{"text": "USER"}]}]


def test_groq_request_shape():
    handler, seen = recorder({"choices": [{"message": {"content": "ok [1]"}}]})
    out = client_for(GroqClient, handler).generate(MSGS, max_tokens=50, temperature=0.2)
    req, body = seen[0], json.loads(seen[0].content)
    assert str(req.url) == "https://api.groq.com/openai/v1/chat/completions"
    assert req.headers["authorization"] == f"Bearer {KEY}"
    assert body["max_tokens"] == 50 and body["temperature"] == 0.2
    assert out.text == "ok [1]"


def test_anthropic_request_shape():
    handler, seen = recorder({"content": [{"type": "text", "text": "ok"}, {"type": "x"}]})
    out = client_for(AnthropicClient, handler).generate(MSGS)
    req, body = seen[0], json.loads(seen[0].content)
    assert str(req.url) == "https://api.anthropic.com/v1/messages"
    assert req.headers["x-api-key"] == KEY and "anthropic-version" in req.headers
    assert body["system"] == "SYS" and body["max_tokens"] > 0
    assert body["messages"] == [{"role": "user", "content": "USER"}]
    assert out.text == "ok"


def test_retries_then_success():
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) < 3:
            return httpx.Response(503, text=f"overloaded {KEY}")
        return httpx.Response(200, json={"message": {"content": "fine"}})

    assert client_for(OllamaClient, handler).generate(MSGS).text == "fine"
    assert len(calls) == 3


@pytest.mark.parametrize("status", [429, 500, 503])
def test_retry_exhausted_maps_to_llmerror_without_key(status):
    handler, seen = recorder({"error": KEY}, status)
    with pytest.raises(LLMError) as e:
        client_for(GroqClient, handler).generate(MSGS)
    assert len(seen) == 3 and f"HTTP {status}" in str(e.value) and KEY not in str(e.value)


def test_client_error_is_not_retried():
    handler, seen = recorder({"error": KEY}, 401)
    with pytest.raises(LLMError, match="HTTP 401") as e:
        client_for(GroqClient, handler).generate(MSGS)
    assert len(seen) == 1 and KEY not in str(e.value)


def test_timeout_and_network_errors():
    def timeout(request):
        raise httpx.ReadTimeout("slow", request=request)

    def refused(request):
        raise httpx.ConnectError(f"refused {KEY}", request=request)

    with pytest.raises(LLMError, match="timeout"):
        client_for(OllamaClient, timeout).generate(MSGS)
    with pytest.raises(LLMError, match="network error") as e:
        client_for(OllamaClient, refused).generate(MSGS)
    assert KEY not in str(e.value)


@pytest.mark.parametrize("body", [{"unexpected": 1}, {"message": {"content": "  "}}, ["x"]])
def test_bad_response_shapes(body):
    handler, _ = recorder(body)
    with pytest.raises(LLMError):
        client_for(OllamaClient, handler).generate(MSGS)


def test_non_json_response():
    handler = lambda r: httpx.Response(200, text="<html>")  # noqa: E731
    with pytest.raises(LLMError, match="not JSON"):
        post_json(
            "http://x",
            provider="p",
            headers={},
            payload={},
            timeout_s=1,
            retries=0,
            transport=httpx.MockTransport(handler),
        )


def test_factory_providers():
    cfg = dict(gemini_api_key=KEY, groq_api_key=KEY, anthropic_api_key=KEY)
    for name, cls in [
        ("extractive", ExtractiveClient),
        ("ollama", OllamaClient),
        ("gemini", GeminiClient),
        ("groq", GroqClient),
        ("anthropic", AnthropicClient),
    ]:
        client = get_llm_client(settings(llm_provider=name, **cfg))
        assert isinstance(client, cls) and client.name == name


def test_factory_unknown_provider():
    with pytest.raises(ValueError, match="Unbekannter LLM_PROVIDER"):
        get_llm_client(settings(llm_provider="openai"))


@pytest.mark.parametrize("name", ["gemini", "groq", "anthropic"])
def test_factory_missing_key(name):
    with pytest.raises(LLMError, match=f"{name.upper()}_API_KEY"):
        get_llm_client(settings(llm_provider=name))
