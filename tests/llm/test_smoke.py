"""Real-provider smoke tests. Run with `pytest -m llm`; skipped unless the key is present."""

import os

import pytest

from finsearch.config import Settings
from finsearch.contracts import ChatMessage
from finsearch.rag.llm_base import get_llm_client

pytestmark = pytest.mark.llm


@pytest.mark.parametrize("provider", ["gemini", "groq", "anthropic"])
def test_real_provider_answers(provider):
    cfg = Settings(llm_provider=provider)
    if not getattr(cfg, f"{provider}_api_key") and not os.environ.get(
        f"{provider.upper()}_API_KEY"
    ):
        pytest.skip(f"{provider.upper()}_API_KEY not set")
    out = get_llm_client(cfg).generate([ChatMessage("user", "Antworte nur mit dem Wort: ok")])
    assert out.text.strip()
