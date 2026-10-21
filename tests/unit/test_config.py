from finsearch.config import Settings


def test_defaults_and_index_name(monkeypatch):
    monkeypatch.delenv("ES_INDEX_VERSION", raising=False)
    s = Settings(_env_file=None)
    assert s.index_name == "finsearch-v1"
    assert s.llm_provider == "extractive"
    assert s.embedding_dim == 384


def test_env_override(monkeypatch):
    monkeypatch.setenv("ES_INDEX_VERSION", "v2")
    assert Settings(_env_file=None).index_name == "finsearch-v2"
