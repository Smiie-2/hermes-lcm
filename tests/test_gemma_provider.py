from hermes_lcm.config import LCMConfig
import hermes_lcm.embedding_provider as provider_module
from hermes_lcm.embedding_provider import HttpResponse, resolve_provider


REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"


def test_local_gemma_uses_once_only_role_prompts_and_validates_identity(monkeypatch):
    seen = []

    def transport(*, url, payload, headers, timeout):
        seen.append((url, payload, headers, timeout))
        return HttpResponse(
            200,
            {},
            b'{"data":[{"index":0,"embedding":[' + b','.join([b"0.036084391"] * 768) + b']}]}'
        )

    monkeypatch.setattr(provider_module, "_default_http_transport", transport)

    config = LCMConfig(
        embedding_provider="gemma",
        embedding_model=f"google/embeddinggemma-2@{REVISION}",
        embedding_base_url="http://127.0.0.1:18891/v1",
        embedding_api_key_env="",
        embedding_query_prefix="task: search result | query: ",
        embedding_document_prefix="title: none | text: ",
        embedding_dimension=768,
    )
    provider = resolve_provider(config)
    query = provider.embed_query("how to restore the service?")
    docs = provider.embed_documents(["title: none | text: saved endpoint"])

    assert seen[0][0] == "http://127.0.0.1:18891/v1/embeddings"
    assert seen[0][1]["input"] == ["task: search result | query: how to restore the service?"]
    assert seen[1][1]["input"] == ["title: none | text: saved endpoint"]
    assert all("Authorization" not in request[2] for request in seen)
    assert len(query) == len(docs[0]) == provider.dim == 768
    assert provider.model_id == f"google/embeddinggemma-2@{REVISION}"


def test_gemma_profile_is_loaded_from_supported_hermes_yaml(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    for name in (
        "LCM_EMBEDDINGS_ENABLED", "LCM_EMBEDDING_PROVIDER", "LCM_EMBEDDING_MODEL",
        "LCM_EMBEDDING_BASE_URL", "LCM_EMBEDDING_QUERY_PREFIX",
        "LCM_EMBEDDING_DOCUMENT_PREFIX", "LCM_EMBEDDING_DIMENSION",
        "LCM_EMBEDDING_MAX_BATCH_ITEMS", "LCM_EMBEDDING_STORAGE_DTYPE",
        "LCM_EMBEDDING_STORE_DIM",
    ):
        monkeypatch.delenv(name, raising=False)
    (tmp_path / "config.yaml").write_text(
        "lcm:\n"
        "  embeddings_enabled: true\n"
        "  embedding_provider: gemma\n"
        "  embedding_model: google/embeddinggemma-2@revision+prompt-v1\n"
        "  embedding_base_url: http://127.0.0.1:18891/v1\n"
        "  embedding_query_prefix: 'task: search result | query: '\n"
        "  embedding_document_prefix: 'title: none | text: '\n"
        "  embedding_dimension: 768\n"
        "  embedding_max_batch_items: 8\n"
        "  embedding_storage_dtype: float32\n"
        "  embedding_store_dim: 768\n"
    )
    monkeypatch.setenv("LCM_EMBEDDING_BASE_URL", "http://127.0.0.1:18892/v1")
    config = LCMConfig.from_env()
    assert config.embeddings_enabled is True
    assert config.embedding_provider == "gemma"
    assert config.embedding_base_url == "http://127.0.0.1:18892/v1"
    assert config.embedding_dimension == 768
    assert config.embedding_query_prefix == "task: search result | query: "
    assert config.embedding_document_prefix == "title: none | text: "
    assert config.ignored_config_yaml_lcm_keys == []
