"""All settings in one typed place, read from environment variables or `.env`.

Every knob the pipeline has (model IDs, how many chunks to retrieve, rerank on/off)
lives here, so experiments like the rerank ablation are config changes, not code changes.

Run `uv run python -m app.config` to see the active settings (the API key is masked).
"""

from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Values come from env vars first, then `.env`. Names are case-insensitive:
    # RERANK_ENABLED=false in the environment sets `rerank_enabled`.
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # SecretStr masks the value in logs, reprs and tracebacks.
    # Call `.get_secret_value()` only at the point where the SDK needs it.
    cohere_api_key: SecretStr | None = None

    database_url: str = "postgresql://rag:rag@localhost:5432/rag"

    # Model IDs, checked against docs.cohere.com/docs/models.
    # Changing the embed model or dimension requires a full re-index.
    embed_model: str = "embed-v4.0"
    embed_dimension: int = 1536
    rerank_model: str = "rerank-v4.0-pro"
    chat_model: str = "command-a-03-2025"
    # Rewrites follow-up questions before retrieval. A small model (e.g. command-r7b-12-2024)
    # would do in production; on the trial key every call costs the same.
    rewrite_model: str = "command-a-03-2025"

    # Chunking strategy, by name (see CHUNKERS in app/chunk.py).
    chunker: str = "block"

    # Retrieval funnel: vector search casts a wide net, rerank narrows it.
    retrieve_top_k: int = 50
    rerank_top_n: int = 8
    # Off = pass the top `rerank_top_n` vector-search results straight to Chat.
    # This is the switch for the with/without-rerank ablation.
    rerank_enabled: bool = True
    # TODO(owner): if the best rerank score is below this, abstain without calling Chat.
    # None = off (always let Command decide). Pick a value from eval results, not by guessing.
    min_rerank_score: float | None = None


@lru_cache
def get_settings() -> Settings:
    """Load settings once and reuse them."""
    return Settings()


if __name__ == "__main__":
    for name, value in get_settings().model_dump().items():
        print(f"{name:18} {value}")
