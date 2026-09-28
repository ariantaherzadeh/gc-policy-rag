"""Build the configured providers. The rest of the app gets its models from here."""

from app.config import get_settings
from app.corpus import ROOT
from app.providers.cache import CachedEmbedder
from app.providers.cohere import CohereEmbedder, make_client

EMBED_CACHE_DIR = ROOT / "data" / "cache" / "embeddings"


def get_embedder() -> CachedEmbedder:
    settings = get_settings()
    inner = CohereEmbedder(make_client(settings), settings.embed_model, settings.embed_dimension)
    return CachedEmbedder(inner, EMBED_CACHE_DIR, settings.embed_model, settings.embed_dimension)
