"""Cohere implementations of the provider interfaces.

API parameters checked against docs.cohere.com/reference/embed and the installed SDK
(cohere 7.x, `ClientV2`). The SDK itself retries rate-limited requests (HTTP 429)
twice, waiting as long as the server's Retry-After header asks.
"""

from typing import Literal

import cohere

from app.config import Settings

# From the Embed API reference: "Maximum number of texts per call is 96."
EMBED_BATCH_SIZE = 96

INPUT_TYPES = {"doc": "search_document", "query": "search_query"}


def make_client(settings: Settings) -> cohere.ClientV2:
    if settings.cohere_api_key is None:
        raise RuntimeError("COHERE_API_KEY is not set. Add it to .env.")
    return cohere.ClientV2(api_key=settings.cohere_api_key.get_secret_value())


class CohereEmbedder:
    def __init__(self, client: cohere.ClientV2, model: str, dimension: int):
        self.client = client
        self.model = model
        self.dimension = dimension
        self.calls = 0  # API calls made, for tracking the monthly budget

    def embed(self, texts: list[str], kind: Literal["doc", "query"]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), EMBED_BATCH_SIZE):
            response = self.client.embed(
                model=self.model,
                texts=texts[start : start + EMBED_BATCH_SIZE],
                input_type=INPUT_TYPES[kind],
                embedding_types=["float"],
                output_dimension=self.dimension,
            )
            self.calls += 1
            vectors.extend(response.embeddings.float_)
        return vectors
