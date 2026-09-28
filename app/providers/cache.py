"""An on-disk cache in front of any Embedder.

Each vector is stored under a hash of everything that determines it: model, dimension,
kind (doc/query) and the exact text. Unchanged text is never embedded twice, whether
that's re-running ingest, re-chunking (unchanged chunks keep their text), or re-running
eval questions (their query embeddings are cached too).

It wraps an Embedder and is itself an Embedder, so nothing else in the app knows it exists.
"""

import hashlib
import json
from pathlib import Path
from typing import Literal

from app.providers.base import Embedder


class CachedEmbedder:
    def __init__(self, inner: Embedder, cache_dir: Path, model: str, dimension: int):
        self.inner = inner
        # One folder per model and dimension, so vectors from different models can never mix.
        self.dir = cache_dir / f"{model}-{dimension}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def _path(self, text: str, kind: str) -> Path:
        digest = hashlib.sha256(f"{kind}\n{text}".encode()).hexdigest()
        return self.dir / f"{digest}.json"

    def uncached(self, texts: list[str], kind: Literal["doc", "query"]) -> list[str]:
        """The texts that would need an API call. Lets callers estimate cost first."""
        return list(dict.fromkeys(t for t in texts if not self._path(t, kind).exists()))

    def embed(self, texts: list[str], kind: Literal["doc", "query"]) -> list[list[float]]:
        missing = self.uncached(texts, kind)
        if missing:
            for text, vector in zip(missing, self.inner.embed(missing, kind), strict=True):
                self._path(text, kind).write_text(json.dumps(vector))
        self.misses += len(missing)
        self.hits += len(texts) - len(missing)
        return [json.loads(self._path(t, kind).read_text()) for t in texts]
