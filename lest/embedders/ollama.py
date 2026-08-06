import os

import httpx
import ollama

from ..errors import EnvironmentError_
from .base import register

BATCH_SIZE = 32

# Models whose queries (but not documents) want an instruction prefix,
# keyed by model-name prefix.
QUERY_PREFIXES = {
    "qwen3-embedding": (
        "Instruct: Given a web search query, retrieve relevant passages that answer the query\n"
        "Query: {query}"
    ),
}


@register("ollama")
class OllamaEmbedder:
    def __init__(self, model: str, host: str | None = None):
        self.model = model
        self.client = ollama.Client(
            host=host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434"
        )
        self._dim: int | None = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), BATCH_SIZE):
            batch = texts[start : start + BATCH_SIZE]
            try:
                response = self.client.embed(model=self.model, input=batch)
            except ollama.ResponseError as exc:
                raise EnvironmentError_(
                    f"Ollama error for model {self.model!r}: {exc.error} "
                    f"(is the model pulled? try `ollama pull {self.model}`)"
                ) from exc
            except (httpx.HTTPError, ConnectionError) as exc:
                raise EnvironmentError_(
                    f"cannot reach Ollama at {self.client._client.base_url}: {exc}"
                ) from exc
            batch_vectors = list(response.embeddings)
            if len(batch_vectors) != len(batch):
                raise EnvironmentError_(
                    f"Ollama returned {len(batch_vectors)} embeddings for {len(batch)} inputs "
                    f"— is {self.model!r} an embedding model?"
                )
            for vector in batch_vectors:
                if not vector:
                    raise EnvironmentError_(
                        f"Ollama returned an empty embedding — is {self.model!r} "
                        "an embedding model?"
                    )
                if self._dim is None:
                    self._dim = len(vector)
                elif len(vector) != self._dim:
                    raise EnvironmentError_(
                        f"embedding dimension drifted ({self._dim} -> {len(vector)}) "
                        f"within model {self.model!r}"
                    )
            vectors.extend(list(v) for v in batch_vectors)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        template = next(
            (tpl for prefix, tpl in QUERY_PREFIXES.items() if self.model.startswith(prefix)),
            None,
        )
        query = template.format(query=text) if template else text
        return self.embed([query])[0]
