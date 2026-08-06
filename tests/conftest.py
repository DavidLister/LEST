import hashlib
import math
import re

import pytest

from lest.embedders import EMBEDDERS

DIM = 32


class FakeEmbedder:
    """Deterministic bag-of-words hashing embedder: shared words => similar vectors."""

    def __init__(self, model: str):
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    @staticmethod
    def _vector(text: str) -> list[float]:
        vector = [0.0] * DIM
        for word in re.findall(r"\w+", text.lower()):
            digest = hashlib.sha256(word.encode()).digest()
            vector[digest[0] % DIM] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]


@pytest.fixture
def fake_embedder(monkeypatch):
    monkeypatch.setitem(EMBEDDERS, "fake", FakeEmbedder)
    return FakeEmbedder


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    directory = tmp_path / "lest-data"
    monkeypatch.setenv("LEST_DATA_DIR", str(directory))
    return directory
