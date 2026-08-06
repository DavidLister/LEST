from typing import Protocol

from ..errors import LestError


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


EMBEDDERS: dict[str, type] = {}


def register(name: str):
    def wrap(cls: type) -> type:
        EMBEDDERS[name] = cls
        return cls

    return wrap


def get_embedder(name: str, model: str) -> Embedder:
    try:
        cls = EMBEDDERS[name]
    except KeyError:
        raise LestError(
            f"unknown embedder {name!r}; available: {', '.join(sorted(EMBEDDERS))}"
        ) from None
    return cls(model)
