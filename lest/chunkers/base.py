from typing import Protocol

from ..errors import LestError


class Chunker(Protocol):
    name: str

    def chunk(self, text: str) -> list[str]: ...


CHUNKERS: dict[str, type] = {}


def register(cls: type) -> type:
    CHUNKERS[cls.name] = cls
    return cls


def get_chunker(name: str) -> Chunker:
    try:
        return CHUNKERS[name]()
    except KeyError:
        raise LestError(
            f"unknown chunker {name!r}; available: {', '.join(sorted(CHUNKERS))}"
        ) from None
