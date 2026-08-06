from . import ollama  # noqa: F401  (registers the default embedder)
from .base import EMBEDDERS, Embedder, get_embedder, register

__all__ = ["EMBEDDERS", "Embedder", "get_embedder", "register"]
