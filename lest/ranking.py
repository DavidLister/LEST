"""Chunk-to-document score aggregation strategies.

Each strategy maps the list of a document's chunk similarities (higher is
better) to a single document score. Selected per-query with --agg, as
"name" or "name:param".
"""

import math
from collections.abc import Callable

from .errors import LestError

Aggregator = Callable[[list[float]], float]

AGG_HELP = "max | topk:K | softmax:T | decay | count:T"


def _topk(scores: list[float], k: int) -> float:
    best = sorted(scores, reverse=True)[:k]
    return sum(best) / len(best)


def _softmax(scores: list[float], temperature: float) -> float:
    weights = [math.exp(s / temperature) for s in scores]
    total = sum(weights)
    return sum(s * w for s, w in zip(scores, weights, strict=True)) / total


def _decay(scores: list[float]) -> float:
    return sum(s / 2**i for i, s in enumerate(sorted(scores, reverse=True)))


def _count(scores: list[float], threshold: float) -> float:
    return float(sum(1 for s in scores if s >= threshold))


def parse_agg(spec: str) -> Aggregator:
    name, _, param = spec.partition(":")
    try:
        if name == "max":
            _reject_param(name, param)
            return max
        if name == "topk":
            k = int(param) if param else 3
            if k < 1:
                raise ValueError
            return lambda scores: _topk(scores, k)
        if name == "softmax":
            temperature = float(param) if param else 0.05
            if temperature <= 0:
                raise ValueError
            return lambda scores: _softmax(scores, temperature)
        if name == "decay":
            _reject_param(name, param)
            return _decay
        if name == "count":
            threshold = float(param) if param else 0.5
            return lambda scores: _count(scores, threshold)
    except ValueError:
        raise LestError(f"invalid parameter {param!r} for aggregation {name!r}") from None
    raise LestError(f"unknown aggregation {spec!r}; available: {AGG_HELP}")


def _reject_param(name: str, param: str) -> None:
    if param:
        raise LestError(f"aggregation {name!r} takes no parameter")
