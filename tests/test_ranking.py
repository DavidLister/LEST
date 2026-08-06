import pytest

from lest.errors import LestError
from lest.ranking import parse_agg


def test_max():
    assert parse_agg("max")([0.2, 0.9, 0.5]) == 0.9
    assert parse_agg("max")([0.4]) == 0.4


def test_topk_mean():
    assert parse_agg("topk:2")([0.2, 0.9, 0.5]) == pytest.approx(0.7)
    # fewer scores than K: average what exists
    assert parse_agg("topk:3")([0.4]) == pytest.approx(0.4)
    assert parse_agg("topk")([1.0, 1.0, 1.0, 0.0]) == pytest.approx(1.0)  # default K=3


def test_softmax_between_mean_and_max():
    scores = [0.9, 0.5, 0.1]
    smooth = parse_agg("softmax:0.05")(scores)
    assert sum(scores) / 3 < smooth <= 0.9
    # near-zero temperature approaches max
    assert parse_agg("softmax:0.001")(scores) == pytest.approx(0.9, abs=1e-3)


def test_decay():
    assert parse_agg("decay")([0.8, 0.4, 0.2]) == pytest.approx(0.8 + 0.2 + 0.05)
    assert parse_agg("decay")([0.4, 0.8, 0.2]) == pytest.approx(0.8 + 0.2 + 0.05)  # order-free


def test_count():
    assert parse_agg("count:0.5")([0.9, 0.5, 0.1]) == 2.0
    assert parse_agg("count")([0.9, 0.4]) == 1.0  # default threshold 0.5


def test_rejects_bad_specs():
    for spec in ("nope", "max:3", "topk:0", "topk:x", "softmax:0", "decay:1"):
        with pytest.raises(LestError):
            parse_agg(spec)
