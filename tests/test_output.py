import json

from lest.output import format_json, format_tsv
from lest.query import SearchResult


def result(**overrides):
    fields = {
        "score": 0.87654321,
        "title": "A Title",
        "paths": ["/tmp/a.pdf", "/tmp/b.pdf"],
        "key": "KEY1",
        "meta": {"year": "2020"},
        "best_chunk": "the best chunk",
    }
    fields.update(overrides)
    return SearchResult(**fields)


def test_tsv_shape():
    line = format_tsv(result())
    assert line == "0.8765\tA Title\t/tmp/a.pdf;/tmp/b.pdf"


def test_tsv_scrubs_tabs_and_newlines():
    line = format_tsv(result(title="Bad\ttitle\nwith breaks\r\n"))
    assert line.count("\t") == 2
    assert "\n" not in line and "\r" not in line
    assert line.split("\t")[1] == "Bad title with breaks"


def test_json_roundtrip_preserves_title():
    payload = json.loads(format_json(result(title="Tabs\tand\nnewlines stay")))
    assert payload["title"] == "Tabs\tand\nnewlines stay"
    assert payload["paths"] == ["/tmp/a.pdf", "/tmp/b.pdf"]
    assert payload["meta"] == {"year": "2020"}
    assert payload["best_chunk"] == "the best chunk"
