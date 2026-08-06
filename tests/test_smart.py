"""Smart search: facet weighting math, parse resolution, listwise rerank."""

import pytest

from lest.catalog import Catalog
from lest.query import SearchResult
from lest.smart import Facet, ParsedQuery, facet_multiplier, parse_query, rerank


def result(**kwargs):
    defaults = dict(score=1.0, title="t", paths=[], key="k", meta={},
                    best_chunk="chunk text", tags=[], authors=[], doc_type=None)
    defaults.update(kwargs)
    return SearchResult(**defaults)


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)

    def ping(self):
        pass

    def call(self, prompt, schema, **kwargs):
        return self.responses.pop(0) if self.responses else None


def test_weight_one_is_a_filter():
    parsed = ParsedQuery("q", facets=[Facet({"Neugebauer, Jörg"}, 1.0, "author")])
    match = result(authors=["Neugebauer, Jörg"])
    miss = result(authors=["Smith, John"])
    assert facet_multiplier(match, parsed) == 1.0
    assert facet_multiplier(miss, parsed) == 0.0


def test_partial_weight_nudges():
    parsed = ParsedQuery("q", facets=[Facet({"mocvd"}, 0.4, "tag")])
    assert facet_multiplier(result(tags=["mocvd"]), parsed) == 1.0
    assert facet_multiplier(result(tags=["other"]), parsed) == pytest.approx(0.6)


def test_facets_compound():
    parsed = ParsedQuery("q", facets=[
        Facet({"mocvd"}, 0.5, "tag"),
        Facet({"review"}, 0.5, "doctype"),
    ])
    neither = result(tags=["x"], doc_type="paper")
    assert facet_multiplier(neither, parsed) == pytest.approx(0.25)


def test_author_falls_back_to_creators_meta():
    parsed = ParsedQuery("q", facets=[Facet({"Neugebauer, Jörg"}, 1.0, "author")])
    legacy = result(authors=[], meta={"creators": "Neugebauer, Jörg; Van de Walle, Chris"})
    assert facet_multiplier(legacy, parsed) == 1.0
    other = result(authors=[], meta={"creators": "Smith, John"})
    assert facet_multiplier(other, parsed) == 0.0


def test_year_range_nudges_not_filters():
    parsed = ParsedQuery("q", year_from=2000, year_to=2005)
    assert facet_multiplier(result(meta={"year": "2003"}), parsed) == 1.0
    assert facet_multiplier(result(meta={"year": "1990"}), parsed) == pytest.approx(0.4)
    assert facet_multiplier(result(meta={}), parsed) == pytest.approx(0.4)


def test_parse_query_resolves_through_catalog(data_dir):
    cat = Catalog()
    cat.resolve_term("tag", "gallium nitride")
    cat.add_alias("tag", "gallium nitride", "gan")
    cat.resolve_author("Neugebauer, Jörg")
    cat.close()

    fake = FakeClient([{
        "semantic_query": "point defect formation energies",
        "tags": [{"name": "GaN", "weight": 0.5}],
        "authors": [{"name": "neugebauer", "weight": 1.0}],
        "doc_types": [],
        "year_from": 0, "year_to": 0,
    }])
    parsed = parse_query(fake, "defect formation energies in GaN by neugebauer")
    assert parsed.semantic_query == "point defect formation energies"
    tag_facet = next(f for f in parsed.facets if f.kind == "tag")
    assert "gallium nitride" in tag_facet.terms and "gan" in tag_facet.terms
    author_facet = next(f for f in parsed.facets if f.kind == "author")
    assert "Neugebauer, Jörg" in author_facet.terms
    assert author_facet.weight == 1.0


def test_parse_drops_generic_doctype_facets(data_dir):
    fake = FakeClient([{
        "semantic_query": "water ice on the moon",
        "tags": [], "authors": [{"name": "Li", "weight": 0.6}],
        "doc_types": [{"name": "Papers", "weight": 1.0}],  # e4b-qat junk facet
        "year_from": 0, "year_to": 0,
    }])
    parsed = parse_query(fake, "papers by Li about water ice on the moon")
    assert [f.kind for f in parsed.facets] == ["author"]


def test_parse_failure_degrades_to_plain(data_dir):
    parsed = parse_query(FakeClient([None]), "some query")
    assert parsed.semantic_query == "some query"
    assert parsed.facets == []


def test_rerank_reorders_and_survives_garbage():
    results = [result(title=f"doc{i}", score=1.0 - i / 10) for i in range(5)]
    fake = FakeClient([{"ranking": [3, 0, 99, 3, 1]}])  # dupe + out-of-range
    reranked = rerank(fake, "q", results)
    assert [r.title for r in reranked] == ["doc3", "doc0", "doc1", "doc2", "doc4"]

    # failed call keeps original order
    assert [r.title for r in rerank(FakeClient([None]), "q", results)] == [
        f"doc{i}" for i in range(5)
    ]

    # single result: no call needed
    assert rerank(FakeClient([]), "q", results[:1]) == results[:1]
