"""Search-side behavior: hybrid fusion, facet filters, duplicate collapse."""

import pytest

from lest.catalog import Catalog
from lest.query import search_directory
from lest.store import Store, db_path_for


@pytest.fixture
def corpus(tmp_path, data_dir, fake_embedder):
    """Two near-duplicate docs + one acronym-only doc, indexed by hand so the
    chunk contents are exactly controlled."""
    directory = tmp_path / "corpus"
    directory.mkdir()
    from conftest import FakeEmbedder

    emb = FakeEmbedder("fake-model")
    store = Store(db_path_for(directory), create=True)
    store.set_meta("model", "fake-model")
    store.set_meta("embedder", "fake")

    def add(key, title, chunks, meta=None, tags=(), authors=(), doc_type=None):
        doc_id = store.upsert_document(key, title, meta or {})
        vectors = emb.embed([t for _, t in chunks])
        store.ensure_vector_table(len(vectors[0]))
        store.replace_file(doc_id, f"/tmp/{key}.pdf", "1:1", "ok", chunks, vectors)
        if tags:
            store.set_doc_tags(doc_id, list(tags))
        if authors:
            store.set_doc_authors(doc_id, list(authors))
        if doc_type:
            store.set_doc_type(doc_id, doc_type)
        return doc_id

    add("dupA", "Growth of GaN films", [("body", "gallium nitride epitaxy on sapphire")],
        meta={"year": "2001"}, tags=("gallium nitride",), authors=("Neugebauer, Jörg",),
        doc_type="research article")
    add("dupB", "Growth of GaN Films", [("body", "gallium nitride epitaxy on sapphire wafers")],
        meta={"year": "2001"})
    add("acro", "Deposition methods survey",
        [("body", "MOVPE reactors use showerhead injection designs")],
        meta={"year": "2010"}, doc_type="review")

    store.close()
    return directory


def test_dedup_collapses_twin_entries(corpus):
    results = search_directory(corpus, "gallium nitride epitaxy sapphire", n=10)
    titles = [r.title.lower() for r in results]
    assert titles.count("growth of gan films") == 1
    top = next(r for r in results if r.title.lower() == "growth of gan films")
    assert top.duplicate_keys  # the twin is recorded
    assert len(top.paths) == 2

    undeduped = search_directory(
        corpus, "gallium nitride epitaxy sapphire", n=10, dedup=False
    )
    assert [r.title.lower() for r in undeduped].count("growth of gan films") == 2


def test_hybrid_search_runs_and_ranks(corpus):
    results = search_directory(corpus, "MOVPE showerhead", n=3)
    assert results[0].key == "acro"
    # vector-only path still works
    assert search_directory(corpus, "MOVPE showerhead", n=3, hybrid=False)


def test_rrf_fusion_prefers_double_hits():
    from lest.query import _fuse
    from lest.store import ChunkHit

    def hit(chunk_id, similarity):
        return ChunkHit(
            similarity=similarity, chunk_id=chunk_id, chunk_text=f"c{chunk_id}",
            kind="body", document_id=1, document_key="k", title="t", meta_json="{}",
        )

    vec_hits = [hit(1, 0.9), hit(2, 0.8), hit(3, 0.7)]
    fts_hits = [hit(2, 5.0), hit(4, 4.0)]
    fused = _fuse(vec_hits, fts_hits)
    order = [h.chunk_id for h in fused]
    assert order[0] == 2  # present in both lists beats either single #1
    assert set(order) == {1, 2, 3, 4}
    # rank-based: fused scores are reciprocal-rank sums, not cosines
    assert fused[0].similarity == pytest.approx(1 / 62 + 1 / 61)


def test_facet_filters_via_catalog(corpus, tmp_path):
    cat = Catalog()  # lives in LEST_DATA_DIR (data_dir fixture)
    cat.resolve_term("tag", "gallium nitride")
    cat.add_alias("tag", "gallium nitride", "gan")
    cat.resolve_author("Neugebauer, Jörg")
    cat.close()

    results = search_directory(corpus, "epitaxy", n=10, tags=["GaN"])
    assert {r.key for r in results} == {"dupA"}

    results = search_directory(corpus, "epitaxy", n=10, authors=["neugebauer, j"])
    assert {r.key for r in results} == {"dupA"}

    results = search_directory(corpus, "deposition", n=10, doc_type="review")
    assert {r.key for r in results} == {"acro"}

    results = search_directory(corpus, "epitaxy", n=10, tags=["nonexistent tag"])
    assert results == []
