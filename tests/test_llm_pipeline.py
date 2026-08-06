"""End-to-end llm-mode indexing with a scripted fake gemma (no Ollama)."""

import pytest

from lest.pipeline import index_directory
from lest.query import search_directory, status_directory


class FakeGemma:
    """Routes calls by schema shape, like the real model would answer."""

    def __init__(self):
        self.calls = []

    def ping(self):
        pass

    def call(self, prompt, schema, images=None, **kwargs):
        properties = schema.get("properties", {})
        self.calls.append(sorted(properties))
        if "sections" in properties:
            # anchor into the fixture text so the cut path is exercised
            return {"sections": [{
                "title": "All", "context": "the whole document",
                "ideas": [{"first_words": prompt.split("PAPER TEXT:")[-1].strip()[:30]}],
            }]}
        if "figures" in properties:
            return {"figures": [{"page": 1, "description": "a plot of growth rate"}]}
        if "notable" in properties:
            return {
                "notable": "crystals racing on glass",
                "main_ideas": "fastest oriented grains dominate the film",
                "methods": "sputtering and electron microscopy",
                "why_cite": "cite for evolutionary selection growth",
            }
        if "tags" in properties:
            return {"tags": ["crystal growth", "thin films"]}
        if "doc_type" in properties:
            return {"doc_type": "research notes"}
        if "choice" in properties:
            return {"choice": "NEW"}
        raise AssertionError(f"unexpected schema: {sorted(properties)}")


@pytest.fixture
def llm_corpus(tmp_path, data_dir, fake_embedder, monkeypatch):
    directory = tmp_path / "corpus"
    directory.mkdir()
    (directory / "growth.txt").write_text(
        "Evolutionary selection growth means the fastest grains win. " * 40
    )
    fake = FakeGemma()
    import lest.chunkers.llm as chunker_mod

    monkeypatch.setattr(chunker_mod, "LlmClient", lambda *a, **k: fake)
    return directory, fake


def test_llm_mode_end_to_end(llm_corpus):
    directory, fake = llm_corpus
    stats = index_directory(
        directory, model="fake-model", embedder_name="fake", chunker_name="llm"
    )
    assert stats.files_indexed == 1
    assert stats.files_fallback == 0

    text = status_directory(directory)
    assert "chunker: llm" in text
    assert "body=" in text and "view=" in text  # .txt source: no figure call

    results = search_directory(directory, "fastest grains", n=5)
    assert results
    top = results[0]
    assert top.doc_type == "research notes"
    assert set(top.tags) == {"crystal growth", "thin films"}
    assert top.meta.get("summary", "").startswith("fastest oriented grains")

    # impression query hits via a view chunk
    impression = search_directory(directory, "crystals racing glass", n=1)
    assert impression[0].kind in ("view", "body")


def test_llm_failure_marks_pending_and_retries(tmp_path, data_dir, fake_embedder, monkeypatch):
    directory = tmp_path / "corpus"
    directory.mkdir()
    (directory / "doc.txt").write_text("Some content that is long enough to chunk. " * 30)

    class ExplodingGemma(FakeGemma):
        def call(self, prompt, schema, **kwargs):
            raise RuntimeError("model exploded")

    fake = ExplodingGemma()
    import lest.chunkers.llm as chunker_mod

    monkeypatch.setattr(chunker_mod, "LlmClient", lambda *a, **k: fake)
    stats = index_directory(
        directory, model="fake-model", embedder_name="fake", chunker_name="llm"
    )
    assert stats.files_skipped == 1
    assert stats.skipped[0][1] == "llm_pending"

    # next run retries the pending file even though the fingerprint matches
    working = FakeGemma()
    monkeypatch.setattr(chunker_mod, "LlmClient", lambda *a, **k: working)
    stats = index_directory(directory, embedder_name="fake")
    assert stats.files_indexed == 1
    assert stats.files_skipped == 0
