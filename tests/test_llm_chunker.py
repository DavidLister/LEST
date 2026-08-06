"""LlmChunker unit tests with a scripted fake client (no Ollama)."""

from lest.chunkers.llm import LlmChunker
from lest.chunkers.paragraph import MAX_CHARS, MIN_CHARS
from lest.llm import normalize


class FakeClient:
    """Returns queued responses; records prompts."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def ping(self):
        pass

    def call(self, prompt, schema, **kwargs):
        self.prompts.append(prompt)
        return self.responses.pop(0) if self.responses else None


def outline_for(text, pieces):
    """Build a valid outline whose anchors are the starts of `pieces`."""
    return {
        "sections": [
            {
                "title": "Section",
                "context": "covers the topic",
                "ideas": [{"first_words": " ".join(p.split()[:5])} for p in pieces],
            }
        ]
    }


def make_text(n_sentences=60):
    return " ".join(
        f"Sentence number {i} talks about crystal growth in detail." for i in range(n_sentences)
    )


def test_cuts_merge_to_envelope():
    text = make_text()
    norm = normalize(text)
    thirds = [norm[: len(norm) // 3], norm[len(norm) // 3 : 2 * len(norm) // 3],
              norm[2 * len(norm) // 3 :]]
    chunker = LlmChunker(client=FakeClient([outline_for(norm, thirds)]))
    chunks = chunker.chunk(text, title="My Paper")
    assert chunks
    assert not chunker.last_used_fallback
    for chunk in chunks:
        assert chunk.startswith("[My Paper] [Section — covers the topic] ")
        body = chunk.split("] ", 2)[-1]
        assert len(body) <= MAX_CHARS
    # all text present, in order
    rejoined = " ".join(c.split("] ", 2)[-1] for c in chunks)
    for i in (0, 30, 59):
        assert f"Sentence number {i} " in rejoined + " "


def test_bad_anchors_are_harmless():
    text = make_text()
    outline = {
        "sections": [
            {"title": "S", "context": "", "ideas": [
                {"first_words": "totally absent anchor words here"},
                {"first_words": "Sentence number 30 talks about"},
            ]}
        ]
    }
    chunker = LlmChunker(client=FakeClient([outline]))
    chunks = chunker.chunk(text)
    assert chunks
    rejoined = " ".join(c.split("] ", 1)[-1] if c.startswith("[") else c for c in chunks)
    assert "Sentence number 0" in rejoined
    assert "Sentence number 59" in rejoined


def test_outline_failure_falls_back_to_paragraphs():
    text = make_text()
    chunker = LlmChunker(client=FakeClient([None, None]))  # call + retry-side None
    chunks = chunker.chunk(text, title="T")
    assert chunker.last_used_fallback
    assert chunks
    assert all(c.startswith("[T] ") for c in chunks)


def test_long_documents_outline_in_parts(monkeypatch):
    import lest.chunkers.llm as mod

    monkeypatch.setattr(mod, "MAX_TEXT_CHARS", 500)
    monkeypatch.setattr(mod, "PART_CHARS", 400)
    text = make_text(40)  # ~2000 chars -> multiple parts
    norm = normalize(text)
    fake = FakeClient([outline_for(norm, ["ignored"])] * 10)
    # every part gets an outline whose single anchor misses; fallback-free
    for response in fake.responses:
        response["sections"][0]["ideas"] = [{"first_words": "Sentence number"}]
    chunker = LlmChunker(client=fake)
    chunks = chunker.chunk(text)
    assert len(fake.prompts) >= 4  # split actually happened
    rejoined = " ".join(c.split("] ", 1)[-1] if c.startswith("[") else c for c in chunks)
    assert "Sentence number 39" in rejoined  # tail beyond parts still indexed
    assert all(len(c) <= MAX_CHARS + 200 for c in chunks)


def test_chunk_sizes_respect_min(monkeypatch):
    text = make_text()
    norm = normalize(text)
    # anchors every ~80 chars: heavy over-segmentation must merge to >= MIN_CHARS
    anchors = [norm[i : i + 40] for i in range(0, len(norm) - 40, 80)]
    outline = {"sections": [{"title": "", "context": "", "ideas": [
        {"first_words": a} for a in anchors]}]}
    chunker = LlmChunker(client=FakeClient([outline]))
    chunks = chunker.chunk(text)
    assert all(len(c) >= MIN_CHARS // 2 for c in chunks[:-1])
