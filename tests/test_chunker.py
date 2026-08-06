from lest.chunkers import get_chunker
from lest.chunkers.paragraph import MAX_CHARS, MIN_CHARS


def chunk(text):
    return get_chunker("paragraph").chunk(text)


def test_empty_text():
    assert chunk("") == []
    assert chunk("\n\n  \n\n") == []


def test_short_text_single_chunk():
    assert chunk("Just one small paragraph.") == ["Just one small paragraph."]


def test_paragraphs_merge_to_min_size():
    paragraphs = [f"Paragraph number {i} with a little bit of text in it." for i in range(30)]
    chunks = chunk("\n\n".join(paragraphs))
    assert len(chunks) > 1
    # all but the last chunk reached the minimum size
    assert all(len(c) >= MIN_CHARS for c in chunks[:-1])
    # nothing was lost
    assert "".join(chunks).replace("\n\n", "") == "".join(paragraphs)


def test_giant_paragraph_split_at_sentences():
    text = " ".join(f"This is sentence number {i} of the giant paragraph." for i in range(200))
    chunks = chunk(text)
    assert len(chunks) > 1
    assert all(len(c) <= MAX_CHARS for c in chunks)
    assert all(c.rstrip().endswith(".") for c in chunks)


def test_unbroken_run_hard_split():
    text = "x" * (3 * MAX_CHARS)
    chunks = chunk(text)
    assert all(len(c) <= MAX_CHARS for c in chunks)
    assert sum(len(c) for c in chunks) == 3 * MAX_CHARS


def test_ragged_pdf_style_text():
    text = "Title Line\n\n" + "\n\n".join(["Short."] * 5) + "\n\nA closing fragment"
    chunks = chunk(text)
    assert len(chunks) == 1  # tiny fragments merged together
    assert "Title Line" in chunks[0] and "closing fragment" in chunks[0]
