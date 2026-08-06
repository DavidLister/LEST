import pymupdf
import pytest

from lest.extract import extract


def make_pdf(path, texts):
    doc = pymupdf.open()
    for text in texts:
        page = doc.new_page()
        if text:
            page.insert_text((72, 72), text)
    doc.save(path)
    doc.close()


def test_pdf_text(tmp_path):
    pdf = tmp_path / "paper.pdf"
    make_pdf(pdf, ["First page text.", "Second page text."])
    text = extract(pdf)
    assert "First page text." in text
    assert "Second page text." in text
    assert text.index("First") < text.index("Second")


def test_pdf_without_text_layer(tmp_path):
    pdf = tmp_path / "scanned.pdf"
    make_pdf(pdf, [""])  # one blank page => no text layer
    assert extract(pdf) is None


def test_pdf_by_content_type_despite_extension(tmp_path):
    odd = tmp_path / "ie00003a036"  # real Zotero case: PDF without .pdf suffix
    make_pdf(odd, ["Hidden pdf content."])
    assert "Hidden pdf content." in extract(odd, content_type="application/pdf")


def test_corrupt_pdf_raises(tmp_path):
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"%PDF-not really a pdf")
    with pytest.raises(pymupdf.FileDataError):
        extract(bad)


def test_text_files(tmp_path):
    txt = tmp_path / "notes.txt"
    txt.write_text("plain text content")
    assert extract(txt) == "plain text content"
    empty = tmp_path / "empty.md"
    empty.write_text("   \n")
    assert extract(empty) is None


def test_unknown_suffix(tmp_path):
    other = tmp_path / "data.csv"
    other.write_text("a,b,c")
    assert extract(other) is None
