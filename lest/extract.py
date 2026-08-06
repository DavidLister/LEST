from pathlib import Path

import pymupdf

# Corrupt PDFs in the wild make MuPDF spam stderr; errors surface as exceptions instead.
pymupdf.TOOLS.mupdf_display_errors(False)

TEXT_SUFFIXES = {".txt", ".md"}
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | {".pdf"}


def extract(path: Path, content_type: str | None = None) -> str | None:
    """Return the file's plain text, or None if it contains no extractable text.

    Exceptions (unreadable/corrupt files) propagate; callers record them per-file.
    """
    if content_type == "application/pdf" or path.suffix.lower() == ".pdf":
        return _extract_pdf(path)
    if path.suffix.lower() in TEXT_SUFFIXES:
        text = path.read_text(errors="replace")
        return text if text.strip() else None
    return None


def _extract_pdf(path: Path) -> str | None:
    with pymupdf.open(path) as doc:
        text = "\n\n".join(page.get_text() for page in doc)
    return text if text.strip() else None
