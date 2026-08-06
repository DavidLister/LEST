import hashlib
import logging
from pathlib import Path

import pymupdf

from .store import data_dir

log = logging.getLogger(__name__)

# Corrupt PDFs in the wild make MuPDF spam stderr; errors surface as exceptions instead.
pymupdf.TOOLS.mupdf_display_errors(False)

TEXT_SUFFIXES = {".txt", ".md"}
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | {".pdf"}


def extract(path: Path, content_type: str | None = None) -> str | None:
    """Return the file's plain text, or None if it contains no extractable text.

    Scanned PDFs (no text layer) fall back to an OCR'd sidecar copy in
    <data dir>/ocr/ when one exists (see `lest ocr`); source files are never
    modified. Exceptions (unreadable/corrupt files) propagate; callers record
    them per-file.
    """
    if content_type == "application/pdf" or path.suffix.lower() == ".pdf":
        text = _extract_pdf(path)
        if text is None:
            sidecar = ocr_sidecar(path)
            if sidecar.exists():
                log.debug("using OCR sidecar for %s", path)
                return _extract_pdf(sidecar)
        return text
    if path.suffix.lower() in TEXT_SUFFIXES:
        text = path.read_text(errors="replace")
        return text if text.strip() else None
    return None


def content_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()[:16]


def ocr_sidecar(path: Path) -> Path:
    """Sidecar location for a scanned PDF, keyed by content hash so renames
    and Zotero re-syncs keep their OCR."""
    return data_dir() / "ocr" / f"{content_sha(path)}.pdf"


def _extract_pdf(path: Path) -> str | None:
    with pymupdf.open(path) as doc:
        text = "\n\n".join(page.get_text() for page in doc)
    return text if text.strip() else None
