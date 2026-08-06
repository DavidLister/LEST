from pathlib import Path

from .base import Attachment, Document, Source, fingerprint
from .plaindir import PlainDirSource

__all__ = ["Attachment", "Document", "Source", "fingerprint", "PlainDirSource", "make_source"]


def make_source(root: Path, kind: str) -> Source:
    if kind == "auto":
        kind = "zotero" if (root / "zotero.sqlite").exists() else "plaindir"
    if kind == "zotero":
        from .zotero import ZoteroSource

        return ZoteroSource(root)
    return PlainDirSource(root)
