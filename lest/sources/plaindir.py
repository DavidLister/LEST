from collections.abc import Iterator
from pathlib import Path

from ..extract import SUPPORTED_SUFFIXES
from .base import Attachment, Document, fingerprint


class PlainDirSource:
    """Every supported file in the tree is its own document. Hidden dirs/files skipped."""

    def __init__(self, root: Path):
        self.root = root

    def documents(self) -> Iterator[Document]:
        for path in sorted(self.root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue
            relative = path.relative_to(self.root)
            if any(part.startswith(".") for part in relative.parts):
                continue
            yield Document(
                key=relative.as_posix(),
                title=path.name,
                attachments=[Attachment(path=path, fingerprint=fingerprint(path))],
            )
