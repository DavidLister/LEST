from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass
class Attachment:
    path: Path
    fingerprint: str
    content_type: str | None = None


@dataclass
class Document:
    key: str  # stable ID within the source (Zotero item key, or relative path)
    title: str
    attachments: list[Attachment] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


class Source(Protocol):
    def documents(self) -> Iterator[Document]: ...


def fingerprint(path: Path) -> str:
    st = path.stat()
    return f"{st.st_mtime_ns}:{st.st_size}"
