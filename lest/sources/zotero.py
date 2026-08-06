"""Read a Zotero data directory: one document per bibliographic item, with its
stored PDF attachments; standalone PDF attachments become their own documents.

The database is opened strictly read-only (Zotero may be running).
"""

import logging
import re
import sqlite3
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

from ..errors import EnvironmentError_
from .base import Attachment, Document, fingerprint

log = logging.getLogger(__name__)

PDF = "application/pdf"

# One row per regular bibliographic item (attachments, notes, and PDF
# annotations are not items to index; deleted items are in the trash).
# Field values are joined by fieldName — field IDs vary between profiles.
_ITEMS_SQL = """
SELECT
  i.itemID,
  i.key,
  it.typeName,
  (SELECT idv.value FROM itemData id JOIN itemDataValues idv USING (valueID)
     JOIN fields f USING (fieldID)
    WHERE id.itemID = i.itemID AND f.fieldName = 'title')        AS title,
  (SELECT idv.value FROM itemData id JOIN itemDataValues idv USING (valueID)
     JOIN fields f USING (fieldID)
    WHERE id.itemID = i.itemID AND f.fieldName = 'date')         AS date,
  (SELECT idv.value FROM itemData id JOIN itemDataValues idv USING (valueID)
     JOIN fields f USING (fieldID)
    WHERE id.itemID = i.itemID AND f.fieldName = 'DOI')          AS doi
FROM items i
JOIN itemTypes it USING (itemTypeID)
WHERE it.typeName NOT IN ('attachment', 'note', 'annotation')
  AND i.itemID NOT IN (SELECT itemID FROM deletedItems)
ORDER BY i.itemID
"""

_CREATORS_SQL = """
SELECT ic.itemID,
       CASE c.fieldMode WHEN 1 THEN c.lastName
            ELSE c.lastName || ', ' || c.firstName END AS name
FROM itemCreators ic
JOIN creators c USING (creatorID)
ORDER BY ic.itemID, ic.orderIndex
"""

# Stored PDF attachments: linkMode 0 = imported_file, 1 = imported_url (both
# live under storage/<attachment key>/). linkMode 2/3 (linked file/URL) are
# excluded. contentType is trusted over the file extension.
_ATTACHMENTS_SQL = """
SELECT ia.parentItemID, att.key,
       (SELECT idv.value FROM itemData id JOIN itemDataValues idv USING (valueID)
          JOIN fields f USING (fieldID)
         WHERE id.itemID = att.itemID AND f.fieldName = 'title') AS title,
       ia.path
FROM itemAttachments ia
JOIN items att ON att.itemID = ia.itemID
WHERE ia.contentType = 'application/pdf'
  AND ia.linkMode IN (0, 1)
  AND ia.path LIKE 'storage:%'
  AND att.itemID NOT IN (SELECT itemID FROM deletedItems)
ORDER BY att.itemID
"""


class ZoteroSource:
    def __init__(self, root: Path):
        self.root = root
        self.db_path = root / "zotero.sqlite"

    def documents(self) -> Iterator[Document]:
        conn = self._connect()
        try:
            attached, standalone = self._attachments(conn)
            creators = self._creators(conn)

            for item_id, key, type_name, title, date, doi in conn.execute(_ITEMS_SQL):
                meta = {"type": type_name}
                if creators.get(item_id):
                    meta["creators"] = "; ".join(creators[item_id])
                year = _year(date)
                if year:
                    meta["year"] = year
                if doi:
                    meta["doi"] = doi
                yield Document(
                    key=key,
                    title=title or f"[untitled {type_name} {key}]",
                    attachments=attached.get(item_id, []),
                    meta=meta,
                )

            for key, title, attachment in standalone:
                yield Document(
                    key=key,
                    title=title or attachment.path.name,
                    attachments=[attachment],
                    meta={"type": "attachment"},
                )
        except sqlite3.OperationalError as exc:
            raise EnvironmentError_(
                f"cannot read {self.db_path} ({exc}) — close Zotero or retry"
            ) from exc
        finally:
            conn.close()

    def _connect(self) -> sqlite3.Connection:
        try:
            conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
            conn.execute("PRAGMA busy_timeout = 5000")
            conn.execute("SELECT count(*) FROM items").fetchone()  # probe lock state early
        except sqlite3.OperationalError as exc:
            raise EnvironmentError_(
                f"cannot open {self.db_path} read-only ({exc}) — close Zotero or retry"
            ) from exc
        return conn

    def _creators(self, conn) -> dict[int, list[str]]:
        creators: dict[int, list[str]] = defaultdict(list)
        for item_id, name in conn.execute(_CREATORS_SQL):
            if name and name not in creators[item_id]:  # same person can hold two roles
                creators[item_id].append(name)
        return creators

    def _attachments(self, conn):
        """Returns ({parent item id: [Attachment]}, [(key, title, Attachment)])."""
        attached: dict[int, list[Attachment]] = defaultdict(list)
        standalone: list[tuple[str, str | None, Attachment]] = []
        missing = 0
        for parent_id, key, title, stored_path in conn.execute(_ATTACHMENTS_SQL):
            path = self.root / "storage" / key / stored_path[len("storage:") :]
            if not path.is_file():
                missing += 1
                log.debug("attachment file missing (not synced?): %s", path)
                continue
            attachment = Attachment(
                path=path, fingerprint=fingerprint(path), content_type=PDF
            )
            if parent_id is None:
                standalone.append((key, title, attachment))
            else:
                attached[parent_id].append(attachment)
        if missing:
            log.warning("%d attachment files missing on disk (not synced?) — skipped", missing)
        return attached, standalone


def _year(date: str | None) -> str | None:
    """Zotero dates look like '2010-07-11 2010-07-11' or '2010-00-00 2010' — never ISO."""
    match = re.search(r"\b(\d{4})\b", date or "")
    return match.group(1) if match else None
