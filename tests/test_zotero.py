import sqlite3

import pymupdf
import pytest

from lest.sources import make_source
from lest.sources.zotero import ZoteroSource

_DDL = """
CREATE TABLE itemTypes (itemTypeID INTEGER PRIMARY KEY, typeName TEXT);
CREATE TABLE items (itemID INTEGER PRIMARY KEY, itemTypeID INT, key TEXT, libraryID INT);
CREATE TABLE fields (fieldID INTEGER PRIMARY KEY, fieldName TEXT);
CREATE TABLE itemDataValues (valueID INTEGER PRIMARY KEY, value TEXT);
CREATE TABLE itemData (itemID INT, fieldID INT, valueID INT);
CREATE TABLE creators (creatorID INTEGER PRIMARY KEY, firstName TEXT, lastName TEXT,
                       fieldMode INT);
CREATE TABLE itemCreators (itemID INT, creatorID INT, creatorTypeID INT, orderIndex INT);
CREATE TABLE itemAttachments (itemID INT, parentItemID INT, linkMode INT, contentType TEXT,
                              path TEXT);
CREATE TABLE deletedItems (itemID INT, dateDeleted TEXT);
"""

TYPES = {"journalArticle": 1, "attachment": 2, "note": 3, "annotation": 4}
FIELDS = {"title": 1, "date": 2, "DOI": 3}


def make_zotero_fixture(root):
    """A tiny Zotero data dir covering every adapter edge the plan calls for."""
    root.mkdir()
    conn = sqlite3.connect(root / "zotero.sqlite")
    conn.executescript(_DDL)
    conn.executemany("INSERT INTO itemTypes VALUES (?, ?)", [(v, k) for k, v in TYPES.items()])
    conn.executemany("INSERT INTO fields VALUES (?, ?)", [(v, k) for k, v in FIELDS.items()])

    def add_item(item_id, type_name, key, **field_values):
        conn.execute("INSERT INTO items VALUES (?, ?, ?, 1)", (item_id, TYPES[type_name], key))
        for field, value in field_values.items():
            (value_id,) = conn.execute(
                "INSERT INTO itemDataValues (value) VALUES (?) RETURNING valueID", (value,)
            ).fetchone()
            conn.execute("INSERT INTO itemData VALUES (?, ?, ?)", (item_id, FIELDS[field],
                                                                   value_id))

    def add_pdf(att_id, parent_id, key, filename, link_mode=0, on_disk=True, title=None):
        add_item(att_id, "attachment", key, **({"title": title} if title else {}))
        conn.execute(
            "INSERT INTO itemAttachments VALUES (?, ?, ?, 'application/pdf', ?)",
            (att_id, parent_id, link_mode,
             f"storage:{filename}" if link_mode in (0, 1) else f"E:\\old\\{filename}"),
        )
        if on_disk and link_mode in (0, 1):
            directory = root / "storage" / key
            directory.mkdir(parents=True)
            doc = pymupdf.open()
            doc.new_page().insert_text((72, 72), f"Body text of {filename}.")
            doc.save(directory / filename)
            doc.close()

    # item 1: two on-disk PDFs, two authors (one duplicated across roles)
    add_item(1, "journalArticle", "ITEM0001", title="Mass Transfer in Waveguides",
             date="2010-00-00 2010", DOI="10.1000/mtw")
    conn.execute("INSERT INTO creators VALUES (1, 'Ada', 'Lovelace', 0)")
    conn.execute("INSERT INTO creators VALUES (2, NULL, 'MIT Media Lab', 1)")
    conn.executemany("INSERT INTO itemCreators VALUES (?, ?, ?, ?)",
                     [(1, 1, 1, 0), (1, 2, 2, 1), (1, 1, 3, 2)])  # Lovelace twice
    add_pdf(10, 1, "ATTACH01", "main.pdf", link_mode=0)
    add_pdf(11, 1, "ATTACH02", "supplement.pdf", link_mode=1)

    # item 2: attachment recorded but file missing on disk
    add_item(2, "journalArticle", "ITEM0002", title="Unsynced Paper")
    add_pdf(12, 2, "ATTACH03", "gone.pdf", on_disk=False)

    # item 3: only a linked file (linkMode 2) — excluded, so no attachments
    add_item(3, "journalArticle", "ITEM0003", title="Linked Only")
    add_pdf(13, 3, "ATTACH04", "linked.pdf", link_mode=2)

    # item 4: deleted (in trash), would otherwise qualify
    add_item(4, "journalArticle", "ITEM0004", title="Deleted Paper")
    add_pdf(14, 4, "ATTACH05", "deleted.pdf")
    conn.executemany("INSERT INTO deletedItems VALUES (?, '2026-01-01')", [(4,), (14,)])

    # note and annotation items: never documents
    add_item(5, "note", "NOTE0001")
    add_item(6, "annotation", "ANNO0001")

    # standalone attachment (no parent), with a title
    add_pdf(15, None, "ATTACH06", "standalone.pdf", title="A Standalone Report")

    conn.commit()
    conn.close()
    return root


@pytest.fixture
def zotero_dir(tmp_path):
    return make_zotero_fixture(tmp_path / "zotero")


def test_auto_detection(zotero_dir):
    assert isinstance(make_source(zotero_dir, "auto"), ZoteroSource)


def test_documents_and_attachments(zotero_dir):
    documents = {d.key: d for d in ZoteroSource(zotero_dir).documents()}

    # regular items with no on-disk files still appear (pipeline skips them)
    assert set(documents) == {"ITEM0001", "ITEM0002", "ITEM0003", "ATTACH06"}

    item1 = documents["ITEM0001"]
    assert item1.title == "Mass Transfer in Waveguides"
    assert item1.meta["year"] == "2010"
    assert item1.meta["doi"] == "10.1000/mtw"
    assert item1.meta["creators"] == "Lovelace, Ada; MIT Media Lab"  # deduped, fieldMode=1 ok
    names = [a.path.name for a in item1.attachments]
    assert names == ["main.pdf", "supplement.pdf"]  # linkMode 0 and 1 both included
    assert all(a.path.is_file() and a.content_type == "application/pdf"
               for a in item1.attachments)
    assert item1.attachments[0].path == zotero_dir / "storage" / "ATTACH01" / "main.pdf"

    assert documents["ITEM0002"].attachments == []  # missing file skipped
    assert documents["ITEM0003"].attachments == []  # linked file excluded

    standalone = documents["ATTACH06"]
    assert standalone.title == "A Standalone Report"
    assert [a.path.name for a in standalone.attachments] == ["standalone.pdf"]
    assert standalone.meta["type"] == "attachment"


def test_full_pipeline_over_zotero(zotero_dir, data_dir, fake_embedder):
    from lest.pipeline import index_directory
    from lest.query import search_directory

    stats = index_directory(zotero_dir, model="fake-model", embedder_name="fake")
    # ITEM0001 (2 files) + ATTACH06; empty-attachment documents skipped
    assert stats.documents == 2
    assert stats.files_indexed == 3

    results = search_directory(zotero_dir, "body text of main", n=5)
    assert results[0].title == "Mass Transfer in Waveguides"
    assert len(results[0].paths) == 2  # both PDFs listed for the rofi wrapper
