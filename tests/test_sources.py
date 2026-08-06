from lest.sources import PlainDirSource, make_source


def test_plaindir_walk(tmp_path):
    (tmp_path / "a.txt").write_text("alpha")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.md").write_text("beta")
    (tmp_path / "sub" / "c.docx").write_text("unsupported")
    (tmp_path / ".hidden").mkdir()
    (tmp_path / ".hidden" / "d.txt").write_text("hidden")

    documents = list(PlainDirSource(tmp_path).documents())
    keys = [d.key for d in documents]
    assert keys == ["a.txt", "sub/b.md"]
    assert documents[0].title == "a.txt"
    assert documents[0].attachments[0].path == tmp_path / "a.txt"
    assert ":" in documents[0].attachments[0].fingerprint


def test_fingerprint_changes_with_content(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("one")
    (fp1,) = [d.attachments[0].fingerprint for d in PlainDirSource(tmp_path).documents()]
    f.write_text("two-longer")
    (fp2,) = [d.attachments[0].fingerprint for d in PlainDirSource(tmp_path).documents()]
    assert fp1 != fp2


def test_make_source_auto_detects(tmp_path):
    assert isinstance(make_source(tmp_path, "auto"), PlainDirSource)
    assert isinstance(make_source(tmp_path, "plaindir"), PlainDirSource)
