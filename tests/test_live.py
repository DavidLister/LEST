"""Live (as-you-type) mode: block protocol, drain-to-newest, error survival."""

import shutil
from pathlib import Path

import pytest

from lest.live import _drain_to_newest, live_loop
from lest.pipeline import index_directory

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def corpus(tmp_path, data_dir, fake_embedder):
    directory = tmp_path / "corpus"
    shutil.copytree(FIXTURES, directory)
    index_directory(directory, model="fake-model", embedder_name="fake",
                    chunker_name="paragraph")
    return directory


def test_drain_returns_newest_when_backlogged(tmp_path):
    read_fd = tmp_path / "input"
    read_fd.write_text("stale one\nstale two\nnewest\n")
    with read_fd.open() as stream:
        assert _drain_to_newest(stream) == "newest"
        assert _drain_to_newest(stream) is None  # EOF


def test_live_blocks_and_errors(corpus, capsys, tmp_path):
    feed = tmp_path / "queries"
    feed.write_text("telescope galaxies\n")
    with feed.open() as stream:
        live_loop(corpus, n=2, stream=stream)
    out = capsys.readouterr().out
    block = out.split("\n\n")[0]
    assert "astronomy.txt" in block
    assert out.endswith("\n\n") or out.endswith("\n")

    # empty query -> empty block; loop survives and exits on EOF
    feed.write_text("\n")
    with feed.open() as stream:
        live_loop(corpus, stream=stream)
    assert capsys.readouterr().out == "\n"
