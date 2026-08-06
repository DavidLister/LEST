"""Full pipeline end-to-end with the deterministic fake embedder (no Ollama needed)."""

import json
import shutil
from pathlib import Path

import pytest

from lest.errors import LestError
from lest.output import format_json, format_tsv
from lest.pipeline import index_directory
from lest.query import search_directory, status_directory

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def corpus(tmp_path, data_dir, fake_embedder):
    directory = tmp_path / "corpus"
    shutil.copytree(FIXTURES, directory)
    return directory


def index(directory, **kwargs):
    kwargs.setdefault("model", "fake-model")
    kwargs.setdefault("embedder_name", "fake")
    return index_directory(directory, **kwargs)


def test_index_and_search(corpus):
    stats = index(corpus)
    assert stats.documents == 3
    assert stats.files_indexed == 3
    assert stats.chunks >= 3

    results = search_directory(corpus, "telescope observing galaxies and nebulae", n=2)
    assert results[0].title == "astronomy.txt"
    assert results[0].score > results[-1].score

    line = format_tsv(results[0])
    score, title, paths = line.split("\t")
    assert title == "astronomy.txt"
    assert float(score) > 0
    assert paths.endswith("astronomy.txt")

    payload = json.loads(format_json(results[0]))
    assert payload["title"] == "astronomy.txt"
    assert "telescope" in payload["best_chunk"].lower()


def test_agg_strategies_run(corpus):
    index(corpus)
    for agg in ("max", "topk:2", "softmax:0.05", "decay", "count:0.1"):
        results = search_directory(corpus, "sourdough yeast fermentation", n=3, agg_spec=agg)
        assert results, agg
        assert results[0].title == "baking.txt", agg


def test_first_index_requires_model(corpus):
    with pytest.raises(LestError, match="--model"):
        index_directory(corpus, embedder_name="fake")


def test_model_mismatch_rejected(corpus):
    index(corpus)
    with pytest.raises(LestError, match="refusing to mix"):
        index(corpus, model="other-model")
    index(corpus)  # same model: fine
    index(corpus, model="fake-model")  # explicit same model: fine


def test_incremental_sync(corpus):
    index(corpus)

    # untouched re-run: nothing re-embedded
    stats = index(corpus)
    assert (stats.files_indexed, stats.files_unchanged, stats.files_removed) == (0, 3, 0)

    # modified file re-embeds only itself
    sailing = corpus / "sailing.txt"
    sailing.write_text(sailing.read_text() + "\n\nSpinnakers billow on a downwind run.\n")
    stats = index(corpus)
    assert (stats.files_indexed, stats.files_unchanged) == (1, 2)
    results = search_directory(corpus, "spinnakers billow downwind", n=1)
    assert results[0].title == "sailing.txt"

    # deleted file disappears from results
    sailing.unlink()
    stats = index(corpus)
    assert stats.files_removed == 1
    titles = {r.title for r in search_directory(corpus, "sailboat tacks upwind", n=10)}
    assert "sailing.txt" not in titles


def test_new_file_added_incrementally(corpus):
    index(corpus)
    (corpus / "geology.txt").write_text("Basalt forms when lava cools quickly at the surface.")
    stats = index(corpus)
    assert (stats.files_indexed, stats.files_unchanged) == (1, 3)
    results = search_directory(corpus, "basalt lava cools", n=1)
    assert results[0].title == "geology.txt"


def test_search_unindexed_directory_errors(tmp_path, data_dir):
    with pytest.raises(LestError, match="no index"):
        search_directory(tmp_path, "anything")


def test_status_output(corpus):
    index(corpus)
    text = status_directory(corpus)
    assert "model: fake-model" in text
    assert "chunker: paragraph" in text
    assert "documents: 3" in text
    assert "last_indexed:" in text
