import pytest

from lest.catalog import Catalog, fold


@pytest.fixture
def controlled_embed():
    """Embeds known names to fixed 2-d vectors so similarity is scripted."""
    table = {
        "mocvd": [1.0, 0.0],
        "movpe": [0.99, 0.14],  # cos ~0.99 vs mocvd -> auto-map band
        "chemical vapor deposition": [0.85, 0.53],  # cos ~0.85 -> gray band
        "photoluminescence": [0.0, 1.0],  # distinct
    }

    def embed(texts):
        return [table.get(t, [0.5, -0.5]) for t in texts]

    return embed


@pytest.fixture
def cat(tmp_path, controlled_embed):
    c = Catalog(path=tmp_path / "catalog.db", embed=controlled_embed)
    yield c
    c.close()


def test_exact_and_alias_resolution(cat):
    assert cat.resolve_term("tag", "MOCVD") == "mocvd"
    assert cat.resolve_term("tag", "mocvd") == "mocvd"
    assert cat.counts("tag") == [("mocvd", 2)]


def test_high_similarity_auto_maps(cat):
    cat.resolve_term("tag", "mocvd")
    assert cat.resolve_term("tag", "movpe") == "mocvd"
    assert "movpe" in cat.aliases_of("tag", "mocvd")
    # the alias now resolves directly
    assert cat.resolve_term("tag", "movpe") == "mocvd"
    assert cat.search_terms("tag", "mocvd") == {"mocvd", "movpe"}


def test_gray_zone_adjudication(cat):
    cat.resolve_term("tag", "mocvd")
    calls = []

    def adjudicate_map(kind, proposed, candidates):
        calls.append((kind, proposed, candidates))
        return candidates[0]

    assert cat.resolve_term("tag", "chemical vapor deposition", adjudicate_map) == "mocvd"
    assert calls and calls[0][1] == "chemical vapor deposition"

    def adjudicate_new(kind, proposed, candidates):
        return None

    cat2_name = cat.resolve_term("tag", "photoluminescence", adjudicate_new)
    assert cat2_name == "photoluminescence"
    assert set(cat.names("tag")) == {"mocvd", "photoluminescence"}


def test_distinct_term_created_without_adjudicator(cat):
    cat.resolve_term("tag", "mocvd")
    assert cat.resolve_term("tag", "photoluminescence") == "photoluminescence"


def test_author_initials_merge_prefers_complete_name(cat):
    assert cat.resolve_author("Neugebauer, J.") == "Neugebauer, J."
    canonical = cat.resolve_author("Neugebauer, Jörg")
    assert canonical == "Neugebauer, Jörg"  # more complete spelling wins
    assert cat.names("author") == ["Neugebauer, Jörg"]
    # and the short form still resolves
    assert cat.resolve_author("Neugebauer, J.") == "Neugebauer, Jörg"


def test_author_incompatible_initials_stay_distinct(cat):
    cat.resolve_author("Smith, John")
    assert cat.resolve_author("Smith, Karen") == "Smith, Karen"
    assert len(cat.names("author")) == 2


def test_author_typo_queues_merge_proposal(cat):
    cat.resolve_author("Neugebauer, Jörg")
    created = cat.resolve_author("Neugebaur, Jörg")  # typo surname
    assert created == "Neugebaur, Jörg"  # kept distinct, never auto-merged
    pending = cat.pending_merges()
    assert len(pending) == 1
    _, kind, keep, drop, rationale = pending[0]
    assert kind == "author"
    assert {keep, drop} == {"Neugebauer, Jörg", "Neugebaur, Jörg"}
    assert "similarity" in rationale


def test_apply_merge_moves_usage_and_aliases(cat):
    cat.resolve_author("Neugebauer, Jörg")
    cat.resolve_author("Neugebaur, Jörg")
    merge_id = cat.pending_merges()[0][0]
    cat.apply_merge(merge_id, approve=True)
    assert cat.names("author") == ["Neugebauer, Jörg"]
    assert "Neugebaur, Jörg" in cat.aliases_of("author", "Neugebauer, Jörg")
    assert cat.pending_merges() == []
    # alias-aware search group includes the merged spelling
    assert "Neugebaur, Jörg" in cat.search_terms("author", "Neugebauer, Jörg")


def test_reject_merge_keeps_both(cat):
    cat.resolve_author("Neugebauer, Jörg")
    cat.resolve_author("Neugebaur, Jörg")
    merge_id = cat.pending_merges()[0][0]
    cat.apply_merge(merge_id, approve=False)
    assert len(cat.names("author")) == 2


def test_lookup_never_creates(cat):
    assert cat.lookup_term("tag", "nonexistent") is None
    assert cat.lookup_author("Nobody, Nowhere") is None
    assert cat.names("tag") == []
    assert cat.names("author") == []


def test_lookup_author_fuzzy(cat):
    cat.resolve_author("Neugebauer, Jörg")
    assert cat.lookup_author("Jörg Neugebauer") is not None  # First Last order
    assert cat.lookup_author("neugebauer, j") is not None  # initials
    assert cat.lookup_author("Neugebaur, Jörg") is not None  # typo via ratio


def test_seed_tags(cat, tmp_path):
    vocab = tmp_path / "vocab.txt"
    vocab.write_text(
        "# comment only\n"
        "## materials\n"
        "gallium nitride            # = gan (~345 items)\n"
        "mocvd                      # = mcvd, movpe (~250!)\n"
        "plain-tag\n"
    )
    added = cat.seed_tags(vocab)
    assert added == 3
    assert set(cat.names("tag")) == {"gallium nitride", "mocvd", "plain-tag"}
    assert "gan" in cat.aliases_of("tag", "gallium nitride")
    assert {"mcvd", "movpe"} <= set(cat.aliases_of("tag", "mocvd"))
    assert cat.seed_tags(vocab) == 0  # idempotent


def test_fold():
    assert fold("  Jörg   NEUGEBAUER ") == "jorg neugebauer"
