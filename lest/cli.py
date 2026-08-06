import logging
import sys
from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

from .errors import LestError

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="LEST — semantic search over a directory using local models.",
)
catalog_app = typer.Typer(no_args_is_help=True, help="Inspect and curate the entity catalog.")
app.add_typer(catalog_app, name="catalog")

# TODO(remove-after-llm-migration): the --db flag (and the schema-v1 read
# compatibility it exists for) is temporary A/B plumbing while the paragraph
# baseline and the LLM index coexist. Remove both together.
DB_HELP = "DB directory override (temporary: A/B between baseline and LLM index)."


class SourceKind(StrEnum):
    auto = "auto"
    zotero = "zotero"
    plaindir = "plaindir"


class GpuMode(StrEnum):
    both = "both"
    a2000 = "a2000"


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    for noisy in ("httpx", "httpcore"):  # HTTP client internals drown out -v output
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _apply_gpu_mode(gpu_mode: GpuMode | None) -> None:
    if gpu_mode is not None:
        import os

        os.environ["LEST_GPU_MODE"] = gpu_mode.value


def _parse_stop_at(stop_at: str | None) -> float | None:
    if stop_at is None:
        return None
    try:
        hour, minute = map(int, stop_at.split(":"))
        now = datetime.now()
        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if target <= now:  # a past time means "that time tomorrow" (overnight runs)
            target += timedelta(days=1)
        return target.timestamp()
    except ValueError:
        raise LestError(f"invalid --stop-at {stop_at!r}; expected HH:MM") from None


@app.command()
def index(
    directory: Annotated[Path, typer.Argument(help="Directory to index.")],
    model: Annotated[
        str | None,
        typer.Option(help="Embedding model (required on first index; recorded in the DB)."),
    ] = None,
    chunker: Annotated[
        str | None,
        typer.Option(help="Chunking strategy (default: llm; recorded in the DB)."),
    ] = None,
    source: Annotated[
        SourceKind, typer.Option(help="How to read the directory.")
    ] = SourceKind.auto,
    db: Annotated[Path | None, typer.Option(help=DB_HELP)] = None,
    gpu_mode: Annotated[
        GpuMode | None,
        typer.Option(help="both: gemma on :11435 (ROCm) + embeddings on :11434; "
                          "a2000: everything sequentially on :11434."),
    ] = None,
    stop_at: Annotated[
        str | None,
        typer.Option(help="HH:MM wall-clock budget: start no new file after this "
                          "time (nightly runs; progress is kept, next run resumes)."),
    ] = None,
    limit: Annotated[
        int | None,
        typer.Option(help="Stop after this many files indexed (smoke tests / batches)."),
    ] = None,
    embedder: Annotated[
        str | None, typer.Option(hidden=True, help="Embedding backend (default: ollama).")
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Index (or incrementally re-index) DIRECTORY into its database."""
    _setup_logging(verbose)
    _apply_gpu_mode(gpu_mode)
    from .pipeline import index_directory

    stats = index_directory(
        directory,
        model=model,
        chunker_name=chunker,
        source_kind=source.value,
        embedder_name=embedder,
        db_base=db,
        stop_at=_parse_stop_at(stop_at),
        limit=limit,
    )
    logging.info(
        "done: %d documents, %d files indexed (%d LLM-fallback), %d unchanged, "
        "%d removed, %d skipped",
        stats.documents,
        stats.files_indexed,
        stats.files_fallback,
        stats.files_unchanged,
        stats.files_removed,
        stats.files_skipped,
    )


@app.command()
def search(
    directory: Annotated[Path, typer.Argument(help="Previously indexed directory.")],
    query: Annotated[str, typer.Argument(help="Search query.")],
    n: Annotated[int, typer.Option("-n", "--num-results", help="Number of results.")] = 10,
    agg: Annotated[
        str,
        typer.Option(
            help="Chunk-to-document aggregation: max, topk:K, softmax:T, decay, count:T."
        ),
    ] = "max",
    tag: Annotated[
        list[str] | None, typer.Option(help="Only documents with this tag (repeatable).")
    ] = None,
    author: Annotated[
        list[str] | None,
        typer.Option(help="Only documents by this author (fuzzy; repeatable)."),
    ] = None,
    doc_type: Annotated[
        str | None, typer.Option("--type", help="Only documents of this type.")
    ] = None,
    hybrid: Annotated[
        bool,
        typer.Option(
            "--hybrid/--no-hybrid",
            help="Fuse full-text (BM25) with vector search. Note: fused scores "
                 "are rank-based, not cosine.",
        ),
    ] = True,
    dedup: Annotated[
        bool, typer.Option("--dedup/--no-dedup", help="Collapse duplicate library entries.")
    ] = True,
    smart: Annotated[
        bool,
        typer.Option(
            "--smart",
            help="LLM query understanding: extract weighted author/tag/type/year "
                 "facets from the query, boost matches (weight 1.0 = filter), and "
                 "rerank the shortlist. Adds a few seconds; default is fast mode.",
        ),
    ] = False,
    smart_model: Annotated[
        str | None,
        typer.Option(help="Model for --smart's parse+rerank (env LEST_SMART_MODEL); "
                          "a small model here runs on the A2000 beside the embedders "
                          "so smart search stays fast while the big GPU is busy."),
    ] = None,
    gpu_mode: Annotated[
        GpuMode | None,
        typer.Option(help="Where --smart's LLM runs: both -> :11435, a2000 -> :11434."),
    ] = None,
    db: Annotated[Path | None, typer.Option(help=DB_HELP)] = None,
    json_output: Annotated[bool, typer.Option("--json", help="JSON-lines output.")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Search DIRECTORY's index; prints TSV (score, title, paths) to stdout."""
    _setup_logging(verbose)
    _apply_gpu_mode(gpu_mode)
    if smart_model is not None:
        import os

        os.environ["LEST_SMART_MODEL"] = smart_model
    from .output import format_json, format_tsv
    from .query import search_directory

    results = search_directory(
        directory,
        query,
        n=n,
        agg_spec=agg,
        db_base=db,
        tags=tag or [],
        authors=author or [],
        doc_type=doc_type,
        hybrid=hybrid,
        dedup=dedup,
        smart=smart,
    )
    for result in results:
        print(format_json(result) if json_output else format_tsv(result))


@app.command()
def status(
    directory: Annotated[Path, typer.Argument(help="Previously indexed directory.")],
    db: Annotated[Path | None, typer.Option(help=DB_HELP)] = None,
) -> None:
    """Show what is indexed for DIRECTORY."""
    _setup_logging(False)
    from .query import status_directory

    print(status_directory(directory, db_base=db), end="")


@app.command()
def ocr(
    directory: Annotated[Path, typer.Argument(help="Previously indexed directory.")],
    db: Annotated[Path | None, typer.Option(help=DB_HELP)] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """OCR the index's no-text PDFs into sidecar copies (originals untouched).

    Re-run `lest index` afterwards to pick the recovered text up.
    """
    _setup_logging(verbose)
    from .ocr import ocr_missing

    done, failed = ocr_missing(directory, db_base=db)
    logging.info("OCR: %d sidecars written, %d failed", done, failed)


@catalog_app.command("list")
def catalog_list(
    kind: Annotated[
        str | None, typer.Argument(help="tag | author | doctype (default: all)")
    ] = None,
) -> None:
    """List catalog vocabularies with usage counts and aliases."""
    from .catalog import KINDS, Catalog

    cat = Catalog()
    try:
        for k in [kind] if kind else KINDS:
            entries = cat.counts(k)
            print(f"# {k} ({len(entries)})")
            for name, usage in entries:
                aliases = cat.aliases_of(k, name)
                suffix = f"  (= {', '.join(aliases)})" if aliases else ""
                print(f"{usage:>5}  {name}{suffix}")
    finally:
        cat.close()


@catalog_app.command("review")
def catalog_review() -> None:
    """Review pending merge proposals interactively (y/n/q per proposal)."""
    from .catalog import Catalog

    cat = Catalog()
    try:
        pending = cat.pending_merges()
        if not pending:
            print("no pending merge proposals")
            return
        for merge_id, kind, keep, drop, rationale in pending:
            print(f"#{merge_id} [{kind}] merge {drop!r} into {keep!r}")
            if rationale:
                print(f"    {rationale}")
            answer = typer.prompt("    approve? [y/n/q]", default="n").strip().lower()
            if answer == "q":
                return
            cat.apply_merge(merge_id, approve=answer == "y")
    finally:
        cat.close()


def main() -> None:
    # Click handles usage errors/--help itself; LestError bubbles out of app().
    try:
        app()
    except LestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(exc.exit_code) from exc


if __name__ == "__main__":
    main()
