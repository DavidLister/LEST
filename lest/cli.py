import logging
import sys
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

from .errors import LestError

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="LEST — semantic search over a directory using a local embedding model.",
)


class SourceKind(StrEnum):
    auto = "auto"
    zotero = "zotero"
    plaindir = "plaindir"


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )


@app.command()
def index(
    directory: Annotated[Path, typer.Argument(help="Directory to index.")],
    model: Annotated[
        str | None,
        typer.Option(help="Embedding model (required on first index; recorded in the DB)."),
    ] = None,
    chunker: Annotated[
        str | None,
        typer.Option(help="Chunking strategy (default: paragraph; recorded in the DB)."),
    ] = None,
    source: Annotated[
        SourceKind, typer.Option(help="How to read the directory.")
    ] = SourceKind.auto,
    embedder: Annotated[
        str | None, typer.Option(hidden=True, help="Embedding backend (default: ollama).")
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Index (or incrementally re-index) DIRECTORY into its database."""
    _setup_logging(verbose)
    from .pipeline import index_directory

    stats = index_directory(
        directory,
        model=model,
        chunker_name=chunker,
        source_kind=source.value,
        embedder_name=embedder,
    )
    logging.info(
        "done: %d documents, %d files indexed, %d unchanged, %d removed, %d skipped",
        stats.documents,
        stats.files_indexed,
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
    json_output: Annotated[bool, typer.Option("--json", help="JSON-lines output.")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Search DIRECTORY's index; prints TSV (score, title, paths) to stdout."""
    _setup_logging(verbose)
    from .output import format_json, format_tsv
    from .query import search_directory

    results = search_directory(directory, query, n=n, agg_spec=agg)
    for result in results:
        print(format_json(result) if json_output else format_tsv(result))


@app.command()
def status(
    directory: Annotated[Path, typer.Argument(help="Previously indexed directory.")],
) -> None:
    """Show what is indexed for DIRECTORY."""
    _setup_logging(False)
    from .query import status_directory

    print(status_directory(directory), end="")


def main() -> None:
    # Click handles usage errors/--help itself; LestError bubbles out of app().
    try:
        app()
    except LestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(exc.exit_code) from exc


if __name__ == "__main__":
    main()
