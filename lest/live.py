"""Persistent as-you-type search loop.

Protocol: one query per stdin line; per query, results are printed (TSV or
JSON lines) followed by one blank line, then flushed. An empty query yields
an empty block. If several lines are queued (the user typed faster than we
searched), only the newest is answered. EOF ends the loop.
"""

import select
import sys
from pathlib import Path

from .errors import LestError
from .output import format_json, format_tsv
from .query import search_directory


def _drain_to_newest(stream) -> str | None:
    """Blocking read of one line, then swallow any already-queued lines and
    return the newest. None on EOF."""
    line = stream.readline()
    if line == "":
        return None
    while select.select([stream], [], [], 0)[0]:
        newer = stream.readline()
        if newer == "":  # EOF after queued input: serve what we have
            break
        line = newer
    return line.rstrip("\n")


def live_loop(
    directory: Path,
    n: int = 10,
    agg_spec: str = "max",
    db_base: Path | None = None,
    json_output: bool = False,
    stream=None,
) -> None:
    stream = stream if stream is not None else sys.stdin
    while True:
        query = _drain_to_newest(stream)
        if query is None:
            return
        if query.strip():
            try:
                results = search_directory(
                    directory, query, n=n, agg_spec=agg_spec, db_base=db_base
                )
                for result in results:
                    print(format_json(result) if json_output else format_tsv(result))
            except LestError as exc:  # keep the session alive on bad input
                print(f"error: {exc}", file=sys.stderr)
        print(flush=True)
