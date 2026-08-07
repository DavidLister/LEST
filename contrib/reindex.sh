#!/usr/bin/env bash
# Launch (or resume) the full LLM re-index into data/llm, detached.
# Usage: contrib/reindex.sh [--stop-at HH:MM]
# Stop it any time with:  pkill -f "lest.cli index"   (checkpointed per file)
set -euo pipefail
cd "$(dirname "$0")/.."
log="data/llm/reindex-$(date +%Y%m%d-%H%M).log"
setsid nohup nix develop -c python -m lest.cli index "$HOME/Zotero" \
    --db data/llm --gpu-mode both -v "$@" \
    > "$log" 2>&1 &
disown
echo "re-index running, log: $log"
