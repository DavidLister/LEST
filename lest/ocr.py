"""OCR recovery for scanned PDFs, via ocrmypdf into sidecar copies.

Sidecars live in <data dir>/ocr/ keyed by content hash; the source files
(e.g. Zotero storage) are never modified. `lest index` picks sidecars up
automatically through extract().
"""

import logging
import shutil
import subprocess
from pathlib import Path

from .errors import EnvironmentError_
from .extract import ocr_sidecar
from .store import Store, db_path_for

log = logging.getLogger(__name__)


def ocr_missing(directory: Path, db_base: Path | None = None) -> tuple[int, int]:
    """OCR every 'no_text' PDF recorded in DIRECTORY's index. Returns
    (sidecars written, failures)."""
    if shutil.which("ocrmypdf") is None:
        raise EnvironmentError_(
            "ocrmypdf not found — enter the dev shell (`nix develop`) or add "
            "ocrmypdf to your environment"
        )
    store = Store(db_path_for(directory.expanduser(), base=db_base))
    try:
        targets = [
            Path(path)
            for path, status in store.skipped_files()
            if status == "no_text" and path.lower().endswith(".pdf")
        ]
    finally:
        store.close()

    done = failed = 0
    for path in targets:
        if not path.exists():
            log.warning("missing file: %s", path)
            failed += 1
            continue
        sidecar = ocr_sidecar(path)
        if sidecar.exists():
            log.info("sidecar already present for %s", path)
            done += 1
            continue
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        tmp = sidecar.with_suffix(".tmp.pdf")
        result = subprocess.run(
            ["ocrmypdf", "--skip-text", "--optimize", "0", "--quiet", str(path), str(tmp)],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and tmp.exists():
            tmp.rename(sidecar)
            log.info("OCR ok: %s", path)
            done += 1
        else:
            tmp.unlink(missing_ok=True)
            log.warning("OCR failed for %s: %s", path, result.stderr.strip()[:300])
            failed += 1
    return done, failed
