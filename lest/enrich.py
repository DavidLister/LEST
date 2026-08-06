"""Document-level LLM enrichment: figure-description chunks from page images
(kept strictly out of the outline call — the pilot showed mixing collapses
anchor hit rates), reflection-view chunks, and tag / doc-type proposals for
the catalog to resolve.

Figure calls are batched (~8 pages per call): the pilot's single-call variant
returned zero figures on papers beyond ~10 pages.
"""

import logging
from pathlib import Path

import pymupdf

from .llm import (
    DOC_TYPE_PROMPT,
    DOC_TYPE_SCHEMA,
    FIGURES_PROMPT,
    FIGURES_SCHEMA,
    MAX_TEXT_CHARS,
    SMALL_CTX,
    TAGS_PROMPT,
    TAGS_SCHEMA,
    VIEWS,
    VIEWS_PROMPT,
    VIEWS_SCHEMA,
    LlmClient,
)

log = logging.getLogger(__name__)

RENDER_DPI = 150
PAGES_PER_CALL = 8
MAX_FIGURE_PAGES = 64  # bound cost on books; coverage beyond this is logged & skipped
FIGURES_BUDGET = 4096
VIEWS_BUDGET = 2048


class Enricher:
    def __init__(self, client: LlmClient):
        self.client = client

    # -- figures ------------------------------------------------------------

    def figure_chunks(self, pdf_path: Path, title: str) -> list[tuple[str, str]]:
        try:
            with pymupdf.open(pdf_path) as doc:
                page_count = doc.page_count
                images = [
                    doc[i].get_pixmap(dpi=RENDER_DPI).tobytes("png")
                    for i in range(min(page_count, MAX_FIGURE_PAGES))
                ]
        except Exception as exc:
            log.warning("page render failed for %s: %s", pdf_path, exc)
            return []
        if page_count > MAX_FIGURE_PAGES:
            log.info(
                "figure scan truncated to first %d of %d pages: %s",
                MAX_FIGURE_PAGES, page_count, pdf_path,
            )
        chunks: list[tuple[str, str]] = []
        for start in range(0, len(images), PAGES_PER_CALL):
            batch = images[start : start + PAGES_PER_CALL]
            result = self.client.call(
                FIGURES_PROMPT.format(
                    title=title, first_page=start + 1, last_page=start + len(batch)
                ),
                FIGURES_SCHEMA,
                images=batch,
                num_predict=FIGURES_BUDGET,
            )
            for fig in (result or {}).get("figures", []):
                page = fig.get("page")
                # models sometimes number within the batch; renumber if so
                if page is not None and page <= len(batch) and start:
                    page += start
                desc = fig.get("description", "").strip()
                if desc:
                    chunks.append(
                        ("figure", f"[{title}] [figure p.{page}] {desc}")
                    )
        return chunks

    # -- views --------------------------------------------------------------

    def view_chunks(self, text: str, title: str) -> tuple[list[tuple[str, str]], dict]:
        views = self.client.call(
            VIEWS_PROMPT.format(text=text[:MAX_TEXT_CHARS]),
            VIEWS_SCHEMA,
            num_predict=VIEWS_BUDGET,
        )
        if not views:
            return [], {}
        chunks = [
            ("view", f"[{title}] [{name}] {views[name]}")
            for name in VIEWS
            if views.get(name)
        ]
        return chunks, views

    # -- catalog proposals --------------------------------------------------

    def propose_tags(self, title: str, summary: str, vocab: list[str]) -> list[str]:
        result = self.client.call(
            TAGS_PROMPT.format(
                vocab="\n".join(vocab) if vocab else "(vocabulary is still empty)",
                title=title,
                summary=summary or "(none)",
            ),
            TAGS_SCHEMA,
            num_ctx=SMALL_CTX,
            num_predict=256,
        )
        tags = [t.strip().lower() for t in (result or {}).get("tags", []) if t.strip()]
        return list(dict.fromkeys(tags))[:5]

    def choose_doc_type(self, title: str, excerpt: str, vocab: list[str]) -> str | None:
        """Forced choice from the existing taxonomy (enum-constrained), used
        when the free proposal came back generic."""
        schema = {
            "type": "object",
            "properties": {"doc_type": {"type": "string", "enum": vocab}},
            "required": ["doc_type"],
        }
        result = self.client.call(
            DOC_TYPE_PROMPT.format(
                vocab="\n".join(vocab),
                hint="(choose the closest existing type)",
                title=title,
                excerpt=excerpt[:4000],
            ),
            schema,
            num_ctx=SMALL_CTX,
            num_predict=64,
        )
        doc_type = (result or {}).get("doc_type", "").strip().lower()
        return doc_type or None

    def propose_doc_type(
        self, title: str, excerpt: str, hint: str, vocab: list[str]
    ) -> str | None:
        result = self.client.call(
            DOC_TYPE_PROMPT.format(
                vocab="\n".join(vocab) if vocab else "(vocabulary is still empty)",
                hint=hint or "unknown",
                title=title,
                excerpt=excerpt[:4000],
            ),
            DOC_TYPE_SCHEMA,
            num_ctx=SMALL_CTX,
            num_predict=64,
        )
        doc_type = (result or {}).get("doc_type", "").strip().lower()
        return doc_type or None
