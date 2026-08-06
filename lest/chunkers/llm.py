"""LLM-assisted chunker (pilot winner): gemma4 outlines the document, the
outline's `first_words` anchors become candidate cut points, and cuts are
merged mechanically into the paragraph chunker's size envelope. Anchors that
miss are harmless (the surrounding text merges over them); a failed outline
falls back to plain paragraph chunking.

Long documents are outlined in parts (the pilot's 180k-char truncation becomes
a split instead), so full text is always indexed.
"""

import logging

from ..llm import MAX_TEXT_CHARS, OUTLINE_PROMPT, OUTLINE_SCHEMA, LlmClient, normalize
from .base import register
from .paragraph import MIN_CHARS, ParagraphChunker

log = logging.getLogger(__name__)

OUTLINE_BUDGET = 16384  # output tokens; P2b needed a generous budget for 9/10 valid
PART_CHARS = 120_000  # outline call size for long docs; < MAX_TEXT_CHARS
MAX_PARTS = 6  # bound LLM cost for pathological documents (books)


def locate(anchor: str, norm_text: str) -> int:
    """Position of anchor in normalized text, -1 if absent (case-insensitive
    fallback)."""
    a = normalize(anchor)
    if not a:
        return -1
    pos = norm_text.find(a)
    if pos < 0:
        pos = norm_text.lower().find(a.lower())
    return pos


@register
class LlmChunker:
    name = "llm"

    def __init__(self, client: LlmClient | None = None):
        self._client = client
        self._fallback = ParagraphChunker()
        self.last_used_fallback = False  # observability for pipeline stats

    @property
    def client(self) -> LlmClient:
        if self._client is None:
            self._client = LlmClient()
        return self._client

    def chunk(self, text: str, title: str = "") -> list[str]:
        norm = normalize(text)
        sections = self._outline_sections(norm)
        self.last_used_fallback = sections is None
        if sections is None:
            log.info("outline failed; paragraph fallback%s", f" for {title!r}" if title else "")
            prefix = f"[{title}] " if title else ""
            return [prefix + c for c in self._fallback.chunk(text)]
        return self._cut(norm, sections, title)

    # -- outline ------------------------------------------------------------

    def _outline_sections(self, norm_text: str) -> list[dict] | None:
        """Outline the text (in parts when long); returns section dicts with
        located anchor positions, or None when every part failed. Parts beyond
        MAX_PARTS are indexed without an outline (mechanical merge only) to
        bound LLM cost on books."""
        parts = self._split_parts(norm_text)
        sections, any_ok, offset = [], False, 0
        for part_index, part in enumerate(parts):
            outline = None
            if part_index < MAX_PARTS:
                outline = self.client.call(
                    OUTLINE_PROMPT.format(text=part), OUTLINE_SCHEMA,
                    num_predict=OUTLINE_BUDGET,
                )
            elif part_index == MAX_PARTS:
                log.info("outline budget reached; remaining %d part(s) merge "
                         "mechanically", len(parts) - MAX_PARTS)
            if outline:
                any_ok = True
                for section in outline.get("sections", []):
                    positions = []
                    for idea in section.get("ideas", []):
                        pos = locate(idea.get("first_words", ""), part)
                        if pos >= 0:
                            positions.append(offset + pos)
                    sections.append({
                        "title": section.get("title", ""),
                        "context": section.get("context", ""),
                        "positions": positions,
                    })
            offset += len(part) + 1  # parts rejoin with a single space
        return sections if any_ok else None

    @staticmethod
    def _split_parts(norm_text: str) -> list[str]:
        if len(norm_text) <= MAX_TEXT_CHARS:
            return [norm_text]
        parts = []
        start = 0
        while start < len(norm_text):
            end = min(start + PART_CHARS, len(norm_text))
            if end < len(norm_text):  # cut at a sentence-ish boundary
                dot = norm_text.rfind(". ", start + PART_CHARS // 2, end)
                if dot > 0:
                    end = dot + 1
            parts.append(norm_text[start:end].strip())
            start = end
        return [p for p in parts if p]

    # -- cutting ------------------------------------------------------------

    def _cut(self, text: str, sections: list[dict], title: str) -> list[str]:
        """Slice at validated anchors, merge to the size envelope, prefix with
        document title and section context."""
        cuts = []  # (position, section index), strictly increasing
        prev = -1
        for idx, section in enumerate(sections):
            for pos in section["positions"]:
                if pos > prev:
                    cuts.append((pos, idx))
                    prev = pos
        if not cuts or cuts[0][0] > 0:
            cuts.insert(0, (0, cuts[0][1] if cuts else 0))

        pieces = []
        buffer, buffer_section = "", 0
        for (pos, idx), (next_pos, _) in zip(cuts, cuts[1:] + [(len(text), 0)], strict=True):
            piece = text[pos:next_pos].strip()
            if not piece:
                continue
            if not buffer:
                buffer_section = idx
            buffer = f"{buffer} {piece}".strip() if buffer else piece
            if len(buffer) >= MIN_CHARS:
                pieces.append((buffer_section, buffer))
                buffer = ""
        if buffer:
            if pieces and len(buffer) < MIN_CHARS // 2:
                idx, prev_text = pieces[-1]
                pieces[-1] = (idx, prev_text + " " + buffer)
            else:
                pieces.append((buffer_section, buffer))

        out = []
        for idx, piece in pieces:
            section = sections[idx] if idx < len(sections) else {"title": "", "context": ""}
            header = section["title"]
            if section["context"]:
                header = f"{header} — {section['context']}" if header else section["context"]
            prefix = "".join(
                f"[{part}] " for part in (title, header) if part
            )
            for split in ParagraphChunker._split_long(piece):
                out.append(prefix + split)
        return out
