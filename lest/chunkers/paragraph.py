import re

from .base import register

MIN_CHARS = 700
MAX_CHARS = 2000

_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@register
class ParagraphChunker:
    """Merge paragraphs to at least MIN_CHARS; split blocks beyond MAX_CHARS at sentences.

    PDF extraction produces ragged paragraphs (single lines, headers, page joins),
    so tiny fragments are merged forward and a trailing runt is merged backward.
    """

    name = "paragraph"

    def chunk(self, text: str) -> list[str]:
        paragraphs = [p.strip() for p in _PARAGRAPH_BREAK.split(text) if p.strip()]
        merged: list[str] = []
        buffer = ""
        for paragraph in paragraphs:
            buffer = f"{buffer}\n\n{paragraph}" if buffer else paragraph
            if len(buffer) >= MIN_CHARS:
                merged.append(buffer)
                buffer = ""
        if buffer:
            if merged and len(buffer) < MIN_CHARS // 2:
                merged[-1] += "\n\n" + buffer
            else:
                merged.append(buffer)

        chunks: list[str] = []
        for block in merged:
            chunks.extend(self._split_long(block))
        return chunks

    @staticmethod
    def _split_long(block: str) -> list[str]:
        if len(block) <= MAX_CHARS:
            return [block]
        parts: list[str] = []
        buffer = ""
        for sentence in _SENTENCE_END.split(block):
            if buffer and len(buffer) + len(sentence) + 1 > MAX_CHARS:
                parts.append(buffer)
                buffer = sentence
            else:
                buffer = f"{buffer} {sentence}" if buffer else sentence
            while len(buffer) > MAX_CHARS:  # single sentence longer than the cap
                parts.append(buffer[:MAX_CHARS])
                buffer = buffer[MAX_CHARS:]
        if buffer:
            parts.append(buffer)
        return parts
