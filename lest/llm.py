"""Local LLM (gemma4 via Ollama) client and the production prompts/schemas.

Host selection: LEST_GPU_MODE=both (default) sends generation to the ROCm
instance on :11435 and embeddings to :11434; LEST_GPU_MODE=a2000 points
everything at :11434 so the two models share one GPU sequentially.
LEST_LLM_HOST overrides the generation host regardless of mode.
No automatic selection — callers (or a future scheduler) choose the mode.
"""

import json
import logging
import os
import re

import httpx
import ollama

from .errors import EnvironmentError_

log = logging.getLogger(__name__)

LLM_MODEL = os.environ.get("LEST_LLM_MODEL", "gemma4:12B")
BIG_CTX = 65536  # gemma4 defaults to a tiny num_ctx; long papers need this
SMALL_CTX = 8192
MAX_TEXT_CHARS = 180_000  # ~48k tokens: fits BIG_CTX with room for output

GPU_MODES = ("both", "a2000")


def llm_host() -> str:
    override = os.environ.get("LEST_LLM_HOST")
    if override:
        return override
    mode = os.environ.get("LEST_GPU_MODE", "both")
    if mode not in GPU_MODES:
        raise EnvironmentError_(
            f"unknown LEST_GPU_MODE {mode!r}; expected one of {', '.join(GPU_MODES)}"
        )
    return "http://localhost:11434" if mode == "a2000" else "http://localhost:11435"


class LlmClient:
    """Structured-output calls to gemma4 with its quirks baked in:
    think=False (thinking otherwise eats the output budget), explicit num_ctx,
    temperature 0, one retry on invalid JSON."""

    def __init__(self, host: str | None = None, model: str | None = None):
        self.model = model or LLM_MODEL
        self.host = host or llm_host()
        self.client = ollama.Client(host=self.host)

    def ping(self) -> None:
        """Fail fast (EnvironmentError_) if the LLM endpoint or model is unusable."""
        try:
            models = [m.model for m in self.client.list().models]
        except (httpx.HTTPError, ConnectionError) as exc:
            raise EnvironmentError_(
                f"cannot reach Ollama at {self.host}: {exc} — is the service running? "
                "(LEST_GPU_MODE=a2000 uses :11434, both uses :11435)"
            ) from exc
        if not any(m.startswith(self.model.split(":")[0]) for m in models):
            raise EnvironmentError_(
                f"model {self.model!r} not found at {self.host} — "
                f"try `ollama pull {self.model}`"
            )

    def call(
        self,
        prompt: str,
        schema: dict,
        images: list[bytes] | None = None,
        num_ctx: int = BIG_CTX,
        num_predict: int = 4096,
        retries: int = 1,
    ) -> dict | None:
        """One structured call; returns parsed JSON or None after retries."""
        message = {"role": "user", "content": prompt}
        if images:
            message["images"] = images
        for attempt in range(retries + 1):
            try:
                resp = self.client.chat(
                    model=self.model,
                    messages=[message],
                    format=schema,
                    think=False,
                    options={
                        "num_ctx": num_ctx,
                        "temperature": 0.0 if attempt == 0 else 0.3,
                        "num_predict": num_predict,
                    },
                )
            except ollama.ResponseError as exc:
                raise EnvironmentError_(
                    f"Ollama error for model {self.model!r} at {self.host}: {exc.error}"
                ) from exc
            except (httpx.HTTPError, ConnectionError) as exc:
                raise EnvironmentError_(
                    f"cannot reach Ollama at {self.host}: {exc}"
                ) from exc
            try:
                return json.loads(resp["message"]["content"])
            except json.JSONDecodeError:
                log.debug("invalid JSON from %s (attempt %d)", self.model, attempt + 1)
        return None


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------- schemas

OUTLINE_SCHEMA = {
    "type": "object",
    "properties": {
        "sections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "context": {"type": "string"},
                    "ideas": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"first_words": {"type": "string"}},
                            "required": ["first_words"],
                        },
                    },
                },
                "required": ["title", "ideas"],
            },
        }
    },
    "required": ["sections"],
}

FIGURES_SCHEMA = {
    "type": "object",
    "properties": {
        "figures": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "page": {"type": "integer"},
                    "description": {"type": "string"},
                },
                "required": ["page", "description"],
            },
        }
    },
    "required": ["figures"],
}

VIEWS = ["notable", "main_ideas", "methods", "why_cite"]
VIEWS_SCHEMA = {
    "type": "object",
    "properties": {v: {"type": "string"} for v in VIEWS},
    "required": VIEWS,
}

TAGS_SCHEMA = {
    "type": "object",
    "properties": {"tags": {"type": "array", "items": {"type": "string"}}},
    "required": ["tags"],
}

DOC_TYPE_SCHEMA = {
    "type": "object",
    "properties": {"doc_type": {"type": "string"}},
    "required": ["doc_type"],
}

CHOICE_SCHEMA = {
    "type": "object",
    "properties": {"choice": {"type": "string"}},
    "required": ["choice"],
}

# ---------------------------------------------------------------- prompts
# The outline prompt is the pilot's P2b winner (granularity-disciplined) with
# per-section context blurbs (P3-style, measured free).

OUTLINE_PROMPT = """Segment this scientific paper into its logical sections, and each section
into sequential ideas (one idea = one self-contained point, argument, method
step, or result — typically 2-6 ideas per page; an idea is usually one to a
few paragraphs).

CRITICAL granularity rule: one idea spans one to several paragraphs. A typical
page contains 2-4 ideas; never emit more than 5 ideas per page. For long
papers, prefer coarser ideas — a complete derivation, experiment, or
subsection is ONE idea. Total ideas for the whole paper must stay under 80.

Rules for first_words: copy the first 5-8 words of the idea EXACTLY as they
appear in the text, character for character. Never paraphrase them. Ideas must
appear in reading order.

For every section give its title (use the paper's own headings where they
exist) and a context field: 1-2 sentences saying what the section covers
in the context of this specific paper (mention its actual subject,
materials, or methods).

PAPER TEXT:
{text}"""

FIGURES_PROMPT = """These are pages {first_page}-{last_page} of the document titled
"{title}". Describe every figure you see (skip logos, headers, and tables of
pure numbers). For each figure give its 1-based page number within the whole
document and a description of what it shows: axes, quantities, trends, and
what a reader should take from it. Mention the figure number from its caption
when visible. If there are no figures on these pages, return an empty list."""

VIEWS_PROMPT = """Read this document and answer four questions about it.
Write each answer as 2-4 dense sentences that use the document's own key
terminology (these answers will be used for search indexing).

- notable: What is most notable or memorable in this document — the thing a
  reader would remember it by?
- main_ideas: Very concisely, what are the main ideas and conclusions?
- methods: What methods, instruments, techniques, and materials does it use?
- why_cite: What problem does this document address, and in what situation
  would someone cite or reopen it?

DOCUMENT TEXT:
{text}"""

TAGS_PROMPT = """Assign 1-5 topic tags to this document. Prefer tags from the
existing vocabulary below (copy them verbatim, lowercase). Only if no existing
tag fits an important topic of this document, you may propose a new tag:
short (1-3 words, lowercase), reusable across a research library, describing
subject matter, methods, or materials — never publication type or quality.

EXISTING VOCABULARY:
{vocab}

TITLE: {title}

SUMMARY OF THE DOCUMENT:
{summary}"""

DOC_TYPE_PROMPT = """What kind of document is this? Answer with a short lowercase
type name (1-3 words), e.g. the kind of label a librarian would use for what
the document IS (not its topic). Prefer a type from the existing vocabulary
below if one fits; otherwise coin a precise new one. NEVER answer with a
vague catch-all like "misc", "other", "document", or "general".

EXISTING TYPES:
{vocab}

The library record calls it "{hint}", but judge from the content yourself.

TITLE: {title}

BEGINNING OF THE DOCUMENT:
{excerpt}"""

ADJUDICATE_PROMPT = """In a controlled vocabulary of {kind} labels, is the proposed
new label the same concept as one of these existing labels, or genuinely new?

PROPOSED: {proposed}
EXISTING CANDIDATES:
{candidates}

Answer with choice = the existing label it duplicates (copied verbatim), or
choice = "NEW" if it is a distinct concept that deserves its own label."""
