"""Chunking — split documents into retrievable, citation-tagged units.

Why page-aware chunking?
------------------------
Citations must reference a *page*. If we concatenated the whole document and
then split, a chunk could straddle a page boundary and we'd lose the ability to
say "this came from page 3". So we chunk **within** each page and stamp every
chunk with its page number and character offsets.

We use LangChain's ``RecursiveCharacterTextSplitter`` when available (it splits
on a hierarchy of separators — paragraphs → lines → sentences → words — keeping
semantically coherent units), and fall back to an equivalent native
implementation so the package has no hard runtime dependency on LangChain.
"""

from __future__ import annotations

from medintel.config import ChunkingConfig, Settings, get_settings
from medintel.logging_config import get_logger
from medintel.models.schemas import Chunk, Page, RawDocument

logger = get_logger(__name__)

_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]


class Chunker:
    """Split ``RawDocument`` objects into ``Chunk`` lists."""

    def __init__(self, config: ChunkingConfig | None = None, settings: Settings | None = None) -> None:
        settings = settings or get_settings()
        self.config = config or settings.chunking
        self._splitter = self._build_splitter()

    # -- public API ---------------------------------------------------------
    def chunk_document(self, document: RawDocument) -> list[Chunk]:
        """Produce page-aware chunks for a single document."""
        chunks: list[Chunk] = []
        running_index = 0
        for page in document.pages:
            for piece, start in self._split_page(page):
                chunks.append(
                    Chunk(
                        doc_id=document.doc_id,
                        filename=document.filename,
                        page_number=page.page_number,
                        chunk_index=running_index,
                        text=piece,
                        char_start=start,
                        char_end=start + len(piece),
                        metadata={"source_type": document.source_type.value},
                    )
                )
                running_index += 1
        logger.info(
            "Chunked %s into %d chunks (size=%d, overlap=%d)",
            document.filename, len(chunks), self.config.chunk_size, self.config.chunk_overlap,
        )
        return chunks

    def chunk_documents(self, documents: list[RawDocument]) -> list[Chunk]:
        out: list[Chunk] = []
        for doc in documents:
            out.extend(self.chunk_document(doc))
        return out

    # -- internals ----------------------------------------------------------
    def _split_page(self, page: Page) -> list[tuple[str, int]]:
        """Return (text, char_start) tuples for one page."""
        text = page.text.strip()
        if not text:
            return []
        if self._splitter is not None:
            pieces = self._splitter.split_text(text)
        else:
            pieces = self._native_split(text)
        # Recover approximate character offsets for each piece (best-effort).
        results: list[tuple[str, int]] = []
        cursor = 0
        for piece in pieces:
            idx = text.find(piece[:40], cursor) if piece else -1
            start = idx if idx != -1 else cursor
            results.append((piece, start))
            cursor = max(cursor, start + len(piece) - self.config.chunk_overlap)
        return results

    def _build_splitter(self):  # noqa: ANN202 - third-party type optional
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            return RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                separators=_SEPARATORS,
                length_function=len,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("LangChain splitter unavailable (%s); using native splitter.", exc)
            return None

    def _native_split(self, text: str) -> list[str]:
        """Recursive character splitter (LangChain-equivalent fallback)."""
        size, overlap = self.config.chunk_size, self.config.chunk_overlap
        if len(text) <= size:
            return [text]

        def _recurse(segment: str, seps: list[str]) -> list[str]:
            if len(segment) <= size or not seps:
                return _window(segment)
            sep = seps[0]
            parts = segment.split(sep) if sep else list(segment)
            merged: list[str] = []
            buf = ""
            for part in parts:
                candidate = f"{buf}{sep}{part}" if buf else part
                if len(candidate) <= size:
                    buf = candidate
                else:
                    if buf:
                        merged.append(buf)
                    buf = part if len(part) <= size else ""
                    if not buf:
                        merged.extend(_recurse(part, seps[1:]))
            if buf:
                merged.append(buf)
            return merged

        def _window(segment: str) -> list[str]:
            if len(segment) <= size:
                return [segment]
            out, start = [], 0
            while start < len(segment):
                out.append(segment[start : start + size])
                start += size - overlap
            return out

        return [c for c in _recurse(text, _SEPARATORS) if c.strip()]
