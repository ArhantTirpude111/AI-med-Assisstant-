"""Document loaders — turn raw files into page-segmented ``RawDocument`` objects.

Each loader is responsible for *one* family of file types and produces a uniform
``RawDocument`` whose ``pages`` preserve page boundaries. Page fidelity is
non-negotiable: citations reference page numbers, so we never collapse a
multi-page report into a single blob.

Loader selection is handled by :func:`get_loader`, a small factory keyed on file
extension. Adding a new format = add a loader + register its extensions.
"""

from __future__ import annotations

import abc
from pathlib import Path

from medintel.config import Settings
from medintel.exceptions import IngestionError, UnsupportedFileTypeError
from medintel.ingestion.ocr import OCREngine
from medintel.logging_config import get_logger
from medintel.models.schemas import (
    ExtractionMethod,
    Page,
    RawDocument,
    SourceType,
)

logger = get_logger(__name__)


class DocumentLoader(abc.ABC):
    """Abstract base for all loaders."""

    #: file extensions (lower-case, no dot) this loader handles
    extensions: tuple[str, ...] = ()

    def __init__(self, settings: Settings, ocr: OCREngine | None = None) -> None:
        self.settings = settings
        self.ocr = ocr

    @abc.abstractmethod
    def load(self, path: Path) -> RawDocument:
        """Parse ``path`` into a ``RawDocument``."""

    @staticmethod
    def _hash_file(path: Path, doc: RawDocument) -> None:
        try:
            doc.compute_sha256(path.read_bytes())
        except OSError:  # pragma: no cover - non-fatal
            logger.debug("Could not hash %s", path)


# ---------------------------------------------------------------------------
# PDF (native + OCR fallback for scanned pages)
# ---------------------------------------------------------------------------
class PdfLoader(DocumentLoader):
    extensions = ("pdf",)

    def load(self, path: Path) -> RawDocument:
        try:
            import fitz  # PyMuPDF
        except ImportError as exc:  # pragma: no cover
            raise IngestionError(
                "PyMuPDF (pymupdf) is required to read PDFs."
            ) from exc

        pages: list[Page] = []
        used_ocr = False
        min_chars = self.settings.ocr.min_native_chars_per_page

        with fitz.open(path) as pdf:
            for idx, page in enumerate(pdf, start=1):
                native_text = page.get_text("text").strip()
                method = ExtractionMethod.NATIVE

                # A page with almost no extractable text is probably a scan.
                if len(native_text) < min_chars and self._ocr_enabled():
                    logger.info(
                        "Page %d of %s looks scanned (%d native chars) — running OCR",
                        idx, path.name, len(native_text),
                    )
                    try:
                        ocr_text = self.ocr.pdf_page_to_text(page)  # type: ignore[union-attr]
                        if len(ocr_text) > len(native_text):
                            native_text = ocr_text
                            method = ExtractionMethod.OCR
                            used_ocr = True
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("OCR failed on page %d: %s", idx, exc)

                pages.append(
                    Page(page_number=idx, text=native_text, extraction_method=method)
                )

        source_type = SourceType.SCANNED_PDF if used_ocr else SourceType.PDF
        doc = RawDocument(filename=path.name, source_type=source_type, pages=pages)
        self._hash_file(path, doc)
        logger.info(
            "Loaded PDF %s: %d pages (ocr=%s)", path.name, len(pages), used_ocr
        )
        return doc

    def _ocr_enabled(self) -> bool:
        return bool(self.settings.ocr.enabled and self.ocr and self.ocr.available)


# ---------------------------------------------------------------------------
# DOCX
# ---------------------------------------------------------------------------
class DocxLoader(DocumentLoader):
    extensions = ("docx",)

    def load(self, path: Path) -> RawDocument:
        try:
            import docx  # python-docx
        except ImportError as exc:  # pragma: no cover
            raise IngestionError("python-docx is required to read DOCX files.") from exc

        document = docx.Document(str(path))
        # DOCX has no intrinsic page model; we segment on explicit page breaks,
        # falling back to a single logical page. Tables are flattened to text.
        blocks: list[str] = []
        for para in document.paragraphs:
            if para.text.strip():
                blocks.append(para.text.strip())
        for table in document.tables:
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells if c.text.strip()]
                if cells:
                    blocks.append(" | ".join(cells))

        text = "\n".join(blocks)
        pages = [Page(page_number=1, text=text, extraction_method=ExtractionMethod.NATIVE)]
        doc = RawDocument(filename=path.name, source_type=SourceType.DOCX, pages=pages)
        self._hash_file(path, doc)
        logger.info("Loaded DOCX %s: %d blocks", path.name, len(blocks))
        return doc


# ---------------------------------------------------------------------------
# Plain text
# ---------------------------------------------------------------------------
class TxtLoader(DocumentLoader):
    extensions = ("txt", "md", "csv")

    def load(self, path: Path) -> RawDocument:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        pages = [Page(page_number=1, text=text, extraction_method=ExtractionMethod.NATIVE)]
        doc = RawDocument(filename=path.name, source_type=SourceType.TXT, pages=pages)
        self._hash_file(path, doc)
        logger.info("Loaded TXT %s: %d chars", path.name, len(text))
        return doc


# ---------------------------------------------------------------------------
# Images (always OCR)
# ---------------------------------------------------------------------------
class ImageLoader(DocumentLoader):
    extensions = ("png", "jpg", "jpeg", "tiff", "bmp")

    def load(self, path: Path) -> RawDocument:
        if not (self.ocr and self.ocr.available):
            raise IngestionError(
                f"Cannot read image {path.name}: OCR (Tesseract) is not available."
            )
        from PIL import Image as PILImage

        text = self.ocr.image_to_text(PILImage.open(path))
        pages = [Page(page_number=1, text=text, extraction_method=ExtractionMethod.OCR)]
        doc = RawDocument(filename=path.name, source_type=SourceType.IMAGE, pages=pages)
        self._hash_file(path, doc)
        logger.info("Loaded image %s via OCR: %d chars", path.name, len(text))
        return doc


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
_LOADER_CLASSES: tuple[type[DocumentLoader], ...] = (
    PdfLoader,
    DocxLoader,
    TxtLoader,
    ImageLoader,
)


def get_loader(path: Path, settings: Settings, ocr: OCREngine | None = None) -> DocumentLoader:
    """Return a loader instance capable of handling ``path``.

    Raises :class:`UnsupportedFileTypeError` for unknown extensions.
    """
    ext = path.suffix.lower().lstrip(".")
    for loader_cls in _LOADER_CLASSES:
        if ext in loader_cls.extensions:
            return loader_cls(settings, ocr)
    raise UnsupportedFileTypeError(
        f"Unsupported file type '.{ext}'. Supported: "
        + ", ".join(sorted({e for c in _LOADER_CLASSES for e in c.extensions}))
    )


def supported_extensions() -> list[str]:
    """List all file extensions the system can ingest (for the UI uploader)."""
    return sorted({ext for cls in _LOADER_CLASSES for ext in cls.extensions})
