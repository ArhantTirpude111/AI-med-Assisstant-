"""High-level ingestion orchestration.

``DocumentService`` is the public façade the UI / CLI call to turn uploaded
files into ``RawDocument`` objects. It owns the OCR engine, persists uploads to
the configured upload directory (so they can be re-indexed), and routes each file
to the correct loader.
"""

from __future__ import annotations

from pathlib import Path

from medintel.config import Settings, get_settings
from medintel.exceptions import IngestionError
from medintel.ingestion.loaders import get_loader, supported_extensions
from medintel.ingestion.ocr import OCREngine
from medintel.logging_config import get_logger
from medintel.models.schemas import RawDocument

logger = get_logger(__name__)


class DocumentService:
    """Load and persist medical documents."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.ocr = OCREngine(self.settings.ocr)
        self._upload_dir = Path(self.settings.paths.upload_dir)
        self._upload_dir.mkdir(parents=True, exist_ok=True)

    # -- public API ---------------------------------------------------------
    @property
    def supported_extensions(self) -> list[str]:
        return supported_extensions()

    @property
    def ocr_available(self) -> bool:
        return self.ocr.available

    def ingest_path(self, path: str | Path) -> RawDocument:
        """Load a document already present on disk."""
        path = Path(path)
        if not path.exists():
            raise IngestionError(f"File not found: {path}")
        loader = get_loader(path, self.settings, self.ocr)
        doc = loader.load(path)
        self._validate(doc)
        return doc

    def ingest_bytes(self, filename: str, data: bytes) -> RawDocument:
        """Persist raw bytes (e.g. a Streamlit upload) then load them."""
        safe_name = self._sanitize_filename(filename)
        dest = self._upload_dir / safe_name
        dest.write_bytes(data)
        logger.info("Saved upload %s (%d bytes)", safe_name, len(data))
        return self.ingest_path(dest)

    def ingest_many(self, files: list[tuple[str, bytes]]) -> list[RawDocument]:
        """Ingest a batch of (filename, bytes) tuples, skipping failures.

        Returns the successfully ingested documents; per-file errors are logged
        so a single bad upload never aborts the whole batch.
        """
        docs: list[RawDocument] = []
        for filename, data in files:
            try:
                docs.append(self.ingest_bytes(filename, data))
            except IngestionError as exc:
                logger.error("Failed to ingest %s: %s", filename, exc)
        return docs

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def _sanitize_filename(filename: str) -> str:
        """Strip path components and keep a filesystem-safe basename."""
        name = Path(filename).name
        return "".join(c for c in name if c.isalnum() or c in (" ", ".", "_", "-")).strip()

    @staticmethod
    def _validate(doc: RawDocument) -> None:
        if doc.page_count == 0:
            raise IngestionError(f"{doc.filename}: no pages extracted.")
        if not doc.full_text.strip():
            raise IngestionError(
                f"{doc.filename}: no text could be extracted. "
                "If this is a scanned document, ensure OCR (Tesseract) is installed."
            )
