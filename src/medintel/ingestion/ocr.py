"""Optical Character Recognition for scanned documents.

Wraps Tesseract (via ``pytesseract``) behind a small, dependency-tolerant
interface. If Tesseract is not installed, the engine reports ``available == False``
and callers degrade gracefully rather than crashing — important because OCR is an
*optional* system dependency.

Design notes
------------
* PDFs are rendered to images with PyMuPDF at a configurable DPI (300 is the
  sweet spot for clinical scans: high enough for small fonts, cheap enough to be
  fast).
* Light pre-processing (grayscale) measurably improves Tesseract accuracy on
  noisy fax-quality scans without the cost of a full denoising pipeline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from medintel.config import OCRConfig
from medintel.exceptions import OCRError
from medintel.logging_config import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from PIL.Image import Image

logger = get_logger(__name__)


class OCREngine:
    """Thin, lazy wrapper around Tesseract OCR."""

    def __init__(self, config: OCRConfig) -> None:
        self._config = config
        self._pytesseract: Any | None = None
        self._checked = False

    # -- availability -------------------------------------------------------
    @property
    def available(self) -> bool:
        """True if both ``pytesseract`` and the Tesseract binary are usable."""
        self._ensure_loaded()
        return self._pytesseract is not None

    def _ensure_loaded(self) -> None:
        if self._checked:
            return
        self._checked = True
        try:
            import pytesseract  # type: ignore

            if self._config.tesseract_cmd:
                pytesseract.pytesseract.tesseract_cmd = self._config.tesseract_cmd
            # Probe the binary; raises if Tesseract is not on PATH.
            version = pytesseract.get_tesseract_version()
            logger.info("OCR engine ready (Tesseract %s)", version)
            self._pytesseract = pytesseract
        except Exception as exc:  # noqa: BLE001 - we intentionally degrade
            logger.warning(
                "OCR unavailable (%s). Scanned documents will yield no text. "
                "Install Tesseract and `pip install pytesseract` to enable OCR.",
                exc,
            )
            self._pytesseract = None

    # -- image OCR ----------------------------------------------------------
    def image_to_text(self, image: Image) -> str:
        """Run OCR on a single PIL image."""
        self._ensure_loaded()
        if self._pytesseract is None:
            raise OCRError("OCR requested but Tesseract is not available.")
        try:
            prepared = image.convert("L")  # grayscale boosts accuracy
            text = self._pytesseract.image_to_string(
                prepared, lang=self._config.language
            )
            return text.strip()
        except Exception as exc:  # noqa: BLE001
            raise OCRError(f"OCR failed: {exc}") from exc

    # -- PDF page OCR -------------------------------------------------------
    def pdf_page_to_text(self, page: Any) -> str:
        """Rasterize a PyMuPDF page and OCR it.

        ``page`` is a ``fitz.Page``; typed as ``Any`` to keep PyMuPDF an
        import-time-optional dependency.
        """
        self._ensure_loaded()
        if self._pytesseract is None:
            raise OCRError("OCR requested but Tesseract is not available.")
        try:
            from io import BytesIO

            from PIL import Image as PILImage

            # 72 is the PDF base DPI; scale to the requested OCR DPI.
            zoom = self._config.dpi / 72.0
            import fitz  # type: ignore

            matrix = fitz.Matrix(zoom, zoom)
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            image = PILImage.open(BytesIO(pixmap.tobytes("png")))
            return self.image_to_text(image)
        except OCRError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise OCRError(f"PDF page OCR failed: {exc}") from exc
