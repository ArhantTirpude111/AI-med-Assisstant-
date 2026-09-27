"""Ingestion layer: loaders, OCR, and the document service façade."""

from medintel.ingestion.document_service import DocumentService
from medintel.ingestion.loaders import get_loader, supported_extensions
from medintel.ingestion.ocr import OCREngine

__all__ = ["DocumentService", "OCREngine", "get_loader", "supported_extensions"]
