# syntax=docker/dockerfile:1
# =============================================================================
# MedIntel AI — production image
# Multi-stage build: a slim runtime with Tesseract OCR bundled.
# =============================================================================
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# --- System dependencies (Tesseract OCR + libs for PyMuPDF/pdfplumber) -------
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        libtesseract-dev \
        libgl1 \
        libglib2.0-0 \
        poppler-utils \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Python dependencies (layer-cached) --------------------------------------
COPY pyproject.toml requirements.txt README.md ./
COPY src ./src
RUN pip install --upgrade pip && pip install -e ".[ocr]"

# --- Application code ---------------------------------------------------------
COPY app ./app
COPY config ./config

# --- Non-root user & writable data volume ------------------------------------
RUN useradd --create-home --uid 1000 medintel \
    && mkdir -p /app/data /app/data/uploads /app/data/index \
    && chown -R medintel:medintel /app
USER medintel

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]
