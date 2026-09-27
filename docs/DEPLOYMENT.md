# MedIntel AI — Deployment Guide

Covers local development, OCR/NER system dependencies, Docker, cloud options, and
the (non-negotiable) compliance checklist for handling medical data.

---

## 1. Local development

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -e ".[full,dev]"      # core + OCR/NER/local-embeddings + dev tools
cp .env.example .env              # set OPENAI_API_KEY (or use offline mode)

streamlit run app/streamlit_app.py
```

### Offline / no API key

```bash
export MEDINTEL_LLM__PROVIDER=mock
export MEDINTEL_EMBEDDINGS__PROVIDER=huggingface   # or 'hash' for zero downloads
streamlit run app/streamlit_app.py
```

---

## 2. System dependency: Tesseract OCR

OCR for scanned PDFs/images requires the Tesseract binary (Python `pytesseract`
is only a wrapper).

| OS | Install |
|----|---------|
| Windows | `choco install tesseract` — or the UB-Mannheim installer. Then set `MEDINTEL_OCR__TESSERACT_CMD` to `tesseract.exe` if not on PATH. |
| Debian/Ubuntu | `sudo apt-get install -y tesseract-ocr` |
| macOS | `brew install tesseract` |

Verify: `tesseract --version`. The Docker image bundles it automatically.

---

## 3. Optional: SciSpaCy medical NER

```bash
pip install -e ".[ner]"
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz
export MEDINTEL_NER__BACKEND=scispacy
```

Without it, NER falls back to the curated gazetteer (no crash, lower recall).

---

## 4. Docker (recommended for demos)

```bash
# from repo root
docker compose -f docker/docker-compose.yml up --build
# → http://localhost:8501
```

* Tesseract is preinstalled in the image.
* Data (uploads, FAISS index, SQLite DB) persists in the `medintel-data` volume.
* Provide secrets via `.env` (read by compose) or real env vars.

Build/run the image directly:

```bash
docker build -f docker/Dockerfile -t medintel-ai .
docker run -p 8501:8501 --env-file .env -v medintel-data:/app/data medintel-ai
```

---

## 5. Cloud deployment options

| Target | Notes |
|--------|-------|
| **Streamlit Community Cloud** | Fast demo. Add `OPENAI_API_KEY` in app secrets. OCR/SciSpaCy limited. |
| **Azure Container Apps / AWS App Runner / Google Cloud Run** | Push the Docker image; mount a persistent volume or move state to managed Postgres + object storage. |
| **Kubernetes** | Stateless app pods + managed Postgres (pgvector) + S3/GCS for the FAISS index. Use the `VectorStore` interface to externalize the index. |
| **On-prem (PHI)** | Use offline embeddings (`huggingface`) and a self-hosted LLM endpoint (set `MEDINTEL_LLM__PROVIDER=openai` pointed at a compatible gateway). |

### Production hardening

* Move `paths.db_url` to Postgres; externalize the FAISS index to shared storage.
* Run behind TLS + auth (reverse proxy / identity provider).
* Set `MEDINTEL_LOG_JSON=true` and ship logs to your platform.
* Scale stateless replicas; keep the index/db in shared services.

---

## 6. Configuration reference

| Variable | Default | Purpose |
|----------|---------|---------|
| `MEDINTEL_LLM__PROVIDER` | `openai` | `openai` \| `mock` |
| `MEDINTEL_LLM__MODEL` | `gpt-4o-mini` | Chat model |
| `MEDINTEL_EMBEDDINGS__PROVIDER` | `openai` | `openai` \| `huggingface` \| `hash` |
| `MEDINTEL_RETRIEVAL__TOP_K` | `5` | Chunks injected into the prompt |
| `MEDINTEL_RETRIEVAL__MIN_CONFIDENCE` | `0.35` | Refusal threshold |
| `MEDINTEL_OCR__ENABLED` | `true` | Toggle OCR |
| `MEDINTEL_NER__BACKEND` | `auto` | `auto` \| `scispacy` \| `rules` |
| `MEDINTEL_PATHS__DB_URL` | `sqlite:///data/medintel.db` | Relational store |
| `OPENAI_API_KEY` | — | Secret (never commit) |

---

## 7. Health & smoke checks

```bash
# Container health endpoint
curl -fsS http://localhost:8501/_stcore/health

# Headless pipeline smoke test
python -m medintel.cli status
python -m medintel.cli ingest samples/sample_blood_report.txt
python -m medintel.cli ask "Are my liver enzymes elevated?"
```

---

## 8. ⚠️ Compliance checklist (handling medical data)

> This repository is a **reference implementation**, not a certified medical
> device. Before processing real patient data:

- [ ] Execute a **BAA** with any third-party model/API provider, or self-host.
- [ ] **Encrypt** data at rest (DB + index volume) and in transit (TLS).
- [ ] Add **authentication, authorization (RBAC)**, and per-tenant isolation.
- [ ] Enable **audit logging** of access and Q&A (the `qa_history` table is a start).
- [ ] Implement **de-identification** at ingestion if PHI must not leave the boundary.
- [ ] Define **data retention & deletion** policies (right-to-erasure).
- [ ] Keep the **non-diagnostic disclaimer** visible; add human-in-the-loop review.
- [ ] Run a security review; never commit `.env`, `data/`, or any PHI.
