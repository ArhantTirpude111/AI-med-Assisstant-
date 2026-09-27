# MedIntel AI — API / Service Design

The core is exposed through **UI-agnostic service interfaces**. The Streamlit app
and CLI both call these; wrapping them in a REST API (FastAPI) is mechanical
because the services already return Pydantic models.

---

## 1. Service contracts (Python)

```python
# Ingestion
DocumentService.ingest_bytes(filename: str, data: bytes) -> RawDocument
DocumentService.ingest_many(files: list[tuple[str, bytes]]) -> list[RawDocument]

# Orchestrator (the primary façade)
MedIntelAssistant.ingest(files: list[tuple[str, bytes]]) -> IngestReport
MedIntelAssistant.ask(question: str) -> Answer
MedIntelAssistant.summarize() -> ReportSummary
MedIntelAssistant.simplify_term(term: str) -> TerminologySimplification
MedIntelAssistant.differential() -> list[DifferentialDiagnosis]
MedIntelAssistant.followups() -> list[str]
MedIntelAssistant.timeline() -> list[TimelineEvent]
MedIntelAssistant.knowledge_graph() -> networkx.Graph
MedIntelAssistant.reset() -> None
```

All return types are defined in `medintel.models.schemas`, so JSON serialization
is `model.model_dump()`.

---

## 2. Proposed REST surface

| Method | Path | Body | Returns |
|--------|------|------|---------|
| `POST` | `/v1/documents` | multipart files | `IngestReport` |
| `GET`  | `/v1/documents` | — | `list[DocumentMeta]` |
| `POST` | `/v1/ask` | `{ "question": str }` | `Answer` |
| `GET`  | `/v1/summary` | — | `ReportSummary` |
| `POST` | `/v1/terminology` | `{ "term": str }` | `TerminologySimplification` |
| `GET`  | `/v1/differential` | — | `list[DifferentialDiagnosis]` |
| `GET`  | `/v1/followups` | — | `list[str]` |
| `GET`  | `/v1/timeline` | — | `list[TimelineEvent]` |
| `GET`  | `/v1/graph` | — | `{ nodes, edges }` |
| `POST` | `/v1/reset` | — | `204` |
| `GET`  | `/healthz` | — | `{ "status": "ok" }` |

### Response envelope

```jsonc
// POST /v1/ask
{
  "question": "Are my liver enzymes elevated?",
  "answer_text": "Your ALT is 78 U/L, above the reference range of 7–56 [S1]…",
  "citations": [
    { "filename": "blood_panel.pdf", "page_number": 1, "snippet": "ALT 78 U/L (7-56) H", "score": 0.83 }
  ],
  "confidence": { "retrieval_confidence": 0.81, "label": "high", "source_count": 3 },
  "guardrail_flags": [],
  "refused": false,
  "disclaimer": "⚕️ This is an AI-generated explanation for educational purposes only…"
}
```

---

## 3. Reference FastAPI implementation (optional)

`pip install fastapi uvicorn python-multipart`, then:

```python
# api.py
from fastapi import FastAPI, UploadFile
from pydantic import BaseModel

from medintel.assistant import MedIntelAssistant

app = FastAPI(title="MedIntel AI", version="1.0.0")
assistant = MedIntelAssistant()          # one per worker; see scaling notes below


class AskRequest(BaseModel):
    question: str


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "indexed_chunks": assistant.chunk_count}


@app.post("/v1/documents")
async def upload(files: list[UploadFile]):
    payload = [(f.filename, await f.read()) for f in files]
    report = assistant.ingest(payload)
    return {
        "documents": report.document_count,
        "chunks_indexed": report.chunks_indexed,
        "used_ocr": report.used_ocr,
        "errors": report.errors,
    }


@app.post("/v1/ask")
def ask(req: AskRequest):
    return assistant.ask(req.question).model_dump()


@app.get("/v1/summary")
def summary():
    return assistant.summarize().model_dump()


@app.post("/v1/terminology")
def terminology(req: dict):
    return assistant.simplify_term(req["term"]).model_dump()
```

Run: `uvicorn api:app --host 0.0.0.0 --port 8000`.

---

## 4. Production concerns

| Concern | Approach |
|---------|----------|
| **State** | The in-process `assistant` holds a per-session index. For multi-user, key state by `tenant_id`/`session_id` and back the vector store with pgvector/Qdrant namespaces. |
| **Concurrency** | Embedding/LLM calls are I/O-bound — use async endpoints + a thread pool, or offload ingestion to a Celery/RQ worker. |
| **Auth** | Put an API gateway / OIDC in front; scope every request to a tenant. |
| **Idempotency** | `documents.sha256` enables dedup on re-upload. |
| **Rate limiting & quotas** | Enforce per-tenant LLM/embedding budgets. |
| **Observability** | Structured JSON logs (`MEDINTEL_LOG_JSON=true`) + request tracing; persist Q&A to `qa_history`. |
| **Versioning** | Prefix routes with `/v1`; evolve schemas additively. |
