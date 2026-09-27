"""Tests for embeddings, the FAISS vector store, and the retriever."""

from __future__ import annotations

import numpy as np

from medintel.processing.chunking import Chunker
from medintel.rag.embeddings import HashEmbedder, get_embedder
from medintel.rag.retriever import Retriever
from medintel.rag.vector_store import FaissVectorStore


def test_hash_embedder_is_deterministic():
    emb = HashEmbedder(dimension=128)
    a = emb.embed_query("elevated liver enzymes")
    b = emb.embed_query("elevated liver enzymes")
    assert a.shape == (128,)
    assert np.allclose(a, b)


def test_vector_store_roundtrip(settings, sample_document, tmp_path):
    embedder = get_embedder(settings)
    chunks = Chunker(settings=settings).chunk_document(sample_document)
    vectors = embedder.embed_documents([c.text for c in chunks])

    store = FaissVectorStore(dimension=embedder.dimension)
    store.add(chunks, vectors)
    assert store.count == len(chunks)

    store.save(tmp_path / "idx")
    reloaded = FaissVectorStore.load(tmp_path / "idx")
    assert reloaded.count == len(chunks)


def test_retriever_returns_relevant_chunk(settings, sample_document):
    embedder = get_embedder(settings)
    chunks = Chunker(settings=settings).chunk_document(sample_document)
    vectors = embedder.embed_documents([c.text for c in chunks])
    store = FaissVectorStore(dimension=embedder.dimension)
    store.add(chunks, vectors)

    results = Retriever(embedder, store, settings.retrieval).retrieve("metformin medication", top_k=3)
    assert results
    assert 0.0 <= results[0].score <= 1.0
    joined = " ".join(r.chunk.text.lower() for r in results)
    assert "metformin" in joined
