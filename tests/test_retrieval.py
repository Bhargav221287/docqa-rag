"""Unit tests for vector storage operations and RAG threshold retrieval."""

import pytest
from app.ingestion import DocumentChunk
from app.llm import LOW_CONFIDENCE_RESPONSE
from app.retrieval import RAGPipeline, VectorStoreService


def test_vector_store_add_and_list(vector_store_service: VectorStoreService):
    """Verify chunks can be indexed and retrieved in document listings."""
    chunks = [
        DocumentChunk(
            chunk_id="doc1_p1_c0",
            filename="arch.pdf",
            page=1,
            chunk_index=0,
            text="Distributed database replication on PostgreSQL 16.",
        ),
        DocumentChunk(
            chunk_id="doc1_p2_c1",
            filename="arch.pdf",
            page=2,
            chunk_index=1,
            text="Service Level Agreement guarantees 99.95% monthly uptime.",
        ),
    ]

    added = vector_store_service.add_chunks(chunks)
    assert added == 2
    assert vector_store_service.count() == 2

    docs = vector_store_service.list_documents()
    assert len(docs) == 1
    assert docs[0].filename == "arch.pdf"
    assert docs[0].total_chunks == 2
    assert docs[0].pages == [1, 2]


def test_vector_store_delete_document(vector_store_service: VectorStoreService):
    """Verify document deletion removes all associated chunks and returns correct count."""
    chunks = [
        DocumentChunk(
            chunk_id="docA_p1_c0",
            filename="docA.pdf",
            page=1,
            chunk_index=0,
            text="Content for document A.",
        ),
        DocumentChunk(
            chunk_id="docB_p1_c0",
            filename="docB.pdf",
            page=1,
            chunk_index=0,
            text="Content for document B.",
        ),
    ]
    vector_store_service.add_chunks(chunks)
    assert vector_store_service.count() == 2

    # Delete docA
    deleted = vector_store_service.delete_document("docA.pdf")
    assert deleted == 1
    assert vector_store_service.count() == 1

    # Check remaining
    remaining = vector_store_service.list_documents()
    assert len(remaining) == 1
    assert remaining[0].filename == "docB.pdf"

    # Delete non-existent
    deleted_none = vector_store_service.delete_document("non_existent.pdf")
    assert deleted_none == 0


def test_retrieval_similarity_ordering(vector_store_service: VectorStoreService):
    """Verify query results are ordered by similarity descending with valid score range."""
    chunks = [
        DocumentChunk(
            chunk_id="net_p1_c0",
            filename="network.pdf",
            page=1,
            chunk_index=0,
            text="Network traffic is protected using TLS 1.3 encryption and mTLS.",
        ),
        DocumentChunk(
            chunk_id="pricing_p1_c1",
            filename="network.pdf",
            page=2,
            chunk_index=1,
            text="Subscription pricing costs $49 for starter tier accounts.",
        ),
    ]
    vector_store_service.add_chunks(chunks)

    results = vector_store_service.query_similar("What TLS version protects network traffic?", top_k=2)
    assert len(results) == 2
    assert 0.0 <= results[0].similarity <= 1.0
    assert results[0].similarity >= results[1].similarity
    assert "TLS 1.3" in results[0].text


def test_rag_pipeline_grounded_answer(test_rag_pipeline: RAGPipeline):
    """Verify high-similarity query yields grounded answer with proper citations."""
    chunks = [
        DocumentChunk(
            chunk_id="sla_p3_c0",
            filename="sla_policy.pdf",
            page=3,
            chunk_index=0,
            text="Acme Cloud commits to a 99.95% monthly uptime SLA for Enterprise accounts.",
        ),
    ]
    test_rag_pipeline.vector_store.add_chunks(chunks)

    resp = test_rag_pipeline.answer_question(
        question="What is the monthly uptime SLA guarantee for Enterprise accounts?",
        similarity_threshold=0.50,
    )

    assert resp.is_grounded is True
    assert resp.confidence >= 0.50
    assert len(resp.sources) > 0
    assert resp.sources[0].filename == "sla_policy.pdf"
    assert resp.sources[0].page == 3
    assert "99.95%" in resp.answer


def test_rag_pipeline_low_confidence_fallback(test_rag_pipeline: RAGPipeline):
    """Verify unrelated query with low similarity returns 'I don't know' fallback."""
    chunks = [
        DocumentChunk(
            chunk_id="cluster_p1_c0",
            filename="specs.pdf",
            page=1,
            chunk_index=0,
            text="PostgreSQL 16 database replication with Patroni failover in 15 seconds.",
        ),
    ]
    test_rag_pipeline.vector_store.add_chunks(chunks)

    # Ask completely unrelated query requiring high threshold
    resp = test_rag_pipeline.answer_question(
        question="What is the recipe for baking chocolate chip cookies with almond milk?",
        similarity_threshold=0.85,
    )

    assert resp.is_grounded is False
    assert resp.answer == LOW_CONFIDENCE_RESPONSE
    assert len(resp.sources) == 0


def test_rag_pipeline_empty_vector_store(test_rag_pipeline: RAGPipeline):
    """Querying an empty vector store should cleanly return fallback without errors."""
    resp = test_rag_pipeline.answer_question("What is the system uptime?")
    assert resp.is_grounded is False
    assert resp.answer == LOW_CONFIDENCE_RESPONSE
    assert resp.confidence == 0.0
    assert resp.sources == []

