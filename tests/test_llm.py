"""Unit tests for LLM provider interfaces and prompt security guards."""

import math
from app.llm import (
    MockLLMClient,
    LOW_CONFIDENCE_RESPONSE,
    SYSTEM_PROMPT,
)
from app.retrieval import RetrievedChunk, RAGPipeline


def test_mock_llm_embedding_properties():
    """Verify MockLLMClient produces normalized vectors of consistent dimension."""
    client = MockLLMClient()
    text = "Enterprise grade kubernetes security architecture."
    vec = client.embed_query(text)

    assert len(vec) == 256
    norm = math.sqrt(sum(x * x for x in vec))
    assert abs(norm - 1.0) < 1e-4

    batch = client.embed_texts(["doc one", "doc two", "doc three"])
    assert len(batch) == 3
    for v in batch:
        assert len(v) == 256


def test_mock_llm_generates_grounded_answer():
    """Verify MockLLMClient generates citation-annotated answer when context matches query."""
    client = MockLLMClient()
    prompt = (
        "DOCUMENT CONTEXT:\n"
        "---\n"
        "[Source: policy.pdf, Page 1]\n"
        "PostgreSQL database failover promotes a standby replica within 15 seconds.\n"
        "---\n\n"
        "USER QUESTION:\n"
        "How fast does database failover occur?\n\n"
        "ANSWER:\n"
    )

    answer = client.generate_answer(prompt, system_prompt=SYSTEM_PROMPT)
    assert "15 seconds" in answer
    assert "[policy.pdf, Page 1]" in answer


def test_mock_llm_unknown_context_fallback():
    """Verify MockLLMClient returns low confidence fallback for unmatched queries."""
    client = MockLLMClient()
    prompt = (
        "DOCUMENT CONTEXT:\n"
        "---\n"
        "[Source: policy.pdf, Page 1]\n"
        "PostgreSQL database failover promotes a standby replica within 15 seconds.\n"
        "---\n\n"
        "USER QUESTION:\n"
        "What is the average rainfall in the Amazon rainforest?\n\n"
        "ANSWER:\n"
    )

    answer = client.generate_answer(prompt, system_prompt=SYSTEM_PROMPT)
    assert answer == LOW_CONFIDENCE_RESPONSE


def test_prompt_injection_guard_rules():
    """Verify the system prompt contains mandatory prompt injection countermeasures."""
    assert "untrusted data" in SYSTEM_PROMPT.lower()
    assert "never as instructions" in SYSTEM_PROMPT.lower()
    assert "ignore" in SYSTEM_PROMPT.lower()
    assert LOW_CONFIDENCE_RESPONSE in SYSTEM_PROMPT
    assert "[Filename, Page X]" in SYSTEM_PROMPT


def test_prompt_formatting_structure(test_rag_pipeline: RAGPipeline):
    """Verify RAG prompt construction properly formats document chunks and boundaries."""
    chunks = [
        RetrievedChunk(
            chunk_id="c1",
            filename="guide.pdf",
            page=2,
            text="Session tokens expire in 60 minutes.",
            similarity=0.92,
            distance=0.08,
        )
    ]
    prompt = test_rag_pipeline.build_prompt("How long are session tokens valid?", chunks)

    assert "DOCUMENT CONTEXT:" in prompt
    assert "[Source: guide.pdf, Page 2]" in prompt
    assert "Session tokens expire in 60 minutes." in prompt
    assert "USER QUESTION:" in prompt
    assert "How long are session tokens valid?" in prompt
    assert "ANSWER:" in prompt

