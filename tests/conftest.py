"""Pytest configuration and shared test fixtures."""

import io
import shutil
import tempfile
from typing import Generator
import pytest
from fastapi.testclient import TestClient
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, PageBreak
from reportlab.lib.styles import getSampleStyleSheet

from app.config import Settings
from app.llm import MockLLMClient
from app.retrieval import RAGPipeline, VectorStoreService
from app.main import app
import app.main as main_module


@pytest.fixture(scope="session")
def sample_pdf_bytes() -> bytes:
    """Generate an in-memory 2-page sample PDF for testing."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter)
    styles = getSampleStyleSheet()

    story = [
        Paragraph("Acme Cloud Overview and Security Standard", styles["Heading1"]),
        Paragraph(
            "Acme Cloud operates on Kubernetes 1.29 across us-east and eu-central regions. "
            "Data at rest is secured using AES-256 encryption with automated key rotation.",
            styles["Normal"],
        ),
        PageBreak(),
        Paragraph("Compliance and SLA Policy", styles["Heading2"]),
        Paragraph(
            "Enterprise customers receive a guaranteed 99.95% monthly uptime SLA. "
            "System audit logs are preserved for exactly 365 days in immutable storage.",
            styles["Normal"],
        ),
    ]

    doc.build(story)
    buf.seek(0)
    return buf.getvalue()


@pytest.fixture
def temp_chroma_dir() -> Generator[str, None, None]:
    """Provide an isolated temporary directory for ChromaDB storage per test."""
    temp_dir = tempfile.mkdtemp(prefix="chroma_test_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def test_settings(temp_chroma_dir: str) -> Settings:
    """Create test application settings with Mock provider and temp Chroma path."""
    return Settings(
        llm_provider="mock",
        llm_api_key="test-mock-key",
        chroma_persist_dir=temp_chroma_dir,
        similarity_threshold=0.60,
        chunk_size=400,
        chunk_overlap=80,
        default_top_k=3,
    )


@pytest.fixture
def mock_llm_client() -> MockLLMClient:
    """Create an instance of MockLLMClient."""
    return MockLLMClient()


@pytest.fixture
def vector_store_service(
    test_settings: Settings, mock_llm_client: MockLLMClient
) -> VectorStoreService:
    """Create an isolated VectorStoreService instance for testing."""
    return VectorStoreService(test_settings, mock_llm_client)


@pytest.fixture
def test_rag_pipeline(
    vector_store_service: VectorStoreService, mock_llm_client: MockLLMClient
) -> RAGPipeline:
    """Create an isolated RAGPipeline instance for testing."""
    return RAGPipeline(
        vector_store=vector_store_service,
        llm_client=mock_llm_client,
        default_similarity_threshold=0.60,
        default_top_k=3,
    )


@pytest.fixture
def client(
    vector_store_service: VectorStoreService,
    test_rag_pipeline: RAGPipeline,
    test_settings: Settings,
) -> Generator[TestClient, None, None]:
    """Provide a TestClient with globally patched vector store and pipeline services."""
    main_module.vector_store = vector_store_service
    main_module.rag_pipeline = test_rag_pipeline

    with TestClient(app) as test_client:
        yield test_client

    main_module.vector_store = None
    main_module.rag_pipeline = None

