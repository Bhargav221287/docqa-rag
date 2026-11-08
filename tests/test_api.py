"""Integration tests for FastAPI REST API endpoints."""

from fastapi.testclient import TestClient
from app.llm import LOW_CONFIDENCE_RESPONSE


def test_health_endpoint(client: TestClient):
    """Verify GET /health returns status healthy and provider info."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "provider" in data
    assert "total_documents" in data


def test_upload_pdf_endpoint_success(client: TestClient, sample_pdf_bytes: bytes):
    """Verify POST /upload extracts pages, chunks text, and indexes in Chroma."""
    files = {"file": ("sample_specs.pdf", sample_pdf_bytes, "application/pdf")}
    response = client.post("/upload", files=files)

    assert response.status_code == 201
    data = response.json()
    assert data["filename"] == "sample_specs.pdf"
    assert data["total_pages"] == 2
    assert data["chunks_indexed"] >= 2
    assert data["status"] == "success"


def test_upload_invalid_extension(client: TestClient):
    """Verify non-PDF uploads are rejected with HTTP 400."""
    files = {"file": ("notes.txt", b"Plain text notes", "text/plain")}
    response = client.post("/upload", files=files)
    assert response.status_code == 400
    assert "Only PDF files are supported" in response.json()["detail"]


def test_upload_empty_file(client: TestClient):
    """Verify empty uploads are rejected with HTTP 400."""
    files = {"file": ("empty.pdf", b"", "application/pdf")}
    response = client.post("/upload", files=files)
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_upload_corrupt_pdf(client: TestClient):
    """Verify corrupted PDF bytes return HTTP 400."""
    files = {"file": ("corrupt.pdf", b"Random garbage bytes header", "application/pdf")}
    response = client.post("/upload", files=files)
    assert response.status_code == 400
    assert "Invalid or corrupted PDF" in response.json()["detail"]


def test_ask_endpoint_grounded(client: TestClient, sample_pdf_bytes: bytes):
    """Verify POST /ask returns grounded answer and citations for indexed content."""
    # First upload PDF
    files = {"file": ("acme_policy.pdf", sample_pdf_bytes, "application/pdf")}
    upload_res = client.post("/upload", files=files)
    assert upload_res.status_code == 201

    # Ask question
    payload = {
        "question": "What Kubernetes version does Acme Cloud operate?",
        "similarity_threshold": 0.50,
    }
    response = client.post("/ask", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_grounded"] is True
    assert data["confidence"] >= 0.50
    assert len(data["sources"]) > 0
    assert data["sources"][0]["filename"] == "acme_policy.pdf"
    assert data["sources"][0]["page"] == 1
    assert "Kubernetes 1.29" in data["answer"]


def test_ask_endpoint_low_confidence(client: TestClient, sample_pdf_bytes: bytes):
    """Verify POST /ask returns 'I don't know' fallback when confidence is low."""
    files = {"file": ("acme_policy.pdf", sample_pdf_bytes, "application/pdf")}
    client.post("/upload", files=files)

    # Ask unrelated question with high threshold
    payload = {
        "question": "What is the historical temperature in Antarctica during July 1845?",
        "similarity_threshold": 0.85,
    }
    response = client.post("/ask", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_grounded"] is False
    assert data["answer"] == LOW_CONFIDENCE_RESPONSE
    assert data["sources"] == []


def test_ask_endpoint_empty_question_validation(client: TestClient):
    """Verify blank or whitespace query is rejected with HTTP 400 or 422."""
    response = client.post("/ask", json={"question": "   "})
    assert response.status_code in [400, 422]


def test_documents_lifecycle_endpoints(client: TestClient, sample_pdf_bytes: bytes):
    """Verify GET /documents lists indexed files and DELETE /documents/{name} removes them."""
    # 1. Upload doc
    files = {"file": ("lifecycle_test.pdf", sample_pdf_bytes, "application/pdf")}
    client.post("/upload", files=files)

    # 2. List docs
    list_resp = client.get("/documents")
    assert list_resp.status_code == 200
    docs = list_resp.json()["documents"]
    assert any(d["filename"] == "lifecycle_test.pdf" for d in docs)

    # 3. Delete doc
    del_resp = client.delete("/documents/lifecycle_test.pdf")
    assert del_resp.status_code == 200
    assert del_resp.json()["status"] == "success"
    assert del_resp.json()["chunks_deleted"] >= 2

    # 4. Verify gone
    list_after = client.get("/documents")
    assert not any(d["filename"] == "lifecycle_test.pdf" for d in list_after.json()["documents"])

    # 5. Delete non-existent returns 404
    del_missing = client.delete("/documents/lifecycle_test.pdf")
    assert del_missing.status_code == 404


def test_structured_logging_request_id_header(client: TestClient):
    """Verify every HTTP response includes an X-Request-ID tracking header."""
    response = client.get("/health")
    assert "X-Request-ID" in response.headers
    assert len(response.headers["X-Request-ID"]) > 10

