"""Unit tests for PDF extraction and character-overlap chunking logic."""

import pytest
from app.ingestion import (
    clean_text,
    chunk_text,
    chunk_document_pages,
    extract_text_from_pdf,
)


def test_clean_text_removes_control_characters():
    """Verify non-printable characters and excess whitespace are normalized."""
    raw = "Hello\x00World!\x0b\x0c  This   is \n\n\n a test."
    cleaned = clean_text(raw)
    assert "\x00" not in cleaned
    assert "\x0b" not in cleaned
    assert "Hello World!" in cleaned
    assert "This is" in cleaned


def test_chunk_text_empty_input():
    """Empty or whitespace-only strings must return empty list."""
    assert chunk_text("") == []
    assert chunk_text("   \n\t  ") == []


def test_chunk_text_short_string():
    """Text shorter than chunk_size should return as a single intact chunk."""
    text = "Short policy specification sentence."
    chunks = chunk_text(text, chunk_size=800, chunk_overlap=150)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_text_sliding_window_length_and_overlap():
    """Verify long text generates multiple chunks with expected overlap properties."""
    # Create ~2000 character sample text
    paragraph = (
        "Acme Cloud infrastructure architecture enforces continuous compliance across all microservices. "
        "Each service must maintain 99.95% availability and adhere strictly to zero-trust network policies. "
    )
    long_text = paragraph * 12
    assert len(long_text) > 1500

    chunk_size = 500
    overlap = 100
    chunks = chunk_text(long_text, chunk_size=chunk_size, chunk_overlap=overlap)

    assert len(chunks) >= 3
    for chunk in chunks:
        # Chunks should not vastly exceed target chunk_size
        assert len(chunk) <= chunk_size + 60
        assert len(chunk) > 50


def test_chunk_text_preserves_word_boundaries():
    """Verify chunker does not split words mid-token when space/punctuation exists."""
    text = (
        "WordAlpha WordBravo WordCharlie WordDelta WordEcho WordFoxtrot WordGolf "
        "WordHotel WordIndia WordJuliet WordKilo WordLima WordMike WordNovember"
    )
    chunks = chunk_text(text, chunk_size=50, chunk_overlap=15)
    for c in chunks:
        words = c.split()
        for w in words:
            assert w.startswith("Word")


def test_chunk_document_pages_metadata_preservation():
    """Verify page numbers, filenames, and chunk indices are accurately assigned."""
    pages_data = [
        (1, "Page one text content explaining Kubernetes cluster architecture."),
        (2, "Page two text content detailing encryption protocols and key rotation."),
    ]
    filename = "test_policy.pdf"
    chunks = chunk_document_pages(
        pages_data=pages_data,
        filename=filename,
        chunk_size=300,
        chunk_overlap=50,
    )

    assert len(chunks) == 2
    assert chunks[0].filename == filename
    assert chunks[0].page == 1
    assert chunks[0].chunk_index == 0
    assert "Kubernetes" in chunks[0].text

    assert chunks[1].filename == filename
    assert chunks[1].page == 2
    assert chunks[1].chunk_index == 1
    assert "encryption" in chunks[1].text


def test_extract_text_from_pdf_valid(sample_pdf_bytes: bytes):
    """Verify pypdf extracts text and page counts accurately from valid PDF bytes."""
    pages_data = extract_text_from_pdf(sample_pdf_bytes)
    assert len(pages_data) == 2
    assert pages_data[0][0] == 1
    assert "Kubernetes 1.29" in pages_data[0][1]
    assert pages_data[1][0] == 2
    assert "99.95%" in pages_data[1][1]


def test_extract_text_from_pdf_empty_bytes():
    """Empty byte input must raise ValueError."""
    with pytest.raises(ValueError, match="empty"):
        extract_text_from_pdf(b"")


def test_extract_text_from_pdf_corrupt_bytes():
    """Corrupted non-PDF binary data must raise ValueError."""
    with pytest.raises(ValueError, match="Invalid or corrupted PDF"):
        extract_text_from_pdf(b"Not a valid PDF file content at all")

