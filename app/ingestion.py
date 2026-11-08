"""PDF ingestion, text extraction, and character-overlap chunking module."""

from dataclasses import dataclass
import io
import logging
import re
from typing import List, Tuple
from pypdf import PdfReader
from pypdf.errors import PdfReadError

logger = logging.getLogger("docqa_rag.ingestion")


@dataclass
class DocumentChunk:
    """Represents a discrete text chunk extracted from a document page."""

    chunk_id: str
    filename: str
    page: int
    chunk_index: int
    text: str


def clean_text(text: str) -> str:
    """Normalize whitespace and strip null/control characters from raw PDF text."""
    if not text:
        return ""
    # Replace control characters with space to prevent concatenating adjacent words
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
    # Normalize multiple whitespace to single space while preserving paragraph breaks
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def extract_text_from_pdf(file_bytes: bytes) -> List[Tuple[int, str]]:
    """Extract text from a PDF file page by page using pypdf.

    Args:
        file_bytes: Raw binary content of the PDF.

    Returns:
        List of tuples: (1-indexed page number, extracted page text).

    Raises:
        ValueError: If file is empty or cannot be parsed as a PDF.
    """
    if not file_bytes:
        raise ValueError("Uploaded file is empty (0 bytes).")

    try:
        reader = PdfReader(io.BytesIO(file_bytes))
    except (PdfReadError, Exception) as exc:
        logger.error("Failed to parse PDF binary: %s", str(exc))
        raise ValueError(f"Invalid or corrupted PDF file: {exc}") from exc

    if len(reader.pages) == 0:
        raise ValueError("PDF file contains no pages.")

    pages_data: List[Tuple[int, str]] = []
    for idx, page in enumerate(reader.pages):
        page_num = idx + 1
        try:
            raw_text = page.extract_text() or ""
        except Exception as exc:
            logger.warning("Error extracting text on page %d: %s", page_num, exc)
            raw_text = ""
        pages_data.append((page_num, clean_text(raw_text)))

    return pages_data


def chunk_text(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> List[str]:
    """Split text into overlapping chunks, respecting sentence and word boundaries where possible.

    Args:
        text: Input string to chunk.
        chunk_size: Target character length of each chunk (approx 800 chars).
        chunk_overlap: Number of characters to overlap between consecutive chunks (approx 150 chars).

    Returns:
        List of chunk strings.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return []

    if len(cleaned) <= chunk_size:
        return [cleaned]

    if chunk_overlap >= chunk_size:
        chunk_overlap = chunk_size // 4

    step = chunk_size - chunk_overlap
    chunks: List[str] = []
    start = 0
    text_len = len(cleaned)

    while start < text_len:
        end = min(start + chunk_size, text_len)

        # If we are not at the end of text, seek a natural boundary (newline, period, or space)
        if end < text_len:
            # Look backwards up to 60 characters for a sentence or word break
            lookback_zone = cleaned[max(start, end - 60) : end]
            boundary_match = re.search(r"(\.\s+|\n+|;\s+|\?\s+|\!\s+|\s+)", lookback_zone[::-1])
            if boundary_match:
                end = end - boundary_match.start()

        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_len:
            break

        # Next window start with overlap
        next_start = end - chunk_overlap
        # Align next_start to the beginning of a word so we never slice a token mid-word
        if next_start > start and next_start < text_len:
            space_before = cleaned.rfind(" ", start, next_start)
            if space_before != -1 and space_before > start:
                next_start = space_before + 1
            else:
                space_after = cleaned.find(" ", next_start)
                if space_after != -1 and space_after < end:
                    next_start = space_after + 1

        # Ensure forward progress
        if next_start <= start:
            next_start = start + step
        start = next_start

    return chunks


def chunk_document_pages(
    pages_data: List[Tuple[int, str]],
    filename: str,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> List[DocumentChunk]:
    """Chunk all pages of a document while preserving exact page number metadata.

    Args:
        pages_data: List of (page_number, text) tuples.
        filename: Name of the source file.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Character overlap between consecutive chunks.

    Returns:
        List of DocumentChunk objects with unique IDs and metadata.
    """
    all_chunks: List[DocumentChunk] = []
    chunk_counter = 0

    for page_num, page_text in pages_data:
        if not page_text:
            continue

        raw_chunks = chunk_text(
            page_text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        for chunk_str in raw_chunks:
            chunk_id = f"{filename}_p{page_num}_c{chunk_counter}"
            all_chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    filename=filename,
                    page=page_num,
                    chunk_index=chunk_counter,
                    text=chunk_str,
                )
            )
            chunk_counter += 1

    return all_chunks

