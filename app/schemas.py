"""Pydantic request and response schemas for docqa-rag."""

from typing import List, Optional
from pydantic import BaseModel, Field


class SourceCitation(BaseModel):
    """Citation metadata identifying where information was retrieved."""

    filename: str = Field(..., description="Name of the source document.")
    page: int = Field(..., description="1-indexed page number where content resides.")
    snippet: str = Field(..., description="Excerpt of the chunk text supporting the answer.")
    similarity: float = Field(..., description="Cosine similarity score for this chunk (0.0 - 1.0).")


class AskRequest(BaseModel):
    """Request payload for asking a question against indexed documents."""

    question: str = Field(
        ...,
        min_length=1,
        description="User question to be answered using document context only.",
        examples=["What is the SLA uptime guarantee described in the document?"],
    )
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=20,
        description="Optional override for number of chunks to retrieve.",
    )
    similarity_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional override for minimum similarity score threshold.",
    )


class AskResponse(BaseModel):
    """Response payload containing grounded answer and citations."""

    question: str = Field(..., description="Original user question.")
    answer: str = Field(..., description="Grounded answer or fallback message.")
    sources: List[SourceCitation] = Field(
        default_factory=list,
        description="List of cited source excerpts used in generating the answer.",
    )
    confidence: float = Field(
        ...,
        description="Highest similarity score among retrieved chunks (0.0 - 1.0).",
    )
    is_grounded: bool = Field(
        ...,
        description="True if confident retrieval exceeded threshold; False if low confidence.",
    )


class DocumentInfo(BaseModel):
    """Metadata summary of an indexed document."""

    filename: str = Field(..., description="Filename of the indexed document.")
    total_chunks: int = Field(..., description="Number of text chunks indexed for this file.")
    pages: List[int] = Field(
        default_factory=list,
        description="List of unique page numbers represented in the vector store.",
    )


class DocumentsListResponse(BaseModel):
    """Response containing all currently indexed documents."""

    documents: List[DocumentInfo] = Field(..., description="List of indexed documents.")
    total_documents: int = Field(..., description="Total count of unique documents.")
    total_chunks: int = Field(..., description="Total count of chunks in the vector store.")


class UploadResponse(BaseModel):
    """Response returned upon successful document ingestion."""

    filename: str = Field(..., description="Name of the processed PDF.")
    total_pages: int = Field(..., description="Total number of extracted pages.")
    chunks_indexed: int = Field(..., description="Number of text chunks stored in ChromaDB.")
    status: str = Field(default="success", description="Status code string.")
    message: str = Field(..., description="Human-readable result summary.")


class DeleteResponse(BaseModel):
    """Response returned upon document deletion."""

    filename: str = Field(..., description="Name of the deleted file.")
    chunks_deleted: int = Field(..., description="Count of chunks deleted.")
    status: str = Field(default="success", description="Status code string.")
    message: str = Field(..., description="Human-readable result summary.")


class HealthResponse(BaseModel):
    """Service health and diagnostic information."""

    status: str = Field(default="healthy", description="Application health status.")
    provider: str = Field(..., description="Active LLM/Embedding provider.")
    model: str = Field(..., description="Active LLM model name.")
    embedding_model: str = Field(..., description="Active embedding model name.")
    total_documents: int = Field(..., description="Count of unique documents currently indexed.")
    total_chunks: int = Field(..., description="Total count of chunks currently indexed.")

