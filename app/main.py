"""FastAPI application entrypoint for docqa-rag."""

from contextlib import asynccontextmanager
import json
import logging
import sys
import time
from typing import Optional
import uuid

from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import Settings, get_settings
from app.ingestion import chunk_document_pages, extract_text_from_pdf
from app.llm import get_llm_client
from app.retrieval import RAGPipeline, VectorStoreService
from app.schemas import (
    AskRequest,
    AskResponse,
    DeleteResponse,
    DocumentsListResponse,
    HealthResponse,
    UploadResponse,
)

# Configure base logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("docqa_rag")


class StructuredLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware that assigns a unique request ID and logs structured JSON telemetry for every request."""

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        start_time = time.perf_counter()

        # Attach request_id to request state
        request.state.request_id = request_id

        try:
            response: Response = await call_next(request)
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

            log_entry = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "client_ip": request.client.host if request.client else "unknown",
                "status_code": response.status_code,
                "duration_ms": duration_ms,
            }
            logger.info(json.dumps(log_entry))
            response.headers["X-Request-ID"] = request_id
            return response
        except Exception as exc:
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
            log_entry = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "client_ip": request.client.host if request.client else "unknown",
                "status_code": 500,
                "duration_ms": duration_ms,
                "error": str(exc),
            }
            logger.error(json.dumps(log_entry))
            raise


# Global application components
vector_store: Optional[VectorStoreService] = None
rag_pipeline: Optional[RAGPipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager initializing vector store and RAG pipeline components on startup."""
    global vector_store, rag_pipeline
    if vector_store is None:
        settings = get_settings()
        llm_client = get_llm_client(settings)
        vector_store = VectorStoreService(settings, llm_client)
    if rag_pipeline is None:
        settings = vector_store.settings
        rag_pipeline = RAGPipeline(
            vector_store=vector_store,
            llm_client=vector_store.llm_client,
            default_similarity_threshold=settings.similarity_threshold,
            default_top_k=settings.default_top_k,
        )
    logger.info("Application initialized with provider='%s'", vector_store.settings.llm_provider)
    yield
    logger.info("Shutting down docqa-rag service.")


app = FastAPI(
    title="docqa-rag API",
    description="Production-ready grounded Document QA RAG service with source citation and confidence thresholds.",
    version="1.0.0",
    lifespan=lifespan,
)

# Middlewares
app.add_middleware(StructuredLoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_services() -> tuple[VectorStoreService, RAGPipeline, Settings]:
    """Dependency helper returning active services."""
    global vector_store, rag_pipeline
    if vector_store is None:
        settings = get_settings()
        llm_client = get_llm_client(settings)
        vector_store = VectorStoreService(settings, llm_client)
    if rag_pipeline is None:
        settings = vector_store.settings
        rag_pipeline = RAGPipeline(
            vector_store=vector_store,
            llm_client=vector_store.llm_client,
            default_similarity_threshold=settings.similarity_threshold,
            default_top_k=settings.default_top_k,
        )
    return vector_store, rag_pipeline, vector_store.settings


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """Return health status, current vector store counts, and active LLM configuration."""
    v_store, _, settings = get_services()
    docs = v_store.list_documents()
    total_chunks = v_store.count()
    return HealthResponse(
        status="healthy",
        provider=settings.llm_provider,
        model=settings.llm_model,
        embedding_model=settings.embedding_model,
        total_documents=len(docs),
        total_chunks=total_chunks,
    )


@app.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Ingestion"],
)
async def upload_pdf(file: UploadFile = File(...)) -> UploadResponse:
    """Upload and index a PDF document page-by-page into ChromaDB with character-overlap chunking.

    Args:
        file: Uploaded PDF file binary multipart data.

    Returns:
        UploadResponse detailing extracted pages and chunk count.
    """
    v_store, _, settings = get_services()

    # Input validation
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must have a valid filename.",
        )

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type for '{file.filename}'. Only PDF files are supported.",
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes).",
        )

    try:
        pages_data = extract_text_from_pdf(file_bytes)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    total_extracted_text = "".join(text for _, text in pages_data).strip()
    if not total_extracted_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"No readable text could be extracted from PDF '{file.filename}'. Scanned images require OCR.",
        )

    chunks = chunk_document_pages(
        pages_data=pages_data,
        filename=file.filename,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unable to generate text chunks from the extracted content.",
        )

    chunks_indexed = v_store.add_chunks(chunks, overwrite=True)

    return UploadResponse(
        filename=file.filename,
        total_pages=len(pages_data),
        chunks_indexed=chunks_indexed,
        status="success",
        message=f"Successfully extracted {len(pages_data)} pages and indexed {chunks_indexed} chunks.",
    )


@app.post("/ask", response_model=AskResponse, tags=["Retrieval"])
async def ask_question(request: AskRequest) -> AskResponse:
    """Retrieve grounded context chunks, apply confidence thresholds, and synthesize an answer.

    Args:
        request: AskRequest containing query question and optional retrieval parameters.

    Returns:
        AskResponse with grounded answer, source citations, and confidence score.
    """
    _, pipeline, _ = get_services()

    cleaned_question = request.question.strip()
    if not cleaned_question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question string cannot be empty.",
        )

    response = pipeline.answer_question(
        question=cleaned_question,
        top_k=request.top_k,
        similarity_threshold=request.similarity_threshold,
    )
    return response


@app.get("/documents", response_model=DocumentsListResponse, tags=["Documents"])
async def list_documents() -> DocumentsListResponse:
    """List all currently indexed documents with chunk and page counts."""
    v_store, _, _ = get_services()
    docs = v_store.list_documents()
    total_chunks = v_store.count()
    return DocumentsListResponse(
        documents=docs,
        total_documents=len(docs),
        total_chunks=total_chunks,
    )


@app.delete(
    "/documents/{name}",
    response_model=DeleteResponse,
    tags=["Documents"],
)
async def delete_document(name: str) -> DeleteResponse:
    """Delete all indexed chunks associated with a specific document name.

    Args:
        name: Name of the document file to remove.

    Returns:
        DeleteResponse with count of removed chunks.
    """
    v_store, _, _ = get_services()
    deleted_count = v_store.delete_document(name)
    if deleted_count == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{name}' was not found in the vector index.",
        )

    return DeleteResponse(
        filename=name,
        chunks_deleted=deleted_count,
        status="success",
        message=f"Successfully deleted {deleted_count} chunks for document '{name}'.",
    )

