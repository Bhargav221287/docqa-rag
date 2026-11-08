"""Vector store integration using ChromaDB and RAG retrieval pipeline."""

from dataclasses import dataclass
import logging
from typing import Dict, List, Optional, Set, Tuple
import chromadb
from chromadb.config import Settings as ChromaSettings
from app.config import Settings
from app.ingestion import DocumentChunk
from app.llm import (
    BaseLLMClient,
    ChromaEmbeddingAdapter,
    LOW_CONFIDENCE_RESPONSE,
    SYSTEM_PROMPT,
)
from app.schemas import AskResponse, DocumentInfo, SourceCitation

logger = logging.getLogger("docqa_rag.retrieval")

COLLECTION_NAME = "docqa_rag_documents"


@dataclass
class RetrievedChunk:
    """Represents a chunk retrieved from the vector store with similarity scoring."""

    chunk_id: str
    filename: str
    page: int
    text: str
    similarity: float
    distance: float


class VectorStoreService:
    """Encapsulates persistent ChromaDB vector storage operations."""

    def __init__(self, settings: Settings, llm_client: BaseLLMClient):
        self.settings = settings
        self.llm_client = llm_client
        self.embedding_function = ChromaEmbeddingAdapter(llm_client)

        logger.info("Initializing ChromaDB persistent client at '%s'", settings.chroma_persist_dir)
        self.client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )

        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: List[DocumentChunk], overwrite: bool = True) -> int:
        """Add document chunks to the collection, optionally deleting existing chunks for the file.

        Args:
            chunks: List of DocumentChunk instances.
            overwrite: If True, existing chunks for the same filename are deleted first.

        Returns:
            Number of chunks successfully added.
        """
        if not chunks:
            return 0

        filename = chunks[0].filename
        if overwrite:
            self.delete_document(filename)

        ids = [c.chunk_id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [
            {"filename": c.filename, "page": c.page, "chunk_index": c.chunk_index}
            for c in chunks
        ]

        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )
        logger.info("Indexed %d chunks for file '%s'", len(chunks), filename)
        return len(chunks)

    def query_similar(self, query: str, top_k: int = 4) -> List[RetrievedChunk]:
        """Query the vector store for top-k similar chunks.

        Args:
            query: User question or search query string.
            top_k: Number of nearest neighbors to retrieve.

        Returns:
            List of RetrievedChunk objects ordered by cosine similarity descending.
        """
        total_items = self.collection.count()
        if total_items == 0:
            logger.info("Vector collection is empty, returning empty search results.")
            return []

        n_results = min(top_k, total_items)
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        retrieved: List[RetrievedChunk] = []
        if not results or not results["ids"] or not results["ids"][0]:
            return []

        ids = results["ids"][0]
        documents = results["documents"][0] if results["documents"] else []
        metadatas = results["metadatas"][0] if results["metadatas"] else []
        distances = results["distances"][0] if results["distances"] else []

        for i, chunk_id in enumerate(ids):
            meta = metadatas[i] if i < len(metadatas) else {}
            doc_text = documents[i] if i < len(documents) else ""
            dist = distances[i] if i < len(distances) else 1.0

            # For cosine distance d in [0, 2], similarity S = 1.0 - d
            similarity = max(0.0, min(1.0, 1.0 - float(dist)))

            retrieved.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    filename=str(meta.get("filename", "unknown")),
                    page=int(meta.get("page", 1)),
                    text=doc_text,
                    similarity=round(similarity, 4),
                    distance=round(float(dist), 4),
                )
            )

        # Sort descending by similarity
        retrieved.sort(key=lambda x: x.similarity, reverse=True)
        return retrieved

    def list_documents(self) -> List[DocumentInfo]:
        """Aggregate and list all unique documents present in the vector store."""
        total_count = self.collection.count()
        if total_count == 0:
            return []

        # Fetch all metadata records
        records = self.collection.get(include=["metadatas"])
        metadatas = records.get("metadatas") or []

        file_pages: Dict[str, Set[int]] = {}
        file_chunk_counts: Dict[str, int] = {}

        for meta in metadatas:
            fn = str(meta.get("filename", "unknown"))
            page = int(meta.get("page", 1))

            file_chunk_counts[fn] = file_chunk_counts.get(fn, 0) + 1
            if fn not in file_pages:
                file_pages[fn] = set()
            file_pages[fn].add(page)

        doc_infos = [
            DocumentInfo(
                filename=fn,
                total_chunks=file_chunk_counts[fn],
                pages=sorted(list(file_pages[fn])),
            )
            for fn in sorted(file_chunk_counts.keys())
        ]
        return doc_infos

    def delete_document(self, filename: str) -> int:
        """Delete all chunks belonging to a specific document.

        Args:
            filename: Target document filename.

        Returns:
            Count of deleted chunks.
        """
        # Find all chunk IDs for this filename
        records = self.collection.get(
            where={"filename": filename},
            include=["metadatas"],
        )
        ids_to_delete = records.get("ids", [])
        if not ids_to_delete:
            return 0

        self.collection.delete(ids=ids_to_delete)
        logger.info("Deleted %d chunks for document '%s'", len(ids_to_delete), filename)
        return len(ids_to_delete)

    def count(self) -> int:
        """Return total chunks in the vector collection."""
        return self.collection.count()


class RAGPipeline:
    """Coordinates retrieval, confidence threshold validation, prompt formatting, and LLM synthesis."""

    def __init__(
        self,
        vector_store: VectorStoreService,
        llm_client: BaseLLMClient,
        default_similarity_threshold: float = 0.65,
        default_top_k: int = 4,
    ):
        self.vector_store = vector_store
        self.llm_client = llm_client
        self.default_similarity_threshold = default_similarity_threshold
        self.default_top_k = default_top_k

    def build_prompt(self, question: str, chunks: List[RetrievedChunk]) -> str:
        """Build the grounded user prompt with document excerpts and injection boundaries."""
        context_blocks = []
        for c in chunks:
            # Clean snippet and escape markdown backticks if any
            clean_snippet = c.text.strip()
            context_blocks.append(
                f"[Source: {c.filename}, Page {c.page}]\n{clean_snippet}"
            )

        context_str = "\n---\n".join(context_blocks)
        prompt = (
            f"DOCUMENT CONTEXT:\n"
            f"---\n"
            f"{context_str}\n"
            f"---\n\n"
            f"USER QUESTION:\n"
            f"{question}\n\n"
            f"ANSWER:\n"
        )
        return prompt

    def answer_question(
        self,
        question: str,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
    ) -> AskResponse:
        """Execute the end-to-end RAG pipeline for a user query.

        Args:
            question: User query string.
            top_k: Optional top-k chunk count override.
            similarity_threshold: Optional minimum cosine similarity override.

        Returns:
            AskResponse object with answer, source citations, and confidence score.
        """
        k = top_k if top_k is not None else self.default_top_k
        threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else self.default_similarity_threshold
        )

        # Retrieve top candidates
        retrieved_chunks = self.vector_store.query_similar(question, top_k=k)

        # Compute overall confidence as maximum similarity among candidates
        confidence = retrieved_chunks[0].similarity if retrieved_chunks else 0.0

        # Filter chunks that pass the minimum similarity threshold
        passing_chunks = [c for c in retrieved_chunks if c.similarity >= threshold]

        # Low-confidence threshold guard
        if not passing_chunks or confidence < threshold:
            logger.info(
                "Query '%s' yielded low confidence (max=%.4f < threshold=%.4f). Returning fallback.",
                question,
                confidence,
                threshold,
            )
            return AskResponse(
                question=question,
                answer=LOW_CONFIDENCE_RESPONSE,
                sources=[],
                confidence=confidence,
                is_grounded=False,
            )

        # Construct prompt and call LLM
        prompt = self.build_prompt(question, passing_chunks)
        answer = self.llm_client.generate_answer(prompt, system_prompt=SYSTEM_PROMPT)

        # Build citations
        citations = [
            SourceCitation(
                filename=c.filename,
                page=c.page,
                snippet=c.text[:200] + ("..." if len(c.text) > 200 else ""),
                similarity=c.similarity,
            )
            for c in passing_chunks
        ]

        return AskResponse(
            question=question,
            answer=answer,
            sources=citations,
            confidence=confidence,
            is_grounded=True,
        )

