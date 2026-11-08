"""Automated evaluation script assessing answer accuracy, citation correctness, and rejection handling."""

import argparse
import json
import os
import sys
from typing import Any, Dict, List
import requests

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.config import get_settings
from app.ingestion import chunk_document_pages, extract_text_from_pdf
from app.llm import LOW_CONFIDENCE_RESPONSE, get_llm_client
from app.retrieval import RAGPipeline, VectorStoreService
from data.sample_docs.generate_sample_pdf import generate_sample_pdf


def ensure_sample_pdf_indexed(api_url: str = "", in_process_pipeline: Any = None) -> None:
    """Ensure the sample policy PDF is generated and indexed in the vector store."""
    sample_pdf_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data", "sample_docs", "sample_policy.pdf")
    )
    if not os.path.exists(sample_pdf_path):
        print(f"[EVAL SETUP] Generating sample PDF at {sample_pdf_path}...")
        generate_sample_pdf(sample_pdf_path)

    if in_process_pipeline:
        v_store: VectorStoreService = in_process_pipeline.vector_store
        docs = v_store.list_documents()
        indexed_names = [d.filename for d in docs]
        filename = os.path.basename(sample_pdf_path)
        if filename not in indexed_names:
            print(f"[EVAL SETUP] Indexing {filename} into vector store...")
            with open(sample_pdf_path, "rb") as f:
                pdf_bytes = f.read()
            pages = extract_text_from_pdf(pdf_bytes)
            settings = get_settings()
            chunks = chunk_document_pages(
                pages,
                filename=filename,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            )
            v_store.add_chunks(chunks)
            print(f"[EVAL SETUP] Indexed {len(chunks)} chunks.")
    elif api_url:
        filename = os.path.basename(sample_pdf_path)
        try:
            resp = requests.get(f"{api_url}/documents", timeout=5)
            if resp.status_code == 200:
                docs = resp.json().get("documents", [])
                if not any(d["filename"] == filename for d in docs):
                    print(f"[EVAL SETUP] Uploading {filename} to {api_url}/upload...")
                    with open(sample_pdf_path, "rb") as f:
                        upload_resp = requests.post(
                            f"{api_url}/upload",
                            files={"file": (filename, f, "application/pdf")},
                            timeout=15,
                        )
                    print(f"[EVAL SETUP] Upload status: {upload_resp.status_code}")
        except Exception as e:
            print(f"[EVAL SETUP] Warning: Could not verify documents via API: {e}")


def run_evaluation(
    api_url: str = "",
    questions_file: str = "eval/questions.json",
    similarity_threshold: float = 0.53,
) -> Dict[str, Any]:
    """Execute evaluation benchmark across the test dataset."""
    questions_path = os.path.abspath(questions_file)
    if not os.path.exists(questions_path):
        questions_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "questions.json"))

    with open(questions_path, "r", encoding="utf-8") as f:
        questions: List[Dict[str, Any]] = json.load(f)

    use_live_api = False
    in_process_pipeline = None

    if api_url:
        try:
            health = requests.get(f"{api_url}/health", timeout=3)
            if health.status_code == 200:
                use_live_api = True
                print(f"[EVAL] Connected to live API at {api_url}")
        except Exception:
            print(f"[EVAL] Could not reach API at {api_url}, falling back to in-process evaluation.")

    if not use_live_api:
        settings = get_settings()
        llm = get_llm_client(settings)
        v_store = VectorStoreService(settings, llm)
        in_process_pipeline = RAGPipeline(
            vector_store=v_store,
            llm_client=llm,
            default_similarity_threshold=settings.similarity_threshold,
            default_top_k=settings.default_top_k,
        )

    ensure_sample_pdf_indexed(
        api_url=api_url if use_live_api else "",
        in_process_pipeline=in_process_pipeline,
    )

    results = []
    correct_answers = 0
    correct_citations = 0
    total_confidence = 0.0

    print("\n" + "=" * 90)
    print(f"{'ID':<3} | {'STATUS':<6} | {'CONF':<6} | {'CITATION':<8} | {'QUESTION':<55}")
    print("=" * 90)

    for q in questions:
        q_id = q["id"]
        question_text = q["question"]
        is_oos = q.get("is_out_of_scope", False)
        expected_keywords = q.get("expected_keywords", [])
        expected_file = q.get("expected_file")
        expected_page = q.get("expected_page")

        if use_live_api:
            resp = requests.post(
                f"{api_url}/ask",
                json={"question": question_text, "similarity_threshold": similarity_threshold},
                timeout=10,
            )
            data = resp.json()
            answer = data.get("answer", "")
            sources = data.get("sources", [])
            confidence = float(data.get("confidence", 0.0))
        else:
            rag_resp = in_process_pipeline.answer_question(
                question_text,
                similarity_threshold=similarity_threshold,
            )
            answer = rag_resp.answer
            sources = [
                {"filename": s.filename, "page": s.page, "snippet": s.snippet, "similarity": s.similarity}
                for s in rag_resp.sources
            ]
            confidence = float(rag_resp.confidence)

        total_confidence += confidence

        # Evaluate Answer Accuracy
        answer_correct = False
        if is_oos:
            if LOW_CONFIDENCE_RESPONSE.lower() in answer.lower():
                answer_correct = True
        else:
            # Check for expected keywords
            matches = [kw.lower() in answer.lower() for kw in expected_keywords]
            if any(matches):  # At least one core factual keyword must match
                answer_correct = True

        # Evaluate Citation Accuracy
        citation_correct = False
        if is_oos:
            # For out of scope, either sources are empty or no false citation is asserted
            citation_correct = len(sources) == 0 or (LOW_CONFIDENCE_RESPONSE.lower() in answer.lower())
        else:
            # Source must mention expected file and expected page
            has_file = any(s.get("filename") == expected_file for s in sources)
            has_page = any(s.get("page") == expected_page for s in sources)
            in_answer_cite = (
                expected_file in answer and (f"Page {expected_page}" in answer or f"page {expected_page}" in answer)
            ) if expected_file and expected_page else False

            citation_correct = (has_file and has_page) or in_answer_cite

        if answer_correct:
            correct_answers += 1
        if citation_correct:
            correct_citations += 1

        status_str = "PASS" if (answer_correct and citation_correct) else "FAIL"
        cite_str = "PASS" if citation_correct else "FAIL"
        trunc_q = (question_text[:52] + "...") if len(question_text) > 55 else question_text

        print(f"{q_id:<3} | {status_str:<6} | {confidence:<6.3f} | {cite_str:<8} | {trunc_q:<55}")

        results.append(
            {
                "id": q_id,
                "question": question_text,
                "answer": answer,
                "sources": sources,
                "confidence": confidence,
                "answer_correct": answer_correct,
                "citation_correct": citation_correct,
            }
        )

    total_q = len(questions)
    ans_acc = round((correct_answers / total_q) * 100, 2)
    cite_acc = round((correct_citations / total_q) * 100, 2)
    mean_conf = round(total_confidence / total_q, 3)

    print("=" * 90)
    print(f"EVALUATION SUMMARY:")
    print(f"  Total Questions:     {total_q}")
    print(f"  Answer Accuracy:     {ans_acc}% ({correct_answers}/{total_q})")
    print(f"  Citation Accuracy:   {cite_acc}% ({correct_citations}/{total_q})")
    print(f"  Mean Confidence:     {mean_conf}")
    print("=" * 90 + "\n")

    return {
        "total_questions": total_q,
        "answer_accuracy_pct": ans_acc,
        "citation_accuracy_pct": cite_acc,
        "mean_confidence": mean_conf,
        "details": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run docqa-rag evaluation suite.")
    parser.add_argument("--api-url", default="", help="Optional API URL (e.g. http://localhost:8000)")
    parser.add_argument("--questions", default="eval/questions.json", help="Path to questions.json")
    parser.add_argument("--threshold", type=float, default=0.60, help="Similarity threshold")
    args = parser.parse_args()

    run_evaluation(
        api_url=args.api_url,
        questions_file=args.questions,
        similarity_threshold=args.threshold,
    )

