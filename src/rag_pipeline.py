"""
RAG Pipeline Module
Handles: Connecting retrieval (vector search) with LLM generation.
Orchestrates the full Retrieval-Augmented Generation flow.
"""

import os
from langchain_ollama import ChatOllama
from src.chains import (
    get_qa_chain,
    get_chunk_summary_chain,
    get_combine_summary_chain,
    get_compare_chain,
    get_structured_summary_chain,
    get_structured_comparison_chain,
)
from src.document_processor import load_pdf, chunk_documents, similarity_search
from src.schemas import (
    DocumentSummarySchema,
    DocumentComparisonSchema,
    parse_llm_json_to_model,
)

# --- Helpers ---

def _extract_response_text(result) -> str:
    """Extract string content from LLM response (AIMessage, dict, or str)."""
    if hasattr(result, "content"):
        return result.content
    if isinstance(result, dict) and "text" in result:
        return result["text"]
    return str(result)

# --- LLM Wrapper ---

def get_llm(max_new_tokens: int = 1024):
    """Initialize and return the API-based LLM instance (Ollama)."""
    model_name = os.getenv("LLM_MODEL_NAME", "llama3.1:8b")
    
    # Using the local Ollama integration
    # Temperature and other settings preserved as closely as possible
    chat_model = ChatOllama(
        model=model_name,
        temperature=0.1,
    )
    return chat_model


# --- RAG Question Answering ---

def answer_question(vector_store, question: str):
    """
    Full RAG pipeline:
    1. Retrieve relevant chunks via similarity search
    2. Build a context-augmented prompt
    3. Send to LLM and return the answer with source references
    """
    # 1. Retrieve relevant chunks
    chunks = similarity_search(vector_store, question, k=3)
    
    # 2. Build context string and extract metadata
    context_texts = []
    sources = []
    
    for chunk in chunks:
        context_texts.append(chunk.page_content)
        
        # Preserve original Document metadata
        # Prefer page_label, fallback to page + 1
        page_num = chunk.metadata.get("page_label") 
        if not page_num and "page" in chunk.metadata:
            # page is 0-indexed in PyPDFLoader
            page_num = str(chunk.metadata["page"] + 1)
        if not page_num:
            page_num = "Unknown"
            
        source_file = chunk.metadata.get("source", "Unknown")
        
        # Short text preview
        preview = chunk.page_content[:150].strip() + "..."
        
        sources.append({
            "page": page_num,
            "source": source_file,
            "preview": preview
        })
        
    context_str = "\n\n".join(context_texts)
    
    # 3. Call the QA chain
    llm = get_llm()
    qa_chain = get_qa_chain(llm)
    
    # Execute the LCEL chain
    result = qa_chain.invoke({
        "context": context_str,
        "question": question
    })
    
    # Extract the generated answer
    answer = _extract_response_text(result)
    
    return {
        "answer": answer.strip(),
        "sources": sources
    }


# --- Document Summarization (Phase 5) ---

def summarize_document(file_path: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> dict:
    """
    Summarize an entire academic PDF document using a Map-Reduce flow:
    PDF -> load documents -> chunk documents -> summarize each chunk -> combine summaries -> final summary

    Args:
        file_path: Path to the academic PDF document.
        chunk_size: Character size for chunking (default 1000).
        chunk_overlap: Overlap characters between chunks (default 100).

    Returns:
        dict containing:
            - final_summary: Cohesive synthesized document summary.
            - chunk_summaries: List of individual chunk summaries.
            - num_chunks: Total number of chunks processed.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    # 1. Load documents
    documents = load_pdf(file_path)
    if not documents:
        raise ValueError(f"No document content could be loaded from: {file_path}")

    # 2. Chunk documents
    chunks = chunk_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not chunks:
        raise ValueError("Document produced 0 chunks after splitting.")

    # 3. Initialize LLM and chains
    llm = get_llm()
    chunk_chain = get_chunk_summary_chain(llm)
    combine_chain = get_combine_summary_chain(llm)

    # 4. Summarize each chunk
    chunk_summaries = []
    for idx, chunk in enumerate(chunks, 1):
        response = chunk_chain.invoke({"chunk_text": chunk.page_content})
        summary_text = _extract_response_text(response).strip()
        chunk_summaries.append(summary_text)

    # 5. Combine chunk summaries into final summary
    if len(chunk_summaries) == 1:
        final_summary = chunk_summaries[0]
    else:
        combined_sections = "\n\n".join(
            [f"--- Section {idx} ---\n{summary}" for idx, summary in enumerate(chunk_summaries, 1)]
        )
        combine_response = combine_chain.invoke({"chunk_summaries": combined_sections})
        final_summary = _extract_response_text(combine_response).strip()

    return {
        "final_summary": final_summary,
        "chunk_summaries": chunk_summaries,
        "num_chunks": len(chunks)
    }


# --- Document Comparison (Phase 6) ---

def compare_documents(file_path_a: str, file_path_b: str) -> dict:
    """
    Two-Document Comparison Flow (Phase 6):
    PDF A -> load -> chunk -> summarize \
                                         -> comparison chain -> final comparison
    PDF B -> load -> chunk -> summarize /

    Args:
        file_path_a: Path to the first academic PDF document.
        file_path_b: Path to the second academic PDF document.

    Returns:
        dict containing:
            - comparison: Grounded academic comparison text.
            - doc_a_summary: Final summary of Document A.
            - doc_b_summary: Final summary of Document B.
            - doc_a_path: Path to Document A.
            - doc_b_path: Path to Document B.
    """
    if not os.path.exists(file_path_a):
        raise FileNotFoundError(f"Document A not found: {file_path_a}")
    if not os.path.exists(file_path_b):
        raise FileNotFoundError(f"Document B not found: {file_path_b}")

    # 1. Summarize Document A using Phase 5 summarization
    summary_a_result = summarize_document(file_path_a)
    doc_a_summary = summary_a_result["final_summary"]

    # 2. Summarize Document B using Phase 5 summarization
    summary_b_result = summarize_document(file_path_b)
    doc_b_summary = summary_b_result["final_summary"]

    # 3. Initialize LLM and comparison chain
    llm = get_llm()
    compare_chain = get_compare_chain(llm)

    # 4. Generate comparison
    comparison_response = compare_chain.invoke({
        "doc_a_content": doc_a_summary,
        "doc_b_content": doc_b_summary,
    })
    comparison_text = _extract_response_text(comparison_response).strip()

    return {
        "comparison": comparison_text,
        "doc_a_summary": doc_a_summary,
        "doc_b_summary": doc_b_summary,
        "doc_a_path": file_path_a,
        "doc_b_path": file_path_b,
    }


# --- Structured Output (Phase 7) ---

def summarize_document_structured(file_path: str) -> DocumentSummarySchema:
    """
    Summarize a document and return a validated Pydantic DocumentSummarySchema.

    Flow:
    1. Call summarize_document() to get the free-text summary (Phase 5).
    2. Pass the summary through the structured summary chain to extract JSON.
    3. Parse and validate the JSON into a DocumentSummarySchema.

    Args:
        file_path: Path to the academic PDF document.

    Returns:
        A validated DocumentSummarySchema instance.

    Raises:
        ValueError: If JSON extraction or Pydantic validation fails.
    """
    # 1. Get the free-text summary from Phase 5
    summary_result = summarize_document(file_path)
    summary_text = summary_result["final_summary"]

    # 2. Use the structured summary chain to extract JSON
    llm = get_llm()
    structured_chain = get_structured_summary_chain(llm)
    response = structured_chain.invoke({"summary_text": summary_text})
    raw_output = _extract_response_text(response).strip()

    # 3. Parse and validate with Pydantic
    return parse_llm_json_to_model(raw_output, DocumentSummarySchema)


def compare_documents_structured(file_path_a: str, file_path_b: str) -> DocumentComparisonSchema:
    """
    Compare two documents and return a validated Pydantic DocumentComparisonSchema.

    Flow:
    1. Call compare_documents() to get the free-text comparison (Phase 6).
    2. Pass the comparison through the structured comparison chain to extract JSON.
    3. Parse and validate the JSON into a DocumentComparisonSchema.

    Args:
        file_path_a: Path to the first academic PDF document.
        file_path_b: Path to the second academic PDF document.

    Returns:
        A validated DocumentComparisonSchema instance.

    Raises:
        ValueError: If JSON extraction or Pydantic validation fails.
    """
    # 1. Get the free-text comparison from Phase 6
    comparison_result = compare_documents(file_path_a, file_path_b)
    comparison_text = comparison_result["comparison"]

    # 2. Use the structured comparison chain to extract JSON
    llm = get_llm()
    structured_chain = get_structured_comparison_chain(llm)
    response = structured_chain.invoke({"comparison_text": comparison_text})
    raw_output = _extract_response_text(response).strip()

    # 3. Parse and validate with Pydantic
    return parse_llm_json_to_model(raw_output, DocumentComparisonSchema)


# --- Phase 4 Standalone Test Flow ---
def test_phase4_rag():
    """
    Standalone test for Phase 4 end-to-end RAG QA.
    """
    from src.document_processor import create_vector_store
    from dotenv import load_dotenv
    load_dotenv()
    
    pdf_path = "data/sample_docs/sample.pdf"
    if not os.path.exists(pdf_path):
        print(f"Error: {pdf_path} not found.")
        return
        
    print("=" * 65)
    print("  AI Academic Document Assistant — Phase 4 (RAG QA)")
    print("=" * 65)
    
    print(f"Loading and processing {pdf_path}...")
    docs = load_pdf(pdf_path)
    chunks = chunk_documents(docs)
    vector_store = create_vector_store(chunks)
    
    question = "What is Retrieval-Augmented Generation (RAG)?"
    print(f"\nQuestion: {question}")
    print("\nRetrieving context and generating answer...")
    
    try:
        response = answer_question(vector_store, question)
        
        print("\n" + "-" * 65)
        print("  Answer:")
        print("-" * 65)
        print(response["answer"])
        
        print("\n" + "-" * 65)
        print("  Sources Used:")
        print("-" * 65)
        for idx, src in enumerate(response["sources"], 1):
            print(f"\n[Source {idx}] Page: {src['page']} | Source File: {src['source']}")
            print(f"Preview: {src['preview']}")
            
        print("\n" + "=" * 65)
        print("Phase 4 test completed successfully!")
        print("=" * 65)
        
    except Exception as e:
        print(f"\n[Error] {e}")
        if "Connection" in str(e) or "connect" in str(e).lower():
            print("\nPlease ensure that Ollama is running and the llama3.1:8b model is installed.")
            print("Run 'ollama serve' and 'ollama run llama3.1:8b' in your terminal.")


# --- Phase 5 Standalone Test Flow ---
def test_phase5_summarization(file_path: str = "data/sample_docs/sample.pdf"):
    """
    Standalone test for Phase 5 whole-document summarization.
    """
    from dotenv import load_dotenv
    load_dotenv()

    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    print("=" * 65)
    print("  AI Academic Document Assistant — Phase 5 (Summarization)")
    print("=" * 65)
    print(f"\nDocument: {file_path}")
    print("\nProcessing document chunks and generating summaries...")

    try:
        result = summarize_document(file_path)

        print("\n" + "-" * 65)
        print("  Final Summary:")
        print("-" * 65)
        print(result["final_summary"])

        print("\n" + "=" * 65)
        print("Phase 5 test completed successfully!")
        print("=" * 65)

    except Exception as e:
        print(f"\n[Error] {e}")
        if "Connection" in str(e) or "connect" in str(e).lower():
            print("\nPlease ensure that Ollama is running and the llama3.1:8b model is installed.")
            print("Run 'ollama serve' and 'ollama run llama3.1:8b' in your terminal.")


# --- Phase 6 Standalone Test Flow ---
def test_phase6_comparison(
    file_path_a: str = "data/sample_docs/sample.pdf",
    file_path_b: str = "data/sample_docs/sample2.pdf",
):
    """
    Standalone test for Phase 6 two-document comparison.
    """
    from dotenv import load_dotenv
    load_dotenv()

    if not os.path.exists(file_path_a):
        print(f"Error: {file_path_a} not found.")
        return
    if not os.path.exists(file_path_b):
        print(f"Error: {file_path_b} not found.")
        return

    print("=" * 65)
    print("  AI Academic Document Assistant — Phase 6 (Document Comparison)")
    print("=" * 65)
    print(f"\nDocument A: {file_path_a}")
    print(f"Document B: {file_path_b}")
    print("\nSummarizing documents and generating comparison...")

    try:
        result = compare_documents(file_path_a, file_path_b)

        print("\n" + "-" * 65)
        print("  Comparison:")
        print("-" * 65)
        print("\n" + result["comparison"])

        print("\n" + "=" * 65)
        print("Phase 6 test completed successfully!")
        print("=" * 65)

    except Exception as e:
        print(f"\n[Error] {e}")
        if "Connection" in str(e) or "connect" in str(e).lower():
            print("\nPlease ensure that Ollama is running and the llama3.1:8b model is installed.")
            print("Run 'ollama serve' and 'ollama run llama3.1:8b' in your terminal.")


# --- Phase 7 Standalone Test Flow ---
def test_phase7_structured_output(
    file_path_a: str = "data/sample_docs/sample.pdf",
    file_path_b: str = "data/sample_docs/sample2.pdf",
):
    """
    Standalone test for Phase 7 structured output validation.
    Tests:
    1. Structured summary (Pydantic DocumentSummarySchema)
    2. Structured comparison (Pydantic DocumentComparisonSchema)
    3. Instance type validation
    4. Graceful handling of malformed data
    """
    from dotenv import load_dotenv
    load_dotenv()

    print("=" * 65)
    print("  AI Academic Document Assistant -- Phase 7 (Structured Output)")
    print("=" * 65)

    passed = 0
    failed = 0

    # --- Test 1: Structured Summary ---
    print("\n[Test 1] Structured Summary (DocumentSummarySchema)")
    print("-" * 50)
    try:
        summary_obj = summarize_document_structured(file_path_a)

        # Verify it is the correct Pydantic type
        assert isinstance(summary_obj, DocumentSummarySchema), (
            f"Expected DocumentSummarySchema, got {type(summary_obj)}"
        )
        # Verify fields are populated
        assert len(summary_obj.document_title) > 0, "document_title is empty"
        assert len(summary_obj.main_topic) > 0, "main_topic is empty"
        assert len(summary_obj.key_points) > 0, "key_points is empty"
        assert len(summary_obj.summary) > 0, "summary is empty"

        print(f"  Type: {type(summary_obj).__name__} [OK]")
        print(f"  document_title: {summary_obj.document_title}")
        print(f"  main_topic: {summary_obj.main_topic}")
        print(f"  key_points ({len(summary_obj.key_points)} items):")
        for kp in summary_obj.key_points:
            print(f"    - {kp}")
        print(f"  summary: {summary_obj.summary[:200]}...")
        print("  [PASSED]")
        passed += 1
    except Exception as e:
        print(f"  [FAILED] {e}")
        failed += 1

    # --- Test 2: Structured Comparison ---
    print(f"\n[Test 2] Structured Comparison (DocumentComparisonSchema)")
    print("-" * 50)
    try:
        comparison_obj = compare_documents_structured(file_path_a, file_path_b)

        # Verify it is the correct Pydantic type
        assert isinstance(comparison_obj, DocumentComparisonSchema), (
            f"Expected DocumentComparisonSchema, got {type(comparison_obj)}"
        )
        # Verify fields are populated
        assert len(comparison_obj.document_a_topic) > 0, "document_a_topic is empty"
        assert len(comparison_obj.document_b_topic) > 0, "document_b_topic is empty"
        assert len(comparison_obj.key_concepts) > 0, "key_concepts is empty"
        assert len(comparison_obj.similarities) > 0, "similarities is empty"
        assert len(comparison_obj.differences) > 0, "differences is empty"
        assert len(comparison_obj.key_observations) > 0, "key_observations is empty"

        print(f"  Type: {type(comparison_obj).__name__} [OK]")
        print(f"  document_a_topic: {comparison_obj.document_a_topic}")
        print(f"  document_b_topic: {comparison_obj.document_b_topic}")
        print(f"  key_concepts ({len(comparison_obj.key_concepts)} items):")
        for kc in comparison_obj.key_concepts:
            print(f"    - {kc}")
        print(f"  similarities ({len(comparison_obj.similarities)} items):")
        for s in comparison_obj.similarities:
            print(f"    - {s}")
        print(f"  differences ({len(comparison_obj.differences)} items):")
        for d in comparison_obj.differences:
            print(f"    - {d}")
        print(f"  key_observations ({len(comparison_obj.key_observations)} items):")
        for o in comparison_obj.key_observations:
            print(f"    - {o}")
        print("  [PASSED]")
        passed += 1
    except Exception as e:
        print(f"  [FAILED] {e}")
        failed += 1

    # --- Test 3: Malformed data rejection ---
    print(f"\n[Test 3] Malformed Data Rejection")
    print("-" * 50)
    try:
        # Feed clearly invalid JSON to the parser
        parse_llm_json_to_model("This is not JSON at all", DocumentSummarySchema)
        print("  [FAILED] Should have raised ValueError but did not")
        failed += 1
    except ValueError:
        print("  Malformed input correctly rejected with ValueError")
        print("  [PASSED]")
        passed += 1
    except Exception as e:
        print(f"  [FAILED] Unexpected error type: {type(e).__name__}: {e}")
        failed += 1

    # --- Test 4: Incomplete data rejection ---
    print(f"\n[Test 4] Incomplete Data Rejection")
    print("-" * 50)
    try:
        # Feed valid JSON but missing required fields
        parse_llm_json_to_model('{"document_title": "Test"}', DocumentSummarySchema)
        print("  [FAILED] Should have raised ValueError but did not")
        failed += 1
    except ValueError:
        print("  Incomplete JSON correctly rejected with ValueError")
        print("  [PASSED]")
        passed += 1
    except Exception as e:
        print(f"  [FAILED] Unexpected error type: {type(e).__name__}: {e}")
        failed += 1

    # --- Summary ---
    print("\n" + "=" * 65)
    print(f"  Phase 7 Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("  Phase 7 test completed successfully!")
    else:
        print("  Phase 7 test completed with failures.")
    print("=" * 65)


if __name__ == "__main__":
    test_phase4_rag()
    print("\n")
    test_phase5_summarization()
    print("\n")
    test_phase6_comparison()
    print("\n")
    test_phase7_structured_output()

