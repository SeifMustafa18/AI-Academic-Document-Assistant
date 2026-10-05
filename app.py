"""
AI Academic Document Assistant — Streamlit Application

Features:
1. Q&A: Upload an academic PDF, index it into FAISS, and ask questions grounded in the document.
2. Summary: Generate structured academic summaries using a Map-Reduce chunking approach.
3. Compare Documents: Upload two academic PDFs and produce a grounded comparative analysis.
"""

import atexit
import os
import streamlit as st

from src.utils import (
    load_environment,
    save_uploaded_file_to_temp,
    cleanup_temp_file,
)
from src.document_processor import (
    load_pdf,
    chunk_documents,
    create_vector_store,
)
from src.rag_pipeline import (
    answer_question,
    summarize_document,
    compare_documents,
    summarize_document_structured,
    compare_documents_structured,
)
from src.schemas import DocumentSummarySchema, DocumentComparisonSchema

# Load environment variables (.env)
load_environment()

# Register temporary files cleanup on program exit
_registered_temp_files = set()


def _track_and_save_temp(uploaded_file, prefix: str = "doc_") -> str:
    """Save an uploaded file to a temporary location and register it for cleanup."""
    temp_path = save_uploaded_file_to_temp(uploaded_file, prefix=prefix)
    _registered_temp_files.add(temp_path)
    return temp_path


def _replace_session_temp_file(state_key: str, uploaded_file, prefix: str = "doc_") -> str:
    """Safely replace a previous session temp file with a newly uploaded one."""
    previous_path = st.session_state.get(state_key)
    if previous_path:
        cleanup_temp_file(previous_path)
        _registered_temp_files.discard(previous_path)
    new_path = _track_and_save_temp(uploaded_file, prefix=prefix)
    st.session_state[state_key] = new_path
    return new_path


@atexit.register
def _cleanup_all_registered_temp_files():
    """Cleanup any remaining temp files on process exit."""
    for p in list(_registered_temp_files):
        cleanup_temp_file(p)


# --- Page Configuration ---
st.set_page_config(
    page_title="AI Academic Document Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("AI Academic Document Assistant")
st.markdown("Analyze academic PDF documents using RAG, summarization, and document comparison.")


# --- Feature Tabs ---
tab_qa, tab_summary, tab_compare = st.tabs(["Q&A", "Summary", "Compare Documents"])


# ==================================================
# 1. Q&A TAB
# ==================================================
with tab_qa:
    st.subheader("Document Question Answering (RAG)")
    st.markdown("Upload an academic PDF, index it into FAISS, and ask questions answered strictly from the document context.")

    qa_file = st.file_uploader(
        "Upload PDF Document",
        type=["pdf"],
        help="Upload a single academic PDF file.",
        key="qa_file_uploader",
    )

    current_file_id = f"{qa_file.name}_{qa_file.size}" if qa_file else None
    previous_file_id = st.session_state.get("qa_file_id")

    if qa_file is None:
        if previous_file_id is not None:
            # File was removed by the user — clear session state and temp file
            cleanup_temp_file(st.session_state.get("qa_temp_path"))
            _registered_temp_files.discard(st.session_state.get("qa_temp_path"))
            st.session_state["qa_file_id"] = None
            st.session_state["qa_vector_store"] = None
            st.session_state["qa_file_info"] = None
            st.session_state["qa_last_answer"] = None
            st.session_state["qa_last_sources"] = None
            st.session_state["qa_last_question"] = None
    else:
        # File is present; check if it is newly uploaded or changed
        if current_file_id != previous_file_id:
            with st.spinner("Processing document: loading pages, chunking text, and building FAISS vector store..."):
                try:
                    temp_path = _replace_session_temp_file("qa_temp_path", qa_file, prefix="qa_")
                    docs = load_pdf(temp_path)
                    if not docs:
                        st.error("The uploaded PDF does not contain extractable text.")
                    else:
                        chunks = chunk_documents(docs)
                        if not chunks:
                            st.error("No text chunks could be produced from this document.")
                        else:
                            vector_store = create_vector_store(chunks)
                            st.session_state["qa_vector_store"] = vector_store
                            st.session_state["qa_file_id"] = current_file_id
                            st.session_state["qa_file_info"] = {
                                "name": qa_file.name,
                                "pages": len(docs),
                                "chunks": len(chunks),
                            }
                            # Reset previous answer since a new document was loaded
                            st.session_state["qa_last_answer"] = None
                            st.session_state["qa_last_sources"] = None
                            st.session_state["qa_last_question"] = None
                except Exception as e:
                    st.error(f"Failed to process uploaded PDF: {e}")

    # Display document info badge if indexed
    doc_ready = st.session_state.get("qa_vector_store") is not None
    if doc_ready and st.session_state.get("qa_file_info"):
        info = st.session_state["qa_file_info"]
        st.success(f"Indexed: **{info['name']}** ({info['pages']} pages, {info['chunks']} chunks ready for search)")

    # Question Input
    question = st.text_input(
        "Ask a question about the document:",
        placeholder="e.g., What is Retrieval-Augmented Generation (RAG)?",
        key="qa_question_input",
    )

    # Ask button disabled until document has been processed and question exists
    is_ask_disabled = not (doc_ready and question.strip())

    if st.button("Ask Question", disabled=is_ask_disabled, type="primary", key="qa_ask_btn"):
        with st.spinner("Retrieving relevant chunks and generating grounded answer..."):
            try:
                result = answer_question(st.session_state["qa_vector_store"], question.strip())
                st.session_state["qa_last_answer"] = result.get("answer", "")
                st.session_state["qa_last_sources"] = result.get("sources", [])
                st.session_state["qa_last_question"] = question.strip()
            except ValueError as e:
                st.error(f"Configuration or validation error: {e}")
            except Exception as e:
                st.error(f"An error occurred during question answering: {e}")

    # Display Answer and Sources
    if st.session_state.get("qa_last_answer"):
        st.markdown("---")
        st.markdown("### Answer")
        st.write(st.session_state["qa_last_answer"])

        sources = st.session_state.get("qa_last_sources", [])
        if sources:
            st.markdown("### Sources")
            for idx, src in enumerate(sources, 1):
                page_label = src.get("page", "Unknown")
                source_file = os.path.basename(src.get("source", "Unknown"))
                preview = src.get("preview", "")
                st.markdown(f"Page {page_label} — {source_file}")
                if preview:
                    with st.expander(f"Preview excerpt (Page {page_label})"):
                        st.markdown(f"> {preview}")


# ==================================================
# 2. SUMMARY TAB
# ==================================================
with tab_summary:
    st.subheader("Document Summarization")
    st.markdown("Upload an academic PDF to generate a structured synthesis of its key findings and concepts.")

    summary_file = st.file_uploader(
        "Upload PDF Document to Summarize",
        type=["pdf"],
        help="Upload an academic PDF to summarize.",
        key="summary_file_uploader",
    )

    is_summarize_disabled = summary_file is None

    if st.button("Generate Summary", disabled=is_summarize_disabled, type="primary", key="summary_generate_btn"):
        with st.spinner("Summarizing academic document... (processing chunks with LLM)"):
            try:
                temp_path = _replace_session_temp_file("summary_temp_path", summary_file, prefix="summary_")
                # Attempt Phase 7 structured summary first
                try:
                    structured_summary = summarize_document_structured(temp_path)
                    st.session_state["summary_result"] = {
                        "mode": "structured",
                        "data": structured_summary,
                        "file_name": summary_file.name,
                    }
                except Exception:
                    # Fallback to Phase 5 free-text summary if structured extraction fails
                    raw_summary = summarize_document(temp_path)
                    st.session_state["summary_result"] = {
                        "mode": "raw",
                        "data": raw_summary,
                        "file_name": summary_file.name,
                    }
            except ValueError as e:
                st.error(f"Configuration or validation error: {e}")
            except Exception as e:
                st.error(f"An error occurred while generating summary: {e}")

    # Display Summary Result
    if st.session_state.get("summary_result"):
        res = st.session_state["summary_result"]
        st.markdown("---")
        st.caption(f"Summary for: **{res.get('file_name', 'Document')}**")

        if res["mode"] == "structured":
            schema_data: DocumentSummarySchema = res["data"]

            st.markdown("### Document Title")
            st.write(schema_data.document_title)

            st.markdown("### Main Topic")
            st.write(schema_data.main_topic)

            st.markdown("### Key Points")
            for pt in schema_data.key_points:
                st.markdown(f"• {pt}")

            st.markdown("### Summary")
            st.write(schema_data.summary)
        else:
            raw_data = res["data"]
            st.markdown("### Summary")
            st.write(raw_data.get("final_summary", ""))


# ==================================================
# 3. COMPARE DOCUMENTS TAB
# ==================================================
with tab_compare:
    st.subheader("Compare Academic Documents")
    st.markdown("Upload two academic PDFs to cross-analyze their main topics, key concepts, similarities, and differences.")

    col_a, col_b = st.columns(2)
    with col_a:
        file_a = st.file_uploader(
            "Upload Document A",
            type=["pdf"],
            key="compare_file_a",
        )
    with col_b:
        file_b = st.file_uploader(
            "Upload Document B",
            type=["pdf"],
            key="compare_file_b",
        )

    is_compare_disabled = (file_a is None or file_b is None)

    if st.button("Compare Documents", disabled=is_compare_disabled, type="primary", key="compare_execute_btn"):
        with st.spinner("Comparing Document A and Document B... (summarizing and cross-analyzing)"):
            try:
                temp_a = _replace_session_temp_file("compare_temp_path_a", file_a, prefix="comp_a_")
                temp_b = _replace_session_temp_file("compare_temp_path_b", file_b, prefix="comp_b_")

                # Attempt Phase 7 structured comparison first
                try:
                    structured_comp = compare_documents_structured(temp_a, temp_b)
                    st.session_state["compare_result"] = {
                        "mode": "structured",
                        "data": structured_comp,
                        "file_a_name": file_a.name,
                        "file_b_name": file_b.name,
                    }
                except Exception:
                    # Fallback to Phase 6 free-text comparison
                    raw_comp = compare_documents(temp_a, temp_b)
                    st.session_state["compare_result"] = {
                        "mode": "raw",
                        "data": raw_comp,
                        "file_a_name": file_a.name,
                        "file_b_name": file_b.name,
                    }
            except ValueError as e:
                st.error(f"Configuration or validation error: {e}")
            except Exception as e:
                st.error(f"An error occurred while comparing documents: {e}")

    # Display Comparison Result
    if st.session_state.get("compare_result"):
        res = st.session_state["compare_result"]
        st.markdown("---")
        st.caption(f"Comparison: **{res.get('file_a_name', 'Document A')}** vs **{res.get('file_b_name', 'Document B')}**")

        if res["mode"] == "structured":
            comp_data: DocumentComparisonSchema = res["data"]

            st.markdown("### Main Topic - Document A")
            st.write(comp_data.document_a_topic)

            st.markdown("### Main Topic - Document B")
            st.write(comp_data.document_b_topic)

            st.markdown("### Key Concepts")
            for concept in comp_data.key_concepts:
                st.markdown(f"• {concept}")

            st.markdown("### Similarities")
            for sim in comp_data.similarities:
                st.markdown(f"• {sim}")

            st.markdown("### Differences")
            for diff in comp_data.differences:
                st.markdown(f"• {diff}")

            st.markdown("### Key Observations")
            for obs in comp_data.key_observations:
                st.markdown(f"• {obs}")
        else:
            raw_comp = res["data"]
            st.markdown("### Comparison Result")
            st.write(raw_comp.get("comparison", ""))
