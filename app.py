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
tab_qa, tab_summary, tab_compare, tab_quiz, tab_flashcards, tab_study, tab_sources = st.tabs([
    "Q&A", "Summary", "Compare Documents", "Quiz", "Flashcards", "Study Guide", "Source Explorer"
])


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


# ==================================================
# 4. QUIZ TAB
# ==================================================
with tab_quiz:
    st.subheader("Quiz Generator")
    st.markdown("Generate a multiple-choice quiz grounded in the document.")

    quiz_file = st.file_uploader(
        "Upload PDF Document for Quiz",
        type=["pdf"],
        help="Upload a single academic PDF file.",
        key="quiz_file_uploader",
    )

    if quiz_file:
        quiz_temp_path = st.session_state.get("quiz_temp_path")
        if not quiz_temp_path or st.session_state.get("quiz_file_id") != f"{quiz_file.name}_{quiz_file.size}":
            if quiz_temp_path:
                cleanup_temp_file(quiz_temp_path)
                _registered_temp_files.discard(quiz_temp_path)

            new_path = save_uploaded_file_to_temp(quiz_file)
            _registered_temp_files.add(new_path)
            st.session_state["quiz_temp_path"] = new_path
            st.session_state["quiz_file_id"] = f"{quiz_file.name}_{quiz_file.size}"
            st.session_state["quiz_result"] = None

        col1, col2 = st.columns(2)
        with col1:
            num_questions = st.number_input("Number of questions", min_value=1, max_value=20, value=5, key="quiz_num_qs")
        with col2:
            difficulty = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"], index=1, key="quiz_diff")

        if st.button("Generate Quiz", type="primary"):
            with st.spinner("Generating quiz questions..."):
                try:
                    from src.rag_pipeline import generate_quiz
                    st.session_state["quiz_result"] = generate_quiz(
                        st.session_state["quiz_temp_path"],
                        num_questions=num_questions,
                        difficulty=difficulty
                    )
                except Exception as e:
                    st.error(f"Failed to generate quiz: {e}")

        # Display results
        quiz_result = st.session_state.get("quiz_result")
        if quiz_result and "questions" in quiz_result:
            st.success("Quiz generated successfully!")
            for idx, q in enumerate(quiz_result["questions"], 1):
                st.markdown(f"**Q{idx}: {q['question']}**")
                for opt in q['options']:
                    st.markdown(f"- {opt}")

                with st.expander("Show Answer"):
                    st.markdown(f"**Correct Answer:** {q['correct_answer']}")
                    st.markdown(f"**Explanation:** {q['explanation']}")
                    if q.get('source_page'):
                        st.caption(f"Source Page: {q['source_page']}")
                st.divider()

# ==================================================
# 5. FLASHCARDS TAB
# ==================================================
with tab_flashcards:
    st.subheader("Academic Flashcards")
    st.markdown("Generate study flashcards from the document.")

    fc_file = st.file_uploader(
        "Upload PDF Document for Flashcards",
        type=["pdf"],
        help="Upload a single academic PDF file.",
        key="fc_file_uploader",
    )

    if fc_file:
        fc_temp_path = st.session_state.get("fc_temp_path")
        if not fc_temp_path or st.session_state.get("fc_file_id") != f"{fc_file.name}_{fc_file.size}":
            if fc_temp_path:
                cleanup_temp_file(fc_temp_path)
                _registered_temp_files.discard(fc_temp_path)

            new_path = save_uploaded_file_to_temp(fc_file)
            _registered_temp_files.add(new_path)
            st.session_state["fc_temp_path"] = new_path
            st.session_state["fc_file_id"] = f"{fc_file.name}_{fc_file.size}"
            st.session_state["fc_result"] = None

        num_fc = st.number_input("Number of flashcards", min_value=1, max_value=30, value=10, key="fc_num")

        if st.button("Generate Flashcards", type="primary"):
            with st.spinner("Generating flashcards..."):
                try:
                    from src.rag_pipeline import generate_flashcards
                    st.session_state["fc_result"] = generate_flashcards(
                        st.session_state["fc_temp_path"],
                        num_flashcards=num_fc
                    )
                except Exception as e:
                    st.error(f"Failed to generate flashcards: {e}")

        # Display results
        fc_result = st.session_state.get("fc_result")
        if fc_result and "flashcards" in fc_result:
            st.success("Flashcards generated successfully!")

            # Simple UI for flashcards
            for idx, fc in enumerate(fc_result["flashcards"], 1):
                with st.expander(f"Card {idx}: {fc['front']}"):
                    st.markdown(f"**Answer:** {fc['back']}")
                    if fc.get('source_page'):
                        st.caption(f"Source Page: {fc['source_page']}")

# ==================================================
# 6. STUDY GUIDE TAB
# ==================================================
with tab_study:
    st.subheader("Study Guide")
    st.markdown("Generate a structured academic study guide from the document.")

    sg_file = st.file_uploader(
        "Upload PDF Document for Study Guide",
        type=["pdf"],
        help="Upload a single academic PDF file.",
        key="sg_file_uploader",
    )

    if sg_file:
        sg_temp_path = st.session_state.get("sg_temp_path")
        if not sg_temp_path or st.session_state.get("sg_file_id") != f"{sg_file.name}_{sg_file.size}":
            if sg_temp_path:
                cleanup_temp_file(sg_temp_path)
                _registered_temp_files.discard(sg_temp_path)

            new_path = save_uploaded_file_to_temp(sg_file)
            _registered_temp_files.add(new_path)
            st.session_state["sg_temp_path"] = new_path
            st.session_state["sg_file_id"] = f"{sg_file.name}_{sg_file.size}"
            st.session_state["sg_result"] = None

        if st.button("Generate Study Guide", type="primary"):
            with st.spinner("Analyzing document and building study guide..."):
                try:
                    from src.rag_pipeline import generate_study_guide
                    st.session_state["sg_result"] = generate_study_guide(
                        st.session_state["sg_temp_path"]
                    )
                except Exception as e:
                    st.error(f"Failed to generate study guide: {e}")

        # Display results
        sg = st.session_state.get("sg_result")
        if sg:
            st.success("Study Guide generated successfully!")

            st.markdown(f"## {sg.get('main_topic', 'Main Topic')}")

            st.markdown("### Key Concepts")
            for c in sg.get("key_concepts", []):
                st.markdown(f"- {c}")

            st.markdown("### Important Definitions")
            for d in sg.get("important_definitions", []):
                st.markdown(f"- {d}")

            st.markdown("### Key Relationships")
            for r in sg.get("key_relationships", []):
                st.markdown(f"- {r}")

            st.markdown("### Important Points")
            for p in sg.get("important_points", []):
                st.markdown(f"- {p}")

            st.markdown("### Suggested Review Topics")
            for s in sg.get("suggested_review_topics", []):
                st.markdown(f"- {s}")


# ==================================================
# 7. SOURCE EXPLORER TAB
# ==================================================
with tab_sources:
    st.subheader("Source Explorer")
    st.markdown("Understand how the AI generates its answers by inspecting the raw context retrieved from the FAISS vector database.")

    last_q = st.session_state.get("qa_last_question")
    last_sources = st.session_state.get("qa_last_sources")
    vector_store = st.session_state.get("qa_vector_store")

    if not vector_store:
        st.info("No document has been indexed in the Q&A tab yet. Upload a document there first.")
    elif not last_q or not last_sources:
        st.info("No question has been asked yet. Ask a question in the Q&A tab to see the retrieved context.")
    else:
        st.success(f"**Question:** {last_q}")
        st.markdown(f"**Final Answer:**\n> {st.session_state.get('qa_last_answer', '')}")
        st.divider()
        st.markdown("### Retrieved Chunks (Context)")
        st.markdown("The following chunks were retrieved from FAISS via similarity search and sent to the LLM as context.")

        for idx, doc_dict in enumerate(last_sources, 1):
            source_file = doc_dict.get("source", "Unknown")
            page_val = doc_dict.get("page", "Unknown")

            with st.expander(f"Chunk {idx} (Page: {page_val})"):
                st.caption(f"**Source File:** `{source_file}`")
                st.caption(f"**Retrieved Order:** #{idx} most similar")
                st.markdown("**Chunk Preview:**")
                st.code(doc_dict.get("preview", ""), language="text")
