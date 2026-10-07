# Project Architecture — AI Academic Document Assistant

## 1. High-Level Architecture
The AI Academic Document Assistant follows a modular, single-responsibility architecture. It separates the Streamlit web interface from the backend document processing, LangChain orchestration, and Pydantic structured validation.

## 2. Components
- **app.py**: The Streamlit frontend. It manages tab navigation, session state (temporary files and analysis results), and UI interactions. It contains no direct business logic.
- **src/document_processor.py**: Handles PDF ingestion, text extraction, chunking, embeddings generation, and FAISS vector store operations.
- **src/rag_pipeline.py**: The core orchestration layer. It bridges document retrieval with LLM inference, exposing high-level functions for Q&A, Summarization, Comparison, and Academic Tools (Quiz, Flashcards, Study Guide).
- **src/chains.py**: Houses all LangChain PromptTemplates and LCEL (LangChain Expression Language) runnable sequences.
- **src/schemas.py**: Defines Pydantic models for structured outputs and implements the JSON extraction/parsing logic.
- **src/utils.py**: Helper functions for managing temporary files, extracting regex blocks, and loading environment variables.

## 3. Data Flow & Document Processing (RAG QA)
When a user asks a question, the system uses Retrieval-Augmented Generation (RAG):

```text
PDF
 ↓
Loading (PyPDFLoader)
 ↓
Chunking (CharacterTextSplitter: 1000 size, 100 overlap)
 ↓
Embeddings (HuggingFace: all-MiniLM-L6-v2)
 ↓
FAISS (In-Memory Vector Store)
 ↓
Retriever (similarity_search)
 ↓
LLM (Local Ollama → llama3.1:8b → RTX 4060 + Context + Question)
 ↓
Answer + Sources (Returned to UI)
```

## 4. Summarization Architecture
The system uses a Map-Reduce methodology for document summarization to bypass LLM context limits:

```text
PDF
 ↓
Chunking
 ↓
Chunk Summaries (Mapped across all chunks)
 ↓
Combine (All chunk summaries merged)
 ↓
Final Summary
```

## 5. Document Comparison Architecture
Comparison relies on independently summarizing both documents before running a cross-analysis chain:

```text
Document A → Summary A ┐
                       ├→ LCEL Comparison Chain → Final Comparison
Document B → Summary B ┘
```
```

## 6. Academic Tools Architecture (Quiz, Flashcards, Study Guide)
To prevent context overflow and minimize LLM token usage, the academic tools utilize selective chunk sampling or contextual aggregation:
- **Quiz & Flashcards**: The system randomly samples individual document chunks, invoking Map generation over them until the exact requested count (e.g., 5 questions) is reached. This fully preserves specific source page metadata.
- **Study Guide**: The system aggregates chunks (up to a safe token limit), annotating them with page numbers, and passes them to a single comprehensive chain.
- **Source Explorer**: This UI-only feature visualizes the exact FAISS chunks retrieved during Q&A for absolute transparency.

## 7. Pydantic Validation & Structured Outputs
To ensure reliable UI rendering, the system enforces structured outputs. Because the current LLM provider does not natively support `llm.with_structured_output()`, the project implements a robust post-processing pipeline:

```text
Raw LLM Output String
 ↓
Regex JSON Extraction (`_extract_json_from_text`)
 ↓
`json.loads()` Parsing
 ↓
Pydantic Validation (`DocumentSummarySchema` / `DocumentComparisonSchema`)
```

If validation fails, the backend gracefully falls back to displaying the raw generated text to ensure a seamless user experience.

## 7. Streamlit UI & Session State
The frontend is built with Streamlit (`app.py`). It utilizes `st.session_state` to track:
- Uploaded file paths and their FAISS indexes, preventing re-indexing on every UI rerender.
- Past Q&A questions and answers.
- Generated summaries and comparisons, enabling users to switch between tabs without losing data.

## 8. Temporary File Handling
Streamlit `UploadedFile` objects are written to the OS temporary directory via `src/utils.py`. The paths are registered in an `atexit` handler, guaranteeing that all PDF files are securely deleted from the local disk when the Streamlit server shuts down.

## 9. Error Handling
The backend implements rigorous error handling. If the local Ollama server is unavailable, `get_llm()` queries gracefully catch connection errors and report them to the user. Empty documents, missing chunks, and JSON parsing failures are identically caught and communicated without crashing the Streamlit session.
