# 📚 AI Academic Document Assistant

## 1. Project Overview
The AI Academic Document Assistant is an intelligent, local-first web application designed to help students, researchers, and academics process, analyze, and synthesize large PDF documents. Leveraging Retrieval-Augmented Generation (RAG) and Large Language Models (LLMs), it enables users to seamlessly interact with complex academic material through natural language querying, automated structured summarization, and side-by-side document comparison.

## 2. Problem Statement
Reading and extracting valuable information from dense academic papers, textbooks, and lecture notes is a time-consuming and cognitively demanding process. Researchers often struggle to quickly locate specific facts, synthesize entire documents into digestible formats, or accurately identify similarities and differences between related papers. This project aims to solve this by providing an AI-powered assistant that automates extraction, synthesis, and comparison while remaining strictly grounded in the provided texts.

## 3. Features
The platform implements the following core features:
- **PDF Question Answering using RAG**: Upload an academic PDF, index it into FAISS, and ask natural language questions. The AI answers strictly using the document's context.
- **Document Summarization**: Generate cohesive, structured summaries of uploaded documents using a map-reduce chunking approach.
- **Two-Document Comparison**: Compare two documents side by side to extract main topics, key concepts, similarities, differences, and key observations.
- **Quiz Generator**: Create rigorous multiple-choice quizzes (varying difficulties) grounded entirely in the document context.
- **Academic Flashcards**: Generate study flashcards targeting key definitions and concepts.
- **Comprehensive Study Guide**: Synthesize a structured study guide with main topics, relationships, and suggested review areas.
- **Source Explorer**: View the exact document chunks and raw text retrieved from FAISS to understand the RAG context pipeline.
- **Pydantic Structured Output**: All generated summaries, comparisons, quizzes, flashcards, and study guides are strictly validated and formatted into predictable JSON structures.
- **Source/Page References**: Every feature is backed by verifiable citations, pointing to the exact page and source file.
- **Streamlit UI**: A clean, interactive, tab-based web interface.

## 4. System Workflow / Architecture
The system operates sequentially:
1. **Ingestion**: `PyPDF` loads the PDF pages.
2. **Chunking**: Text is split into overlapping chunks (1000 chars, 100 overlap).
3. **Embeddings & Vector Database**: Chunks are embedded using `sentence-transformers` and stored in an in-memory `FAISS` index.
4. **LLM Inference**: The application builds a prompt with retrieved context and executes inference locally via Ollama (`llama3.1:8b`).
5. **Structured Parsing**: Responses are validated through regex-based JSON extraction and Pydantic validation.

## 5. Technologies Used
- **UI Framework**: Streamlit
- **LLM / Inference**: Local Ollama (`llama3.1:8b`)
- **Embeddings**: Hugging Face `sentence-transformers/all-MiniLM-L6-v2`
- **Vector Database**: FAISS (in-memory CPU)
- **Document Processing**: `pypdf`, LangChain `CharacterTextSplitter`
- **Orchestration**: LangChain (LCEL)
- **Validation**: Pydantic

## 6. Project Structure
```text
AI-Academic-Document-Assistant/
├── app.py                    # Streamlit UI entry point
├── src/
│   ├── __init__.py
│   ├── document_processor.py # PDF loading, chunking, embeddings, FAISS vector store
│   ├── rag_pipeline.py       # RAG orchestration (retrieval + generation pipelines)
│   ├── chains.py             # LangChain LCEL chains and PromptTemplates
│   ├── schemas.py            # Pydantic schemas and JSON parsing
│   └── utils.py              # Environment loading and temp file management
├── data/
│   └── sample_docs/          # Test PDFs (e.g., sample.pdf, sample2.pdf)
├── test_integration.py       # Automated integration test suite
├── requirements.txt          # Python dependencies
├── .env.example              # Environment variable template
├── .gitignore                # Git ignore rules
├── PROJECT_ARCHITECTURE.md   # In-depth architectural documentation
└── README.md                 # Project documentation
```

## 7. Installation
Requires Python 3.10+.

```bash
# 1. Clone the repository
git clone <repo-url>
cd AI-Academic-Document-Assistant

# 2. Create a virtual environment
python -m venv .venv

# 3. Activate the virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

## 8. Environment Configuration
The application performs LLM inference locally using Ollama (`llama3.1:8b`). Hugging Face Inference Providers are no longer required for LLM inference, and no credits are consumed by the application.

1. Ensure [Ollama](https://ollama.com/) is installed locally and running.
2. Download the model (requires internet only for the initial download):
   ```bash
   ollama pull llama3.1:8b
   ```
3. (Optional) Set environment variables for embeddings in `.env`.
   ```bash
   cp .env.example .env
   ```
*Note: Your `.env` file is included in `.gitignore` and must never be committed to version control.*

## 9. Usage
The system supports both a user-friendly Web UI and direct CLI invocations for backend testing.

## 10. Running the Application
To launch the primary Streamlit interface, run:
```bash
streamlit run app.py
```
This will open the application in your default web browser at `http://localhost:8501`.

To run backend pipeline tests via CLI:
```bash
python -m src.rag_pipeline
```

## 11. Demo
For a quick instructor demonstration, follow this sequence in the Streamlit UI:

**Demo 1: Question Answering (RAG)**
1. Navigate to the **Q&A** tab.
2. Upload `data/sample_docs/sample.pdf`.
3. In the input box, ask: `"What is Retrieval-Augmented Generation (RAG)?"`
4. *Result*: The AI provides a grounded answer and displays the exact page/source references.

**Demo 2: Document Summarization**
1. Navigate to the **Summary** tab.
2. Upload `data/sample_docs/sample.pdf`.
3. Click **Generate Summary**.
4. *Result*: The app displays the structured Title, Main Topic, Key Points, and Final Summary.

**Demo 3: Two-Document Comparison**
1. Navigate to the **Compare Documents** tab.
2. Upload `data/sample_docs/sample.pdf` as Document A and `data/sample_docs/sample2.pdf` as Document B.
3. Click **Compare Documents**.
4. *Result*: The system extracts the topic of each document and lists Key Concepts, Similarities, Differences, and Key Observations.

## 12. Results
The application successfully prevents hallucination by strictly answering only from the provided PDF context. Pydantic validation ensures that complex multi-document summarizations and comparisons are parsed perfectly into UI-friendly formats every time. 

## 13. Testing
A comprehensive integration test suite verifies the pipeline backend (document processing, FAISS retrieval, Q&A, structured extraction). Run the suite with:
```bash
python test_integration.py
```
*Note: Executing the entire standalone test pipeline consecutively (`python -m src.rag_pipeline`) processes several large documents back-to-back. Depending on your GPU VRAM (e.g., 8GB on an RTX 4060), this may result in an `illegal memory access (status code: 500)` crash in Ollama due to memory exhaustion. It is recommended to test features individually or use the Streamlit UI.*

## 14. Limitations
- **In-Memory FAISS**: The FAISS vector database operates entirely in memory and is recreated per session. There is no persistent database across reboots.
- **Hardware Requirements**: Local LLM inference via Ollama requires sufficient system resources (e.g., a dedicated GPU or adequate RAM) to run the `llama3.1:8b` model smoothly.
- **Complex PDFs**: PDFs heavily reliant on images, complex multi-column layouts, or non-OCR scanned images may have degraded text extraction quality.

## 15. Future Improvements
- Implement a persistent vector database (e.g., ChromaDB or Pinecone) to save documents across sessions.
- Implement conversational memory to support follow-up questions within the Q&A tab.
- Support additional document formats (.docx, .txt, .epub).
