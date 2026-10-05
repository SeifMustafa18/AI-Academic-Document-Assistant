"""
Document Processor Module
Handles: PDF loading, text extraction, chunking, embeddings, and vector store creation.
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


# --- PDF Loading & Text Extraction (Phase 2) ---
# Style reference: reference/update-rag.ipynb
def load_pdf(file_path: str):
    """
    Load a PDF file and return a list of Document objects (one per page).
    
    Uses LangChain's PyPDFLoader to extract text along with page metadata.
    """
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents


# --- Text Chunking (Phase 2) ---
# Style reference: reference/update-rag.ipynb — chunk_size=1000, chunk_overlap=100
def chunk_documents(documents, chunk_size: int = 1000, chunk_overlap: int = 100):
    """
    Split documents into smaller chunks for embedding and retrieval.
    
    Uses CharacterTextSplitter to split text into chunks of specified size
    with overlapping characters between consecutive chunks to preserve context.
    """
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(documents)
    return chunks


# --- Embeddings & Vector Store (Phase 3) ---
# Style reference: reference/update-rag.ipynb — model: sentence-transformers/all-MiniLM-L6-v2
def create_vector_store(chunks, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Create an in-memory FAISS vector store from document chunks.
    
    Uses HuggingFaceEmbeddings with the specified model (sentence-transformers/all-MiniLM-L6-v2)
    to convert chunk text into dense vector embeddings, then stores them in a FAISS index.
    
    Args:
        chunks: List of Document chunks to embed.
        embedding_model_name: HuggingFace model identifier.
        
    Returns:
        FAISS vector store instance.
    """
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store


def similarity_search(vector_store, query: str, k: int = 3):
    """
    Retrieve the top-k most relevant Document chunks for a given query.
    
    Args:
        vector_store: FAISS vector store instance.
        query: The user search query string.
        k: Number of relevant chunks to retrieve (default: 3).
        
    Returns:
        List of the top-k Document chunks matching the query.
    """
    results = vector_store.similarity_search(query, k=k)
    return results


# --- Phase 2 Standalone Test Function ---
def test_document_processing(file_path: str) -> dict:
    """
    Test and demonstrate the Phase 2 ingestion and chunking pipeline on a PDF file.
    
    Reports:
    - Number of pages / documents loaded
    - Number of chunks created
    - First chunk preview
    - Last chunk preview
    
    Returns a dictionary containing summary metrics, documents, and chunks.
    """
    import os

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    print("=" * 60)
    print("  AI Academic Document Assistant — Phase 2 Demo")
    print("=" * 60)
    print(f"Loading PDF from: {file_path}")

    # 1. Load PDF
    documents = load_pdf(file_path)
    num_pages = len(documents)
    print(f"\n[1] Document Loading:")
    print(f"    - Pages/Documents loaded: {num_pages}")

    # 2. Chunk documents
    chunks = chunk_documents(documents, chunk_size=1000, chunk_overlap=100)
    num_chunks = len(chunks)
    print(f"\n[2] Text Chunking (chunk_size=1000, chunk_overlap=100):")
    print(f"    - Total chunks created: {num_chunks}")

    # 3. Previews
    first_preview = ""
    last_preview = ""
    if chunks:
        first_chunk = chunks[0]
        last_chunk = chunks[-1]

        first_preview = first_chunk.page_content.strip()
        last_preview = last_chunk.page_content.strip()

        print("\n[3] First Chunk Preview:")
        print("-" * 40)
        print(first_preview[:300] + ("..." if len(first_preview) > 300 else ""))
        print(f"    [Source Metadata: {first_chunk.metadata}]")

        print("\n[4] Last Chunk Preview:")
        print("-" * 40)
        print(last_preview[:300] + ("..." if len(last_preview) > 300 else ""))
        print(f"    [Source Metadata: {last_chunk.metadata}]")
    else:
        print("\n[!] Warning: No chunks were created.")

    print("\n" + "=" * 60)
    print("Phase 2 test completed successfully!")
    print("=" * 60)

    return {
        "num_pages": num_pages,
        "num_chunks": num_chunks,
        "first_preview": first_preview,
        "last_preview": last_preview,
        "documents": documents,
        "chunks": chunks,
    }


# --- Phase 3 Standalone Test Flow ---
def test_phase3_embeddings_and_search(
    file_path: str = "data/sample_docs/sample.pdf",
    query: str = "What is Retrieval-Augmented Generation (RAG)?",
    k: int = 3,
) -> dict:
    """
    Test flow for Phase 3:
    1. Loads sample PDF
    2. Chunks it
    3. Builds in-memory FAISS vector store with HuggingFace embeddings
    4. Executes similarity search for an example query
    5. Prints chunk count, query, retrieved chunk count, metadata, and previews
    """
    import os

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    print("=" * 65)
    print("  AI Academic Document Assistant — Phase 3 (Embeddings + FAISS)")
    print("=" * 65)
    print(f"Loading document from: {file_path}")

    # 1. Load PDF
    documents = load_pdf(file_path)
    print(f"Pages/documents loaded: {len(documents)}")

    # 2. Chunk documents
    chunks = chunk_documents(documents, chunk_size=1000, chunk_overlap=100)
    print(f"Number of chunks: {len(chunks)}")

    # 3. Create FAISS vector store
    print("\nCreating FAISS vector store with sentence-transformers/all-MiniLM-L6-v2...")
    vector_store = create_vector_store(chunks)
    print("Vector store created successfully in memory.")

    # 4. Run similarity search with example query
    print(f"\nRunning similarity search...")
    print(f"Query: \"{query}\"")
    retrieved_chunks = similarity_search(vector_store, query=query, k=k)
    print(f"Number of retrieved chunks: {len(retrieved_chunks)}")

    # 5. Print retrieved chunks with metadata and short preview
    print("\n" + "-" * 65)
    print("  Retrieved Chunks Details:")
    print("-" * 65)
    for idx, chunk in enumerate(retrieved_chunks, 1):
        source = chunk.metadata.get("source", "Unknown")
        page = chunk.metadata.get("page", "Unknown")
        total_pages = chunk.metadata.get("total_pages", "Unknown")
        print(f"\n[Retrieved Chunk {idx}]")
        print(f"Page / Source Metadata: Page {page} of {total_pages} (Source: {source})")
        print(f"Full Metadata: {chunk.metadata}")
        preview_text = chunk.page_content.strip()
        if len(preview_text) > 250:
            preview_text = preview_text[:250] + "..."
        print(f"Text Preview:\n{preview_text}")

    print("\n" + "=" * 65)
    print("Phase 3 test completed successfully!")
    print("=" * 65)

    return {
        "num_chunks": len(chunks),
        "query": query,
        "num_retrieved": len(retrieved_chunks),
        "retrieved_chunks": retrieved_chunks,
        "vector_store": vector_store,
    }


if __name__ == "__main__":
    import os
    import sys

    # Default to sample PDF if no argument is passed
    default_sample = os.path.join("data", "sample_docs", "sample.pdf")
    pdf_target = sys.argv[1] if len(sys.argv) > 1 else default_sample
    query_target = sys.argv[2] if len(sys.argv) > 2 else "What is Retrieval-Augmented Generation (RAG)?"

    if not os.path.exists(pdf_target):
        print(f"Error: PDF file '{pdf_target}' not found.")
        print("Usage: python src/document_processor.py [path_to_pdf] [optional_query]")
        sys.exit(1)

    test_phase3_embeddings_and_search(pdf_target, query=query_target, k=3)
