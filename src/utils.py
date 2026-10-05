"""
Utilities Module
Handles: Shared helpers — environment loading, JSON extraction, text cleaning.
"""

import os
import re


from dotenv import load_dotenv
import tempfile


def load_environment():
    """Load environment variables from .env file."""
    load_dotenv()


def save_uploaded_file_to_temp(uploaded_file, prefix: str = "doc_") -> str:
    """
    Save a Streamlit UploadedFile to an OS temporary file with a .pdf extension.
    Returns the absolute path to the temporary file.
    User uploads are never saved directly into the repository.
    """
    fd, temp_path = tempfile.mkstemp(prefix=prefix, suffix=".pdf")
    with os.fdopen(fd, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return temp_path


def cleanup_temp_file(file_path: str):
    """Safely remove a temporary file if it exists."""
    if file_path and os.path.exists(file_path):
        try:
            os.remove(file_path)
        except OSError:
            pass


def extract_json_block(text: str) -> str:
    """
    Extract the last JSON code block from LLM output text.
    Reference: Update_OutputParser (1).ipynb — extract_json_block function
    """
    pattern = r'```json\s*(.*?)\s*```'
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return f"```json\n{matches[-1]}\n```"
    return text


def format_sources(sources: list) -> str:
    """Format retrieved document source dictionaries into a readable string."""
    if not sources:
        return "No sources available."
    lines = []
    for idx, src in enumerate(sources, 1):
        page = src.get("page", "Unknown")
        source_name = os.path.basename(src.get("source", "Unknown"))
        lines.append(f"Source {idx}: Page {page} -- {source_name}")
    return "\n".join(lines)
