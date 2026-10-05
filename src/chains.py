"""
Chains Module
Handles: LangChain prompt templates and chains for Q&A, summarization, and document comparison.
"""

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence



def get_qa_chain(llm) -> RunnableSequence:
    """Create a chain for RAG-based question answering."""
    qa_prompt_template = """You are an AI Academic Document Assistant. You answer questions based ONLY on the provided context.

Context:
{context}

Question:
{question}

Instructions:
- Use only the supplied context to answer the question.
- If the answer is not supported by the context, explicitly say "The provided context does not contain the answer to this question."
- Do not invent facts or use outside knowledge.
- Keep the answer clear and concise.

Answer:"""
    
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=qa_prompt_template
    )
    
    # Using modern LangChain LCEL
    return prompt | llm


def get_chunk_summary_chain(llm) -> RunnableSequence:
    """Create an LCEL chain for summarizing an individual academic text chunk."""
    chunk_summary_template = """You are an AI Academic Document Assistant. Generate a concise academic summary of the text chunk below.

Text Chunk:
{chunk_text}

Instructions:
- Identify the main topic and capture the key ideas.
- Preserve essential facts, concepts, and relationships between ideas when clearly present.
- Do not invent information that is not in the chunk.
- Keep the summary concise, objective, and clearly structured.

Summary:"""

    prompt = PromptTemplate(
        input_variables=["chunk_text"],
        template=chunk_summary_template
    )
    return prompt | llm


def get_combine_summary_chain(llm) -> RunnableSequence:
    """Create an LCEL chain for combining section summaries into a cohesive final document summary."""
    combine_summary_template = """You are an AI Academic Document Assistant. You are provided with summaries of different sections of an academic document. Combine them into one coherent, unified final document summary.

Section Summaries:
{chunk_summaries}

Instructions:
- Synthesize the section summaries into a cohesive, well-organized final summary.
- Remove unnecessary repetition across sections while retaining critical concepts and findings.
- Ensure the summary is clear, readable, and academically sound.
- Remain strictly grounded only in the provided document content without introducing outside claims.

Final Summary:"""

    prompt = PromptTemplate(
        input_variables=["chunk_summaries"],
        template=combine_summary_template
    )
    return prompt | llm


def get_summary_chain(llm) -> RunnableSequence:
    """
    Create a chain for document summarization.
    Combines section summaries into a final document summary.
    """
    return get_combine_summary_chain(llm)


def get_compare_chain(llm) -> RunnableSequence:
    """Create an LCEL chain for comparing two academic documents based on their contents or summaries."""
    compare_prompt_template = """You are an AI Academic Document Assistant. You are provided with information from two academic documents: Document A and Document B.

Document A:
{doc_a_content}

Document B:
{doc_b_content}

Instructions:
- Base your analysis ONLY on the provided contents of Document A and Document B. Do not invent facts or use outside knowledge.
- Keep each section concise, focused, and complete.
- Generate a structured academic comparison using EXACTLY the following format:

Main Topic - Document A:
[Concise description of the central subject and focus of Document A]

Main Topic - Document B:
[Concise description of the central subject and focus of Document B]

Key Concepts:
[Bullet points of essential concepts and terminology from Document A and Document B]

Similarities:
[Concise analysis of key similarities and shared themes between both documents]

Differences:
[Concise analysis of key differences, contrasting paradigms, and distinct methodologies]

Key Observations:
[Important takeaways or relationships supported by both documents]

Comparison:"""

    prompt = PromptTemplate(
        input_variables=["doc_a_content", "doc_b_content"],
        template=compare_prompt_template
    )
    return prompt | llm


# --- Structured Output Chains (Phase 7) ---

def get_structured_summary_chain(llm) -> RunnableSequence:
    """
    Create an LCEL chain that converts a free-text document summary into
    a JSON object matching the DocumentSummarySchema.
    """
    structured_summary_template = """You are an AI Academic Document Assistant. You are given a document summary. Extract the structured information and return ONLY a valid JSON object with no additional text.

Document Summary:
{summary_text}

Return a JSON object with exactly these fields:
- "document_title": string (the title or main heading of the document)
- "main_topic": string (the primary topic or subject area)
- "key_points": list of strings (key findings and important points)
- "summary": string (a cohesive final summary paragraph)

Important:
- Return ONLY the JSON object, no extra text before or after.
- Do not invent information not present in the summary.
- key_points must be a JSON array of strings.

JSON:"""

    prompt = PromptTemplate(
        input_variables=["summary_text"],
        template=structured_summary_template
    )
    return prompt | llm


def get_structured_comparison_chain(llm) -> RunnableSequence:
    """
    Create an LCEL chain that converts a free-text document comparison into
    a JSON object matching the DocumentComparisonSchema.
    """
    structured_comparison_template = """You are an AI Academic Document Assistant. You are given a comparison of two academic documents. Extract the structured information and return ONLY a valid JSON object with no additional text.

Document Comparison:
{comparison_text}

Return a JSON object with exactly these fields:
- "document_a_topic": string (main topic of Document A)
- "document_b_topic": string (main topic of Document B)
- "key_concepts": list of strings (key concepts from both documents)
- "similarities": list of strings (similarities between the documents)
- "differences": list of strings (differences between the documents)
- "key_observations": list of strings (important observations supported by both documents)

Important:
- Return ONLY the JSON object, no extra text before or after.
- Do not invent information not present in the comparison.
- All list fields must be JSON arrays of strings.

JSON:"""

    prompt = PromptTemplate(
        input_variables=["comparison_text"],
        template=structured_comparison_template
    )
    return prompt | llm

