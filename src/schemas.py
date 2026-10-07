"""
Schemas Module
Handles: Structured output definitions using Pydantic models.

Phase 7: Structured Output
- DocumentSummarySchema: Structured representation of a document summary.
- DocumentComparisonSchema: Structured representation of a two-document comparison.
- parse_llm_json_to_model: Helper to extract JSON from LLM output and validate with Pydantic.
"""

import json
import re
from typing import List, Optional
from pydantic import BaseModel, Field, ValidationError


# --- Pydantic Schemas ---

class DocumentSummarySchema(BaseModel):
    """Structured output schema for a document summary (Phase 5 + Phase 7)."""
    document_title: str = Field(description="Title or main heading of the document")
    main_topic: str = Field(description="Primary topic or subject area of the document")
    key_points: List[str] = Field(description="List of key points and findings from the document")
    summary: str = Field(description="Cohesive final summary of the document")


class DocumentComparisonSchema(BaseModel):
    """Structured output schema for a two-document comparison (Phase 6 + Phase 7)."""
    document_a_topic: str = Field(description="Main topic of Document A")
    document_b_topic: str = Field(description="Main topic of Document B")
    key_concepts: List[str] = Field(description="Key concepts and terminology from both documents")
    similarities: List[str] = Field(description="Key similarities between the two documents")
    differences: List[str] = Field(description="Key differences between the two documents")
    key_observations: List[str] = Field(description="Important observations or relationships supported by both documents")


class QuizQuestion(BaseModel):
    """Schema for a single multiple-choice quiz question."""
    question: str = Field(description="The multiple choice question")
    options: List[str] = Field(description="List of possible answer options")
    correct_answer: str = Field(description="The exact text of the correct option")
    explanation: str = Field(description="Explanation of why the answer is correct")
    source_page: Optional[str] = Field(default=None, description="Source page number or label if traceable")


class QuizSchema(BaseModel):
    """Schema for a complete generated quiz."""
    questions: List[QuizQuestion] = Field(description="List of multiple-choice questions")


class Flashcard(BaseModel):
    """Schema for a single academic flashcard."""
    front: str = Field(description="The question, concept, or term on the front of the card")
    back: str = Field(description="The answer or definition on the back of the card")
    source_page: Optional[str] = Field(default=None, description="Source page number or label if traceable")


class FlashcardListSchema(BaseModel):
    """Schema for a collection of generated flashcards."""
    flashcards: List[Flashcard] = Field(description="List of flashcards")


class StudyGuideSchema(BaseModel):
    """Schema for a comprehensive academic study guide."""
    main_topic: str = Field(description="The central topic of the document")
    key_concepts: List[str] = Field(description="List of core concepts with optional page references")
    important_definitions: List[str] = Field(description="Key definitions found in the text with optional page references")
    key_relationships: List[str] = Field(description="Important relationships or connections between concepts")
    important_points: List[str] = Field(description="Crucial takeaways or findings")
    suggested_review_topics: List[str] = Field(description="Topics the student should review further")


# --- JSON Extraction and Pydantic Parsing ---

def _extract_json_from_text(text: str) -> str:
    """
    Extract a JSON object from LLM output text.

    Tries multiple strategies:
    1. Look for a ```json ... ``` fenced code block.
    2. Look for the first { ... } block in the text.
    3. Return the raw text as a fallback.
    """
    # Strategy 1: fenced JSON code block
    pattern = r'```json\s*(.*?)\s*```'
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[-1].strip()

    # Strategy 2: find outermost { ... }
    brace_start = text.find('{')
    if brace_start != -1:
        # Find the matching closing brace
        depth = 0
        for i in range(brace_start, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    return text[brace_start:i + 1].strip()

    # Strategy 3: return raw text (will likely fail JSON parse)
    return text.strip()


def parse_llm_json_to_model(text: str, model_class):
    """
    Parse LLM text output into a validated Pydantic model.

    Args:
        text: Raw LLM output text (may contain JSON in a code block or inline).
        model_class: The Pydantic BaseModel class to validate against.

    Returns:
        A validated instance of model_class.

    Raises:
        ValueError: If JSON extraction or Pydantic validation fails.
    """
    json_str = _extract_json_from_text(text)

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Failed to parse JSON from LLM output.\n"
            f"JSON parse error: {e}\n"
            f"Extracted text: {json_str[:500]}"
        )

    try:
        return model_class.model_validate(data)
    except ValidationError as e:
        raise ValueError(
            f"Pydantic validation failed for {model_class.__name__}.\n"
            f"Validation errors: {e}\n"
            f"Parsed data: {json.dumps(data, indent=2)[:500]}"
        )
