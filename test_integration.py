import os
import sys
import unittest
from dotenv import load_dotenv

from src.document_processor import load_pdf, chunk_documents, create_vector_store, similarity_search
from src.rag_pipeline import (
    get_llm, answer_question, summarize_document, compare_documents,
    generate_quiz, generate_flashcards, generate_study_guide
)
from src.schemas import DocumentSummarySchema, DocumentComparisonSchema, QuizSchema, FlashcardListSchema, StudyGuideSchema, parse_llm_json_to_model

# Load env variables for testing
load_dotenv()

class TestPhase9Integration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pdf1 = "data/sample_docs/sample.pdf"
        cls.pdf2 = "data/sample_docs/sample2.pdf"
        cls.docs1 = load_pdf(cls.pdf1)
        cls.docs2 = load_pdf(cls.pdf2)
        cls.chunks1 = chunk_documents(cls.docs1)
        cls.chunks2 = chunk_documents(cls.docs2)
        cls.vector_store1 = create_vector_store(cls.chunks1)

    def test_a_document_processing(self):
        # Verify valid PDFs load and chunking works
        self.assertGreater(len(self.docs1), 0)
        self.assertGreater(len(self.docs2), 0)
        self.assertGreater(len(self.chunks1), 0)

        # Verify metadata
        first_chunk = self.chunks1[0]
        self.assertIn("source", first_chunk.metadata)
        self.assertIn("page", first_chunk.metadata)
        # Check human-readable page label if present or fallback
        page_label = first_chunk.metadata.get("page_label", str(first_chunk.metadata.get("page", 0) + 1))
        self.assertIsNotNone(page_label)

    def test_b_embeddings_faiss(self):
        # Verify search returns relevant chunks
        results = similarity_search(self.vector_store1, "What is Retrieval-Augmented Generation (RAG)?", k=1)
        self.assertGreater(len(results), 0)
        # Assuming the document actually contains RAG
        self.assertIn("RAG", results[0].page_content)

    def test_c_rag_qa(self):
        # 1. Valid question
        res_valid = answer_question(self.vector_store1, "What is Retrieval-Augmented Generation (RAG)?")
        self.assertIn("answer", res_valid)
        self.assertIn("sources", res_valid)
        self.assertGreater(len(res_valid["sources"]), 0)

        # Source Explorer validation: verify metadata exists on chunks
        first_source = res_valid["sources"][0]
        self.assertIn("source", first_source)
        self.assertIn("preview", first_source)

        # 2. Unanswered question
        res_invalid = answer_question(self.vector_store1, "What is the recipe for chocolate cake?")
        self.assertIn("does not contain", res_invalid["answer"].lower())

    def test_d_summarization(self):
        res = summarize_document(self.pdf1)
        self.assertIn("final_summary", res)
        self.assertGreater(len(res["final_summary"]), 0)

    def test_e_comparison(self):
        res = compare_documents(self.pdf1, self.pdf2)
        self.assertIn("comparison", res)
        self.assertGreater(len(res["comparison"]), 0)

        # Compare same document against itself (should not crash)
        res_same = compare_documents(self.pdf1, self.pdf1)
        self.assertIn("comparison", res_same)

    def test_f_pydantic_structured_output(self):
        # Valid JSON
        valid_json = '{"document_title": "Title", "main_topic": "Topic", "key_points": ["Point"], "summary": "Sum"}'
        model = parse_llm_json_to_model(valid_json, DocumentSummarySchema)
        self.assertEqual(model.document_title, "Title")

        # Missing field
        missing_json = '{"document_title": "Title", "main_topic": "Topic", "key_points": ["Point"]}'
        with self.assertRaises(ValueError):
            parse_llm_json_to_model(missing_json, DocumentSummarySchema)

        # Malformed JSON
        malformed_json = '{"document_title": "Title", "main_topic": '
        with self.assertRaises(ValueError):
            parse_llm_json_to_model(malformed_json, DocumentSummarySchema)

        # Quiz Schema Valid
        quiz_json = '{"questions": [{"question": "Q1", "options": ["A", "B", "C", "D"], "correct_answer": "A", "explanation": "Expl", "source_page": "1"}]}'
        quiz_model = parse_llm_json_to_model(quiz_json, QuizSchema)
        self.assertEqual(len(quiz_model.questions), 1)

    def test_g_quiz_generation(self):
        # We test with a small number to save token/compute time
        res = generate_quiz(self.pdf1, num_questions=2, difficulty="Easy")
        self.assertIn("questions", res)
        self.assertGreaterEqual(len(res["questions"]), 1)

        first_q = res["questions"][0]
        self.assertIn("question", first_q)
        self.assertEqual(len(first_q["options"]), 4)
        self.assertIn("correct_answer", first_q)
        self.assertIn("explanation", first_q)


    def test_h_flashcard_generation(self):
        # Flashcard Schema Valid
        fc_json = '{"flashcards": [{"front": "Front text", "back": "Back text", "source_page": "1"}]}'
        fc_model = parse_llm_json_to_model(fc_json, FlashcardListSchema)
        self.assertEqual(len(fc_model.flashcards), 1)
        self.assertEqual(fc_model.flashcards[0].front, "Front text")

        # Flashcard Generation
        res = generate_flashcards(self.pdf1, num_flashcards=2)
        self.assertIn("flashcards", res)
        self.assertGreaterEqual(len(res["flashcards"]), 1)

        first_fc = res["flashcards"][0]
        self.assertIn("front", first_fc)
        self.assertIn("back", first_fc)

    def test_i_study_guide_generation(self):
        # Study Guide Schema Valid
        sg_json = '{"main_topic": "Topic", "key_concepts": ["A"], "important_definitions": ["B"], "key_relationships": ["C"], "important_points": ["D"], "suggested_review_topics": ["E"]}'
        sg_model = parse_llm_json_to_model(sg_json, StudyGuideSchema)
        self.assertEqual(sg_model.main_topic, "Topic")

        # Study Guide Generation
        res = generate_study_guide(self.pdf1)
        self.assertIn("main_topic", res)
        self.assertIn("key_concepts", res)
        self.assertIn("important_definitions", res)
        self.assertGreater(len(res["key_concepts"]), 0)


if __name__ == "__main__":
    unittest.main()
