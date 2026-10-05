import os
import sys
import unittest
from dotenv import load_dotenv

from src.document_processor import load_pdf, chunk_documents, create_vector_store, similarity_search
from src.rag_pipeline import (
    get_llm, answer_question, summarize_document, compare_documents,
    summarize_document_structured, compare_documents_structured
)
from src.schemas import DocumentSummarySchema, DocumentComparisonSchema, parse_llm_json_to_model

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



if __name__ == "__main__":
    unittest.main()
