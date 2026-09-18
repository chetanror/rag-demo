import unittest

from rag_engine import RagEngine, default_documents


class RagEngineTests(unittest.TestCase):
    def setUp(self):
        self.engine = RagEngine(default_documents())

    def test_retrieves_relevant_source(self):
        results = self.engine.retrieve("How does Atlas index uploaded files?")
        self.assertTrue(results)
        self.assertIn("TF-IDF", results[0].chunk.text)

    def test_answer_contains_citation(self):
        answer, results = self.engine.answer("What happens when no passage is relevant?")
        self.assertTrue(results)
        self.assertIn("[1]", answer)
        self.assertIn("does not know", answer)

    def test_unknown_question_is_grounded(self):
        answer, results = self.engine.answer("What is the launch date of Mars?")
        self.assertEqual(results, [])
        self.assertIn("could not find", answer)


if __name__ == "__main__":
    unittest.main()