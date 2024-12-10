import os
import unittest
from datasets import load_from_disk

class TestDataIngestion(unittest.TestCase):
    def setUp(self):
        self.data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'ragData'))

    def test_question_answer_dataset(self):
        dataset_path = os.path.join(self.data_dir, 'question_answer')
        self.assertTrue(os.path.exists(dataset_path), "Question-answer dataset not found.")
        try:
            dataset = load_from_disk(dataset_path)
            self.assertTrue(len(dataset) > 0, "Question-answer dataset is empty.")
        except Exception:
            self.fail("Failed to load question-answer dataset from disk.")

    def test_text_corpus_dataset(self):
        dataset_path = os.path.join(self.data_dir, 'text_corpus')
        self.assertTrue(os.path.exists(dataset_path), "Text-corpus dataset not found.")
        try:
            dataset = load_from_disk(dataset_path)
            self.assertTrue(len(dataset) > 0, "Text-corpus dataset is empty.")
        except Exception:
            self.fail("Failed to load text-corpus dataset from disk.")

if __name__ == '__main__':
    unittest.main()

    """
    The script defines unit tests using Python's unittest framework.
    It checks if the datasets question_answer and text_corpus exist in the specified ragData directory.
    It attempts to load each dataset from disk using load_from_disk.
    It verifies that each dataset is not empty by checking its length.
    If any of these checks fail, the test will report an error or failure, indicating an issue with data ingestion.
    
    """