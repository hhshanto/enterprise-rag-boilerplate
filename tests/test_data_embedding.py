import os
import unittest
import sys
import chromadb
import shutil
import logging
from dotenv import load_dotenv
from datasets import Dataset, load_from_disk
from chromadb.config import Settings

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from embedding.data_embedding import DataEmbedding

# Disable ChromaDB logging during tests
logging.getLogger('chromadb').setLevel(logging.ERROR)

class TestDataEmbedding(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment once before all test methods"""
        # Load environment variables
        load_dotenv()
        
        # Set up test directories
        cls.root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        cls.test_embedding_dir = os.path.join(cls.root_dir, 'data', 'test_embedding')
        cls.test_data_dir = os.path.join(cls.root_dir, 'data', 'test_ragData')
        
        # Ensure test directories exist
        os.makedirs(cls.test_embedding_dir, exist_ok=True)
        os.makedirs(cls.test_data_dir, exist_ok=True)
        
        # Create a test-specific DataEmbedding instance
        cls.data_embedder = DataEmbedding()
        cls.data_embedder.embedding_dir = cls.test_embedding_dir
        cls.data_embedder.data_dir = cls.test_data_dir
        cls.data_embedder.client = chromadb.PersistentClient(
            path=cls.test_embedding_dir, 
            settings=Settings(anonymized_telemetry=False, allow_reset=True)
        )
        
        # Prepare test data
        cls.prepare_test_data()
        
        # Run embedding process
        cls.data_embedder.embed_text_corpus()

    @classmethod
    def prepare_test_data(cls):
        # Create a small test dataset
        test_text = "This is a test document for embedding. It contains multiple sentences to simulate real data."
        test_dataset = {'text': [test_text]}

        # Create the test_ragData directory if it doesn't exist
        os.makedirs(cls.test_data_dir, exist_ok=True)

        # Create the text_corpus directory inside test_ragData
        text_corpus_dir = os.path.join(cls.test_data_dir, 'text_corpus')
        os.makedirs(text_corpus_dir, exist_ok=True)

        # Save the test dataset in the text_corpus directory
        dataset = Dataset.from_dict(test_dataset)
        dataset.save_to_disk(text_corpus_dir)

    def setUp(self):
        """Set up for each test"""
        self.client = self.data_embedder.client

    def test_initialization(self):
        """Test if the DataEmbedding class initializes correctly"""
        self.assertIsNotNone(self.data_embedder.huggingface_token)
        self.assertIsNotNone(self.data_embedder.client)
        self.assertIsNotNone(self.data_embedder.embedding_function)
        self.assertTrue(
            isinstance(self.data_embedder.embedding_function, 
                       chromadb.utils.embedding_functions.HuggingFaceEmbeddingFunction)
        )

    def test_directory_structure(self):
        """Test if all required directories exist"""
        self.assertTrue(os.path.exists(self.root_dir), "Root directory not found")
        self.assertTrue(os.path.exists(self.test_data_dir), "Test data directory not found")
        self.assertTrue(os.path.exists(self.test_embedding_dir), "Test embedding directory not found")

    def test_data_loading(self):
        """Test if the text corpus can be loaded"""
        text_corpus_dir = os.path.join(self.test_data_dir, 'text_corpus')
        self.assertTrue(os.path.exists(text_corpus_dir), "Text corpus directory not found")
        loaded_dataset = load_from_disk(text_corpus_dir)
        self.assertGreater(len(loaded_dataset), 0, "Test dataset is empty")

    def test_collection_creation(self):
        """Test if ChromaDB collection is created correctly"""
        collections = self.client.list_collections()
        collection_names = [col.name for col in collections]
        self.assertIn("text_embeddings", collection_names)

    def test_embedding_storage(self):
        """Test if embeddings are stored correctly"""
        collection = self.client.get_collection(
            name="text_embeddings",
            embedding_function=self.data_embedder.embedding_function
        )
        count = collection.count()
        self.assertGreater(count, 0, "No embeddings stored in collection")

    def test_metadata_structure(self):
        """Test if metadata is stored correctly"""
        collection = self.client.get_collection(
            name="text_embeddings",
            embedding_function=self.data_embedder.embedding_function
        )
        results = collection.query(
            query_texts=["test"],
            n_results=1
        )
        self.assertTrue(len(results['metadatas']) > 0)
        self.assertEqual(results['metadatas'][0][0]['source'], "shakespeare")

    def test_id_format(self):
        """Test if document IDs are formatted correctly"""
        collection = self.client.get_collection(
            name="text_embeddings",
            embedding_function=self.data_embedder.embedding_function
        )
        results = collection.query(
            query_texts=["test"],
            n_results=1
        )
        self.assertRegex(results['ids'][0][0], r'^text_\d+$')

    def test_semantic_search(self):
        """Test if semantic search returns relevant results"""
        collection = self.client.get_collection(
            name="text_embeddings",
            embedding_function=self.data_embedder.embedding_function
        )
        
        results = collection.query(
            query_texts=["test"],
            n_results=1
        )
        
        self.assertTrue(len(results['documents']) > 0)
        self.assertTrue(isinstance(results['documents'][0][0], str))

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests"""
        try:
            if os.path.exists(cls.test_embedding_dir):
                shutil.rmtree(cls.test_embedding_dir)
            if os.path.exists(cls.test_data_dir):
                shutil.rmtree(cls.test_data_dir)
        except Exception as e:
            print(f"Warning: Failed to clean up test directories: {e}")

if __name__ == '__main__':
    unittest.main(verbosity=2)