import os
import unittest
import chromadb
import shutil
import logging
from dotenv import load_dotenv
from embedding.data_embedding import DataEmbedding
from datasets import load_from_disk

# Disable ChromaDB logging during tests
logging.getLogger('chromadb').setLevel(logging.ERROR)

class TestDataEmbedding(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment once before all test methods"""
        # Load environment variables
        load_dotenv()
        
        # Clean up any existing embedding directory
        cls.root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        cls.embedding_dir = os.path.join(cls.root_dir, 'data', 'embedding')
        if os.path.exists(cls.embedding_dir):
            shutil.rmtree(cls.embedding_dir)
        
        # Initialize the embedder
        cls.data_embedder = DataEmbedding()
        cls.data_dir = cls.data_embedder.data_dir
        
        # Run embedding process
        cls.data_embedder.embed_text_corpus()

    def setUp(self):
        """Set up for each test"""
        self.client = self.data_embedder.client

    def test_initialization(self):
        """Test if the DataEmbedding class initializes correctly"""
        self.assertIsNotNone(self.data_embedder.huggingface_token)
        self.assertIsNotNone(self.data_embedder.client)
        self.assertIsNotNone(self.data_embedder.embedding_function)
        # Check if it's using the correct model type without accessing model_name
        self.assertTrue(
            isinstance(self.data_embedder.embedding_function, 
                      chromadb.utils.embedding_functions.HuggingFaceEmbeddingFunction)
        )

    def test_directory_structure(self):
        """Test if all required directories exist"""
        self.assertTrue(os.path.exists(self.root_dir), "Root directory not found")
        self.assertTrue(os.path.exists(self.data_dir), "Data directory not found")
        self.assertTrue(os.path.exists(self.embedding_dir), "Embedding directory not found")

    def test_data_loading(self):
        """Test if the text corpus can be loaded"""
        text_dataset = load_from_disk(os.path.join(self.data_dir, 'text_corpus'))
        self.assertIsNotNone(text_dataset)
        self.assertGreater(len(text_dataset), 0)

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
        # Access the first metadata item correctly
        self.assertEqual(results['metadatas'][0][0]['source'], "shakespeare")

    def test_batch_processing(self):
        """Test if batch processing works correctly"""
        collection = self.client.get_collection(
            name="text_embeddings",
            embedding_function=self.data_embedder.embedding_function
        )
        
        # Test multiple queries at once
        results = collection.query(
            query_texts=["love", "hate", "joy"],
            n_results=1
        )
        self.assertEqual(len(results['ids']), 3)
        self.assertEqual(len(results['documents']), 3)
        self.assertEqual(len(results['metadatas']), 3)

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
        
        # Test semantic similarity
        results = collection.query(
            query_texts=["love"],
            n_results=1
        )
        
        # Verify we get results
        self.assertTrue(len(results['documents']) > 0)
        self.assertTrue(isinstance(results['documents'][0][0], str))

    def test_error_handling(self):
        """Test error handling for invalid queries"""
        collection = self.client.get_collection(
            name="text_embeddings",
            embedding_function=self.data_embedder.embedding_function
        )
        
        # Test empty query
        with self.assertRaises(Exception):
            collection.query(query_texts=[], n_results=1)

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests"""
        try:
            # Clean up the embedding directory
            if os.path.exists(cls.embedding_dir):
                shutil.rmtree(cls.embedding_dir)
        except Exception as e:
            print(f"Warning: Failed to clean up embedding directory: {e}")

if __name__ == '__main__':
    unittest.main(verbosity=2)