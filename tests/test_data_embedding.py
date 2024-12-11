import os
import unittest
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
import logging

# Disable ChromaDB logging to keep test output clean
logging.getLogger('chromadb').setLevel(logging.ERROR)

class TestDataEmbedding(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test method"""
        load_dotenv()
        
        # Initialize paths
        self.root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.embedding_dir = os.path.join(self.root_dir, 'data', 'embedding')
        self.huggingface_token = os.getenv('HuggingAccessToken')
        
        if not self.huggingface_token:
            raise ValueError("HuggingAccessToken not found in environment variables")
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=self.embedding_dir)
        
        # Initialize embedding function
        self.embedding_function = embedding_functions.HuggingFaceEmbeddingFunction(
            api_key=self.huggingface_token,
            model_name="sentence-transformers/all-mpnet-base-v2"
        )

    def test_embedding_directory_exists(self):
        """Test if embedding directory exists"""
        self.assertTrue(
            os.path.exists(self.embedding_dir), 
            "Embedding directory not found"
        )

    def test_text_collection_exists(self):
        """Test if text embeddings collection exists"""
        try:
            collection = self.client.get_collection("text_embeddings")
            self.assertIsNotNone(collection, "Text embeddings collection not found")
        except Exception as e:
            self.fail(f"Failed to get text embeddings collection: {str(e)}")

    def test_text_collection_not_empty(self):
        """Test if text embeddings collection contains documents"""
        collection = self.client.get_collection("text_embeddings")
        result = collection.count()
        self.assertGreater(result, 0, "Text embeddings collection is empty")

    def test_text_embedding_query(self):
        """Test if text embeddings can be queried"""
        collection = self.client.get_collection("text_embeddings")
        
        # Test simple query
        results = collection.query(
            query_texts=["Shakespeare"],
            n_results=1
        )
        
        self.assertTrue(len(results['ids']) > 0, "No results found in text embeddings")
        self.assertTrue(len(results['documents']) > 0, "No documents returned in query results")
        self.assertTrue(len(results['metadatas']) > 0, "No metadata returned in query results")

    def test_metadata_content(self):
        """Test if metadata contains expected fields"""
        collection = self.client.get_collection("text_embeddings")
        results = collection.query(
            query_texts=["test"],
            n_results=1
        )
        
        self.assertTrue(results['metadatas'], "No metadata found")
        self.assertEqual(
            results['metadatas'][0].get('source'), 
            "shakespeare",
            "Incorrect or missing source in metadata"
        )

    def test_embedding_consistency(self):
        """Test if multiple queries for the same text return consistent results"""
        collection = self.client.get_collection("text_embeddings")
        
        # Perform same query twice
        query_text = "test"
        results1 = collection.query(query_texts=[query_text], n_results=1)
        results2 = collection.query(query_texts=[query_text], n_results=1)
        
        self.assertEqual(
            results1['ids'], 
            results2['ids'], 
            "Inconsistent results for same query"
        )

if __name__ == '__main__':
    unittest.main()