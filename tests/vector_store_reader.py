import os
import logging
from chromadb import PersistentClient
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStoreReader:
    def __init__(self):
        self.huggingface_token = os.getenv('HuggingAccessToken')
        if not self.huggingface_token:
            raise ValueError("HuggingAccessToken not found in environment variables")
        
        # Initialize paths
        self.root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.embedding_dir = os.path.join(self.root_dir, 'data', 'embedding')
        
        # Initialize ChromaDB client
        self.client = PersistentClient(path=self.embedding_dir)
        
        # Initialize HuggingFace embedding function
        self.embedding_function = embedding_functions.HuggingFaceEmbeddingFunction(
            api_key=self.huggingface_token,
            model_name="sentence-transformers/all-mpnet-base-v2"
        )

    def get_by_ids(self, ids):
        collection = self.client.get_collection("text_embeddings")
        results = collection.get(
            ids=ids,
            include=["documents", "metadatas", "embeddings"]
        )
        return results

    def query_by_vector(self, query_vector, n_results=5):
        collection = self.client.get_collection("text_embeddings")
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        return results

    def query_by_text(self, query_text, n_results=5):
        collection = self.client.get_collection("text_embeddings", embedding_function=self.embedding_function)
        results = collection.query(
            query_texts=[query_text],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        return results

    def get_all(self):
        collection = self.client.get_collection("text_embeddings")
        results = collection.get(
            include=["documents", "metadatas", "embeddings"]
        )
        return results

    def query_with_filter(self, query_text, filter_condition, n_results=5):
        collection = self.client.get_collection("text_embeddings", embedding_function=self.embedding_function)
        results = collection.query(
            query_texts=[query_text],
            where=filter_condition,
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        return results

    def demonstrate_retrieval(self):
        # Get by IDs
        sample_ids = ["text_0", "text_1", "text_2"]
        id_results = self.get_by_ids(sample_ids)
        logger.info("Results by IDs:")
        logger.info(id_results)

        # Query by text
        query_text = "To be or not to be"
        text_results = self.query_by_text(query_text)
        logger.info(f"Results for query '{query_text}':")
        logger.info(text_results)

        # Query with filter
        filter_condition = {"source": "shakespeare"}
        filter_results = self.query_with_filter(query_text, filter_condition)
        logger.info(f"Results for query '{query_text}' with filter:")
        logger.info(filter_results)

def main():
    reader = VectorStoreReader()
    reader.demonstrate_retrieval()

if __name__ == "__main__":
    main()