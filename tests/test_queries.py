import os
import sys
import logging
from dotenv import load_dotenv

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.query_processor import QueryProcessor
from retrieval.vector_store import VectorStore
from retrieval.ranker import Ranker
from retrieval.retriever import Retriever

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def main():
    # Initialize components
    logger.info("Initializing components...")
    query_processor = QueryProcessor("sentence-transformers/all-mpnet-base-v2")
    
    # Adjust this path to where your vector store is located
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    embedding_dir = os.path.join(root_dir, 'data', 'embedding')
    
    # Initialize VectorStore with just the collection name
    vector_store = VectorStore("text_embeddings")
    
    # If needed, you can set the embedding_dir after initialization
    vector_store.embedding_dir = embedding_dir
    
    ranker = Ranker()
    retriever = Retriever(query_processor, vector_store, ranker)

    # List of queries to test
    queries = [
        "To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?",
        "What is in front of the Notre Dame Main Building?",
        "The Basilica of the Sacred heart at Notre Dame is beside to which structure?",
        "What is the Grotto at Notre Dame?",
        "What sits on top of the Main Building at Notre Dame?",
        "When did the Scholastic Magazine of Notre dame begin publishing?"
    ]

    # Test each query
    for query in queries:
        logger.info(f"\nTesting query: '{query}'")
        results = retriever.retrieve(query, k=3)  # Retrieve top 3 results
        
        if results:
            logger.info(f"Top 3 results for query: '{query}'")
            for i, result in enumerate(results, 1):
                logger.info(f"Result {i}:")
                logger.info(f"  Document: {result['document'][:200]}...")  # Show first 200 characters
                logger.info(f"  Distance: {result['distance']}")
        else:
            logger.warning(f"No results found for query: '{query}'")

if __name__ == "__main__":
    main()