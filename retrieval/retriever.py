from typing import List, Dict, Any
from query_processor import QueryProcessor
from vector_store import VectorStore
from ranker import Ranker
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Retriever:
    def __init__(self, query_processor: QueryProcessor, vector_store: VectorStore, ranker: Ranker):
        self.query_processor = query_processor
        self.vector_store = vector_store
        self.ranker = ranker
        logger.info("Retriever initialized")
        logger.info(f"Vector store collection count: {self.vector_store.collection.count()}")

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        logger.info(f"Retrieving for query: {query}")
        processed_query = self.query_processor.process(query)
        logger.info(f"Processed query: {processed_query}")
        
        try:
            raw_results = self.vector_store.search(processed_query, k)
            logger.debug(f"Raw results: {raw_results}")  # Changed to debug level
            
            formatted_results = self.vector_store.format_results(raw_results)
            logger.debug(f"Formatted results: {formatted_results}")  # Changed to debug level
            
            ranked_results = self.ranker.rank(formatted_results)
            logger.info(f"Retrieved and ranked {len(ranked_results)} results")
            
            return ranked_results
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            return []

    def demonstrate_retrieval(self, query: str, k: int = 5):
        logger.info(f"Demonstrating retrieval for query: '{query}'")
        results = self.retrieve(query, k)
        logger.info(f"Top {k} results:")
        for i, result in enumerate(results, 1):
            logger.info(f"Result {i}:")
            # Display only the first 100 characters of the document
            truncated_doc = result['document'][:100] + "..." if len(result['document']) > 100 else result['document']
            logger.info(f"  Document preview: {truncated_doc}")
            logger.info(f"  Distance: {result['distance']}")
            logger.info(f"  Document length: {len(result['document'])} characters")
            logger.info("---")
        return results

if __name__ == "__main__":
    logger.info("Initializing components...")
    
    # Initialize QueryProcessor
    query_processor = QueryProcessor("sentence-transformers/all-mpnet-base-v2")
    
    # Initialize VectorStore
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    embedding_dir = os.path.join(root_dir, 'data', 'embedding')
    vector_store = VectorStore("text_embeddings")
    vector_store.embedding_dir = embedding_dir
    
    # Initialize Ranker
    ranker = Ranker()
    
    # Initialize Retriever
    retriever = Retriever(query_processor, vector_store, ranker)

    # Test queries
    queries = [
        "To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?",
        "What is in front of the Notre Dame Main Building?",
        "The Basilica of the Sacred heart at Notre Dame is beside to which structure?",
        "What is the Grotto at Notre Dame?",
        "What sits on top of the Main Building at Notre Dame?",
        "When did the Scholastic Magazine of Notre dame begin publishing?"
    ]

    # Demonstrate retrieval for each query
    for query in queries:
        results = retriever.demonstrate_retrieval(query)
        if not results:
            logger.warning(f"No results found for query: '{query}'")
        print("\n")  # Add a blank line between queries

    logger.info("Demonstration complete")