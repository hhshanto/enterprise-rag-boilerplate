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

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        logger.info(f"Retrieving for query: {query}")
        processed_query = self.query_processor.process(query)
        logger.info(f"Processed query: {processed_query}")
        raw_results = self.vector_store.search(processed_query, k)
        formatted_results = self.vector_store.format_results(raw_results)
        ranked_results = self.ranker.rank(formatted_results)
        logger.info(f"Retrieved and ranked {len(ranked_results)} results")
        return ranked_results

    def demonstrate_retrieval(self, query: str, k: int = 5):
        logger.info(f"Demonstrating retrieval for query: '{query}'")
        results = self.retrieve(query, k)
        logger.info(f"Top {k} results:")
        for i, result in enumerate(results, 1):
            logger.info(f"Result {i}:")
            logger.info(f"  Document: {result['document'][:100]}...")
            logger.info(f"  Distance: {result['distance']}")
            logger.info("---")
        return results

if __name__ == "__main__":
    logger.info("Initializing components...")
    query_processor = QueryProcessor("sentence-transformers/all-mpnet-base-v2")
    vector_store = VectorStore("my_collection")
    ranker = Ranker()
    
    logger.info("Checking vector store contents...")
    collection_info = vector_store.collection.count()
    logger.info(f"Number of documents in the collection: {collection_info}")

    retriever = Retriever(query_processor, vector_store, ranker)

    # Path to the evaluation folder
    eval_folder = os.path.join('..', 'data', 'ragdata', 'evaluation')
    logger.info(f"Using evaluation folder: {eval_folder}")
    
    # Read the first 5 questions
    with open(os.path.join(eval_folder, 'questions.txt'), 'r') as f:
        questions = f.readlines()[:5]
    logger.info(f"Loaded {len(questions)} questions for demonstration")

    # Demonstrate retrieval for each question
    for question in questions:
        question = question.strip()
        results = retriever.demonstrate_retrieval(question)
        if not results:
            logger.warning(f"No results found for query: '{question}'")
        print("\n")  # Add a blank line between questions

    logger.info("Demonstration complete")