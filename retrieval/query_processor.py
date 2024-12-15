from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryProcessor:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def process(self, query: str) -> str:
        logger.info(f"Processing query: {query}")
        return query  # Return the original query string

    def get_embedding(self, query: str):
        logger.info(f"Getting embedding for query: {query}")
        return self.model.encode(query)

def test_query_processor():
    processor = QueryProcessor("sentence-transformers/all-mpnet-base-v2")
    query = "Test query"
    processed_query = processor.process(query)
    embedding = processor.get_embedding(query)
    logger.info(f"Processed query: {processed_query}")
    logger.info(f"Embedding shape: {len(embedding)}")

if __name__ == "__main__":
    test_query_processor()