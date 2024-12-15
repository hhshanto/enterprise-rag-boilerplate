import os
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, collection_name: str = "my_collection"):
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        embedding_dir = os.path.join(root_dir, 'data', 'embedding')
        os.makedirs(embedding_dir, exist_ok=True)

        self.client = chromadb.PersistentClient(path=embedding_dir)
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )

    def add_documents(self, documents: List[str], ids: List[str], metadatas: List[Dict] = None):
        logger.info(f"Adding {len(documents)} documents to the vector store")
        self.collection.add(
            documents=documents,
            ids=ids,
            metadatas=metadatas
        )

    def search(self, query: str, k: int = 5) -> Dict[str, Any]:
        logger.info(f"Searching for query: {query}")
        results = self.collection.query(
            query_texts=[query],
            n_results=k
        )
        return results

    def format_results(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        formatted_results = []
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                'id': results['ids'][0][i],
                'document': results['documents'][0][i],
                'distance': results['distances'][0][i]
            })
        return formatted_results

def test_vector_store():
    vector_store = VectorStore("test_collection")
    vector_store.add_documents(
        documents=["This is a test document", "Another test document", "A third test document"],
        ids=["1", "2", "3"],
        metadatas=[{"source": "test"}, {"source": "test"}, {"source": "test"}]
    )
    query = "test document"
    results = vector_store.search(query, k=2)
    logger.info("Raw search results:")
    logger.info(results)
    formatted_results = vector_store.format_results(results)
    logger.info("Formatted results:")
    for result in formatted_results:
        logger.info(result)

if __name__ == "__main__":
    test_vector_store()