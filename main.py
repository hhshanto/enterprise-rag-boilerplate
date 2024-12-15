from data_ingestion.data_ingestion import DataIngestion
from embedding.data_embedding import DataEmbedding
from retrieval.retrieval import Retrieval
import chromadb

def main():
    # Data Ingestion
    data_ingestion = DataIngestion()
    data_ingestion.download_datasets()

    # Data Embedding
    data_embedding = DataEmbedding()
    data_embedding.embed_text_corpus()

    # Retrieval
    client = chromadb.Client()
    collection = client.get_or_create_collection("my_collection")
    retrieval = Retrieval("sentence-transformers/all-mpnet-base-v2", collection)

    # Example usage
    query = "What is the meaning of life?"
    results = retrieval.search(query)
    ranked_results = retrieval.rank_results(results)

    for result in ranked_results:
        print(f"Document: {result['document'][:100]}...")
        print(f"Distance: {result['distance']}")
        print("---")

if __name__ == "__main__":
    main()