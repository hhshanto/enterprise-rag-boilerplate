import os
import logging
from datasets import load_from_disk
from chromadb import Client, Settings
from chromadb.utils import embedding_functions
import chromadb
from dotenv import load_dotenv
import textwrap
import random
import time
from tenacity import retry, stop_after_attempt, wait_exponential

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataEmbedding:
    def __init__(self):
        self.huggingface_token = os.getenv('HuggingAccessToken')
        if not self.huggingface_token:
            raise ValueError("HuggingAccessToken not found in environment variables")
        
        # Initialize paths
        self.root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.data_dir = os.path.join(self.root_dir, 'data', 'ragData')
        self.embedding_dir = os.path.join(self.root_dir, 'data', 'embedding')
        
        # Create embedding directory if it doesn't exist
        os.makedirs(self.embedding_dir, exist_ok=True)
        
        # Initialize ChromaDB client with increased timeout
        self.client = chromadb.PersistentClient(path=self.embedding_dir, settings=Settings(anonymized_telemetry=False, allow_reset=True))
        
        # Initialize HuggingFace embedding function
        self.embedding_function = embedding_functions.HuggingFaceEmbeddingFunction(
            api_key=self.huggingface_token,
            model_name="sentence-transformers/all-mpnet-base-v2"
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def add_to_collection(self, collection, documents, metadatas, ids):
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

    def embed_text_corpus(self):
        """Embed the text corpus dataset"""
        logger.info("Starting text corpus embedding process...")
        text_dataset = load_from_disk(os.path.join(self.data_dir, 'text_corpus'))
        
        text_collection = self.client.get_or_create_collection(
            name="text_embeddings",
            embedding_function=self.embedding_function
        )
        
        all_text = text_dataset[0]['text']
        chunks = textwrap.wrap(all_text, width=11000, break_long_words=False, replace_whitespace=False)
        
        total_chunks = len(chunks)
        total_processed = 0
        batch_size = 20

        logger.info(f"Starting embedding of {total_chunks} text chunks...")

        try:
            for batch_num, i in enumerate(range(0, total_chunks, batch_size)):
                batch = chunks[i:i + batch_size]
                
                documents = []
                metadatas = []
                ids = []
                
                for j, chunk in enumerate(batch):
                    documents.append(chunk)
                    metadatas.append({"source": "shakespeare", "chunk_id": i+j})
                    ids.append(f"text_{i+j}")
                
                try:
                    self.add_to_collection(text_collection, documents, metadatas, ids)
                    total_processed += len(batch)
                    logger.info(f"Batch {batch_num + 1}: Embedded {total_processed}/{total_chunks} text chunks")
                except Exception as e:
                    logger.error(f"Failed to embed batch {batch_num + 1} after retries: {e}")
                
                time.sleep(1)  # Add a small delay between batches
            
            logger.info(f"Completed embedding all {total_processed}/{total_chunks} text chunks")
        except Exception as e:
            logger.error(f"Unexpected error during embedding process: {e}")
        finally:
            logger.info("Text corpus embedding process finished.")

        # Add this line to show the final count in the collection
        logger.info(f"Total embeddings in collection: {text_collection.count()}")

    def show_sample_embeddings(self, n=5):
        """Retrieve and display a sample of embedded data from the vector store"""
        text_collection = self.client.get_collection("text_embeddings")
        
        # Get total count of embeddings
        total_count = text_collection.count()
        logger.info(f"Total number of embeddings in the collection: {total_count}")

        # Get random sample of IDs
        all_ids = [f"text_{i}" for i in range(total_count)]
        sample_ids = random.sample(all_ids, min(n, total_count))

        # Retrieve sample embeddings
        results = text_collection.get(
            ids=sample_ids,
            include=["documents", "metadatas", "embeddings"]
        )

        logger.info(f"\nShowing {len(sample_ids)} sample embeddings:")
        for i, (doc, metadata, embedding) in enumerate(zip(results['documents'], results['metadatas'], results['embeddings'])):
            logger.info(f"\nSample {i+1}:")
            logger.info(f"Document: {doc[:100]}...")  # Show first 100 characters
            logger.info(f"Metadata: {metadata}")
            logger.info(f"Embedding (first 5 dimensions): {embedding[:5]}...")

def main():
    embedder = DataEmbedding()
    embedder.embed_text_corpus()
    embedder.show_sample_embeddings()

if __name__ == "__main__":
    main()