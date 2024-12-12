import os
import logging
from datasets import load_from_disk
from chromadb import Client, Settings
from chromadb.utils import embedding_functions
import chromadb
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
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
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=self.embedding_dir)
        
        # Initialize HuggingFace embedding function
        self.embedding_function = embedding_functions.HuggingFaceEmbeddingFunction(
            api_key=self.huggingface_token,
            model_name="sentence-transformers/all-mpnet-base-v2"
        )

    def embed_text_corpus(self):
        """Embed the text corpus dataset"""
        logger.info("Loading text corpus dataset...")
        text_dataset = load_from_disk(os.path.join(self.data_dir, 'text_corpus'))
        
        # Create or get collection for text corpus
        text_collection = self.client.get_or_create_collection(
            name="text_embeddings",
            embedding_function=self.embedding_function
        )
        
        logger.info("Embedding text corpus...")
        batch_size = 100
        for i in range(0, len(text_dataset), batch_size):
            batch = text_dataset[i:i + batch_size]
            
            documents = []
            metadatas = []
            ids = []
            
            for j, item in enumerate(batch):
                # The item is the text directly, no need to access a 'text' field
                documents.append(str(item))  # Convert to string to ensure compatibility
                metadatas.append({"source": "shakespeare"})
                ids.append(f"text_{i+j}")
            
            # Add batch to collection
            text_collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Embedded {i + len(batch)}/{len(text_dataset)} text documents")
        
        logger.info(f"Completed embedding {len(text_dataset)} text documents")

def main():
    embedder = DataEmbedding()
    embedder.embed_text_corpus()

if __name__ == "__main__":
    main()