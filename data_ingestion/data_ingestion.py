import os
import logging
from datasets import load_dataset
from huggingface_hub import login
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Disable symlinks warning
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_datasets():
    try:
        # Login to Hugging Face using environment variable
        huggingface_token = os.getenv('HuggingAccessToken')
        if not huggingface_token:
            raise ValueError("HuggingAccessToken not found in environment variables")
        
        login(token=huggingface_token)
        logger.info("Successfully logged in to Hugging Face")
        
        # Create data directory
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'ragData'))
        os.makedirs(data_dir, exist_ok=True)
        
        logger.info(f"Downloading datasets to: {data_dir}")
        
        # Load and save question-answer dataset (smaller subset of SQuAD)
        logger.info("Downloading question-answer dataset...")
        question_answer_dataset = load_dataset("squad", split="train[:1000]")  # Only 1000 examples
        
        # Load and save text corpus dataset (smaller dataset)
        logger.info("Downloading text corpus dataset...")
        text_corpus_dataset = load_dataset(
            "tiny_shakespeare",  
            split="train",
            trust_remote_code=True  # Added this parameter
        )
        
        # Save datasets
        logger.info("Saving question-answer dataset...")
        question_answer_dataset.save_to_disk(os.path.join(data_dir, 'question_answer'))
        
        logger.info("Saving text corpus dataset...")
        text_corpus_dataset.save_to_disk(os.path.join(data_dir, 'text_corpus'))
        
        # Verify the saves
        logger.info("Verifying saved datasets...")
        verify_datasets(data_dir)
        
        logger.info("Dataset download and save completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error downloading datasets: {str(e)}")
        return False

def verify_datasets(data_dir):
    """Verify that datasets were saved correctly"""
    from datasets import load_from_disk
    
    # Verify question-answer dataset
    qa_path = os.path.join(data_dir, 'question_answer')
    qa_dataset = load_from_disk(qa_path)
    logger.info(f"Question-Answer dataset loaded successfully.")
    logger.info(f"Number of examples: {len(qa_dataset)}")
    
    # Show sample data
    logger.info("\nSample from question-answer dataset:")
    logger.info(qa_dataset[0])
    
    # Verify text corpus dataset
    text_path = os.path.join(data_dir, 'text_corpus')
    text_dataset = load_from_disk(text_path)
    logger.info(f"Text corpus dataset loaded successfully.")
    logger.info(f"Number of documents: {len(text_dataset)}")
    
    logger.info("\nSample from text corpus dataset:")
    logger.info(text_dataset[0])

if __name__ == "__main__":
    download_datasets()