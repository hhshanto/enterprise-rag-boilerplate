import sys
import os
import logging
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from loaders.azure_loader import AzureOpenAILoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_chat_completion():
    """Test chat completion functionality"""
    loader = AzureOpenAILoader()
    
    # Test basic completion
    prompt = "Explain what is RAG in 2 sentences."
    response = loader.get_completion(
        prompt=prompt,
        max_tokens=100,
        temperature=0.7
    )
    logger.info(f"\nPrompt: {prompt}")
    logger.info(f"Response: {response}")
    
    # Test with different parameters
    prompt = "Write a one-line python function to calculate fibonacci number."
    response = loader.get_completion(
        prompt=prompt,
        max_tokens=50,
        temperature=0.3  # Lower temperature for more focused coding responses
    )
    logger.info(f"\nPrompt: {prompt}")
    logger.info(f"Response: {response}")

def test_embeddings():
    """Test embedding generation functionality"""
    loader = AzureOpenAILoader()
    
    # Test single text embedding
    text = "This is a test sentence for embedding generation."
    embeddings = loader.get_embeddings([text])
    if embeddings:
        logger.info(f"\nSingle embedding dimension: {len(embeddings[0])}")
    
    # Test multiple texts embedding
    texts = [
        "First test sentence",
        "Second test sentence with different content",
        "Third test sentence with completely different information"
    ]
    embeddings = loader.get_embeddings(texts)
    if embeddings:
        logger.info(f"Multiple embeddings generated: {len(embeddings)}")
        logger.info(f"Each embedding dimension: {len(embeddings[0])}")

def test_error_handling():
    """Test error handling scenarios"""
    loader = AzureOpenAILoader()
    
    # Test with empty input
    logger.info("\nTesting empty input:")
    response = loader.get_completion("")
    logger.info(f"Empty prompt response: {response}")
    
    # Test with very long input
    logger.info("\nTesting long input:")
    long_text = "test " * 1000
    embeddings = loader.get_embeddings([long_text])
    logger.info(f"Long text embedding successful: {embeddings is not None}")

def main():
    """Run all tests"""
    logger.info("Starting Azure OpenAI Loader tests...")
    
    try:
        logger.info("\n=== Testing Chat Completion ===")
        test_chat_completion()
        
        logger.info("\n=== Testing Embeddings ===")
        test_embeddings()
        
        logger.info("\n=== Testing Error Handling ===")
        test_error_handling()
        
        logger.info("\nAll tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
