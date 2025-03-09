import os
from typing import Optional, List
import logging
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AzureOpenAILoader:
    """Azure OpenAI loader for managing API connections and embeddings.
    
    This class provides centralized management of Azure OpenAI services including
    chat completions and embeddings functionality.
    """
    
    def __init__(self):
        """Initialize Azure OpenAI loader with environment variables."""
        self.api_key = os.getenv('AZURE_OPENAI_API_KEY')
        self.api_version = os.getenv('AZURE_OPENAI_API_VERSION')
        self.azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        self.completion_deployment = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')
        self.embedding_deployment = os.getenv('AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME')
        
        if not all([self.api_key, self.api_version, self.azure_endpoint, 
                   self.completion_deployment, self.embedding_deployment]):
            raise ValueError("Missing required Azure OpenAI configuration")
        
        # Initialize Azure OpenAI client
        self.client = AzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.azure_endpoint
        )
            
    def get_completion(self, prompt: str, **kwargs) -> Optional[str]:
        """Get completion from Azure OpenAI using chat completions API.
        
        Args:
            prompt (str): The prompt to send to the model
            **kwargs: Additional arguments for the chat completion API
            
        Returns:
            Optional[str]: Completed text if successful, None otherwise
        """
        try:
            # Set defaults if not specified
            if 'max_tokens' not in kwargs:
                kwargs['max_tokens'] = 100
            if 'temperature' not in kwargs:
                kwargs['temperature'] = 0.7
                
            # Use chat completions API for gpt models
            response = self.client.chat.completions.create(
                model=self.completion_deployment,
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error getting completion: {str(e)}")
            return None
            
    def get_embeddings(self, texts: List[str]) -> Optional[List[List[float]]]:
        """Get embeddings from Azure OpenAI.
        
        Args:
            texts (List[str]): List of texts to get embeddings for
            
        Returns:
            Optional[List[List[float]]]: List of embeddings if successful, None otherwise
        """
        try:
            embeddings = []
            for text in texts:
                response = self.client.embeddings.create(
                    input=text,
                    model=self.embedding_deployment
                )
                embeddings.append(response.data[0].embedding)
            return embeddings
        except Exception as e:
            logger.error(f"Error getting embeddings: {str(e)}")
            return None

def test_azure_openai_loader():
    """Test function demonstrating the usage of AzureOpenAILoader."""
    loader = AzureOpenAILoader()
    
    # Test completion
    completion = loader.get_completion("Hello, how are you?", max_tokens=50)
    if completion:
        logger.info(f"Completion received: {completion}")
        
    # Test embeddings
    embeddings = loader.get_embeddings(["Test text for embedding"])
    if embeddings:
        logger.info(f"Embedding generated with length: {len(embeddings[0])}")

if __name__ == "__main__":
    test_azure_openai_loader()
