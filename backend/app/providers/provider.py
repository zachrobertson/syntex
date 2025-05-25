from abc import ABC
from typing import Coroutine, Any, List

class ModelProvider(ABC):
    def __init__(self, text_generation_model: str, embedding_model: str):
        """
        Initialize the model provider.
        
        Args:
            text_generation_model: The name of the text generation model to use
            embedding_model: The name of the embedding model to use
        """
        self.text_model = text_generation_model
        self.embedding_model = embedding_model
    
    async def genText(self, prompt: str) -> str:
        """
        Run text generation with the model.
        
        Args:
            prompt: The input prompt for the model
            
        Returns:
            The model's response as a string
        """
        raise NotImplementedError("Subclasses must implement genText method")
    
    async def embed(self, document: str) -> List[float]:
        """
        Generate an embedding from the provided document.

        Args:
            document: The document to be embedded

        Returns:
            The model's embedding as a list of floats
        """
        raise NotImplementedError("Subclasses must implement embed method")
    
    async def checkHealth(self) -> bool:
        """
        Check if the model provider is healthy and available.
        
        Returns:
            True if the model provider is healthy, False otherwise
        """
        raise NotImplementedError("Subclasses must implement checkHealth method")
