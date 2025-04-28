from abc import ABC

class ModelProvider(ABC):
    def __init__(self, model_name: str):
        """
        Initialize the model provider.
        
        Args:
            model_name: The name of the model to use
        """
        self.model_name = model_name
    
    def genText(self, prompt: str) -> str:
        """
        Run text generation with the model.
        
        Args:
            prompt: The input prompt for the model
            
        Returns:
            The model's response as a string
        """
        raise NotImplementedError("Subclasses must implement genText method")
    
    def embed(self, document: str) -> list[float]:
        """
        Generate an embedding from the provided document.

        Args:
            document: The document to be embedded

        Returns:
            The model's embedding as a list of floats
        """
        raise NotImplementedError("Subclasses must implement embed method")
    
    def checkHealth(self) -> bool:
        """
        Check if the model provider is healthy and available.
        
        Returns:
            True if the model provider is healthy, False otherwise
        """
        raise NotImplementedError("Subclasses must implement checkHealth method")
