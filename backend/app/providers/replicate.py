import os
import json
import replicate
import asyncio

from typing import Dict, Any, List
from providers.provider import ModelProvider

from models import ReplicateModelConfig

class ReplicateModel(ModelProvider):
    def __init__(self, config_path: str):
        """
        Initialize the Replicate model provider.
        
        Args:
            config_path: Path to the JSON configuration file
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            config_data = json.load(f)

        self.config = ReplicateModelConfig(**config_data)

        api_key=os.getenv("REPLICATE_API_KEY")
        if not api_key:
            raise EnvironmentError("REPLICATE_API_KEY not found in environment variables")

        self.client = replicate.Client(os.getenv("REPLICATE_API_KEY"))
        super().__init__(self.config.text.model, self.config.embedding.model)

    async def _request(self, model: str, input: Dict[str, Any], options: Dict[str, Any] = None) -> Any:
        return await self.client.async_run(
            model,
            input,
            **(options or {})
        )

    async def genText(self, prompt: str) -> str:
        """
        Run text generation with the Replicate model.
        
        Args:
            prompt: The input prompt for the model
            
        Returns:
            str: The model's generated text response
        """
        try:
            response = await self._request(
                self.text_model,
                {"prompt": prompt}
            )
            return str(response)
        except Exception as e:
            raise Exception(f"Replicate API error: {str(e)}")

    async def embed(self, document: str) -> List[float]:
        """
        Generate embeddings for the given document using Replicate's embedding model.
        
        Args:
            document: The text to embed
            
        Returns:
            List[float]: The embedding vector
        """
        try:
            response = await self._request(
                self.embedding_model,
                {"text": document}
            )
            return response
        except Exception as e:
            raise Exception(f"Replicate embedding error: {str(e)}")

    async def checkHealth(self) -> bool:
        """
        Check if the Replicate API is healthy and available.
        
        Returns:
            bool: True if the API is healthy, False otherwise
        """
        try:
            await asyncio.gather(
                self._request(self.text_model, self.config.text.healthCheck),
                self._request(self.embedding_model, self.config.embedding.healthCheck)
            )
            return True
        except Exception as e:
            print(f"Health check failed: {str(e)}")
            return False
