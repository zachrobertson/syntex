import os
import openai
import asyncio
from typing import List, Optional

from providers.provider import ModelProvider

class OpenAIModel(ModelProvider):
    models = [
        "gpt-4o",
        "gpt-4.1",
        "o3",
        "o3-mini",
        "o4-mini"
    ]
    embedding_models = [
        "text-embedding-3-small",
        "text-embedding-3-large",
        "text-embedding-ada-002"
    ]

    def __init__(self, text_model: str, embedding_model: Optional[str] = "text-embedding-3-small", api_key: Optional[str] = None):
        if text_model not in self.models:
            raise ValueError(f"Model is not a valid OpenAI model: {text_model}, valid models are: {self.models}")
        if embedding_model not in self.embedding_models:
            raise ValueError(f"Model is not a valid OpenAI embedding model: {embedding_model}, valid models are: {self.embedding_models}")
        super().__init__(text_model, embedding_model)
        
        self.text_model = text_model
        self.embedding_model = embedding_model
        self.client = openai.AsyncOpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY")
        )
        if not self.client.api_key:
            raise ValueError("OpenAI API key is required. Set it via OPENAI_API_KEY environment variable or pass it directly.")

    async def genText(self, prompt: str) -> str:
        """
        Run text generation with the OpenAI model.
        
        Args:
            prompt: The input prompt for the model
            
        Returns:
            str: The model's generated text response
            
        Raises:
            openai.OpenAIError: If there's an error with the OpenAI API
        """
        try:
            response = await self.client.chat.completions.create(
                model=self.text_model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except openai.OpenAIError as e:
            raise Exception(f"OpenAI API error: {str(e)}")
        
    async def embed(self, document: str) -> List[float]:
        """
        Generate embeddings for the given document using OpenAI's embedding model.
        
        Args:
            document: The text to embed
            
        Returns:
            List[float]: The embedding vector
            
        Raises:
            Exception: If there's an error with the OpenAI API
        """
        try:
            response = await self.client.embeddings.create(
                model=self.embedding_model,
                input=document
            )
            return response.data[0].embedding
        except Exception as e:
            raise Exception(f"OpenAI embedding error: {str(e)}")

    async def checkHealth(self) -> bool:
        """
        Check if the OpenAI API is healthy and available.
        
        Returns:
            bool: True if the API is healthy, False otherwise
        """
        try:
            await asyncio.gather(
                self.genText("Test of the text generation endpoint"),
                self.embed("Test of the embedding endpoint")
            )
            return True
        except Exception as e:
            print(f"Health check failed: {str(e)}")
            return False
