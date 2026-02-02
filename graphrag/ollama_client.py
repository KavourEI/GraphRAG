"""
Ollama Client Module.

Provides integration with Ollama for embeddings generation and LLM response generation.
"""

import logging
from typing import Optional

import requests

logger = logging.getLogger(__name__)


class OllamaClient:
    """
    A client for interacting with Ollama API.
    
    Provides methods for:
    - Generating text embeddings
    - Computing vector similarity
    - Generating LLM responses
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        embedding_model: str = "nomic-embed-text",
        llm_model: str = "llama3.2",
        timeout: int = 120,
    ):
        """
        Initialize the Ollama client.

        Args:
            base_url: Base URL of the Ollama API.
            embedding_model: Name of the model to use for embeddings.
            llm_model: Name of the LLM to use for response generation.
            timeout: Request timeout in seconds.
        """
        self.base_url = base_url.rstrip("/")
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.timeout = timeout

    def generate_embedding(self, text: str) -> list[float]:
        """
        Generate an embedding vector for the given text.

        Args:
            text: The text to embed.

        Returns:
            A list of floats representing the embedding vector.

        Raises:
            requests.RequestException: If the API request fails.
        """
        url = f"{self.base_url}/api/embed"
        payload = {
            "model": self.embedding_model,
            "input": text,
        }

        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            result = response.json()
            
            # Ollama returns embeddings in the "embeddings" field (list of lists)
            embeddings = result.get("embeddings", [[]])
            if embeddings and len(embeddings) > 0:
                return embeddings[0]
            return []
        except requests.RequestException as e:
            logger.error(f"Embedding generation failed: {e}")
            raise

    def generate_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: A list of texts to embed.

        Returns:
            A list of embedding vectors.
        """
        url = f"{self.base_url}/api/embed"
        payload = {
            "model": self.embedding_model,
            "input": texts,
        }

        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            result = response.json()
            return result.get("embeddings", [])
        except requests.RequestException as e:
            logger.error(f"Batch embedding generation failed: {e}")
            raise

    def compute_similarity(
        self,
        embedding1: list[float],
        embedding2: list[float],
    ) -> float:
        """
        Compute cosine similarity between two embedding vectors.

        Args:
            embedding1: First embedding vector.
            embedding2: Second embedding vector.

        Returns:
            Cosine similarity score between -1 and 1.
        """
        if not embedding1 or not embedding2:
            return 0.0
        
        if len(embedding1) != len(embedding2):
            logger.warning("Embedding dimensions don't match")
            return 0.0

        # Compute dot product and magnitudes
        dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
        magnitude1 = sum(a * a for a in embedding1) ** 0.5
        magnitude2 = sum(b * b for b in embedding2) ** 0.5

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)

    def find_most_similar(
        self,
        query_embedding: list[float],
        candidate_embeddings: list[list[float]],
        top_k: int = 5,
    ) -> list[tuple[int, float]]:
        """
        Find the most similar candidates to the query based on embeddings.

        Args:
            query_embedding: The embedding of the query.
            candidate_embeddings: List of candidate embeddings.
            top_k: Number of top results to return.

        Returns:
            A list of (index, similarity_score) tuples, sorted by similarity.
        """
        similarities = []
        for i, candidate in enumerate(candidate_embeddings):
            sim = self.compute_similarity(query_embedding, candidate)
            similarities.append((i, sim))
        
        # Sort by similarity score in descending order
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def generate_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate a response using the LLM.

        Args:
            prompt: The user prompt.
            system_prompt: Optional system prompt for context.
            temperature: Sampling temperature (0.0 to 1.0).
            max_tokens: Maximum number of tokens to generate.

        Returns:
            The generated response text.

        Raises:
            requests.RequestException: If the API request fails.
        """
        url = f"{self.base_url}/api/generate"
        
        payload = {
            "model": self.llm_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens

        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")
        except requests.RequestException as e:
            logger.error(f"LLM response generation failed: {e}")
            raise

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate a chat response using the LLM.

        Args:
            messages: List of message dictionaries with 'role' and 'content'.
            temperature: Sampling temperature (0.0 to 1.0).
            max_tokens: Maximum number of tokens to generate.

        Returns:
            The generated response text.

        Raises:
            requests.RequestException: If the API request fails.
        """
        url = f"{self.base_url}/api/chat"
        
        payload = {
            "model": self.llm_model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }
        
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens

        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            result = response.json()
            return result.get("message", {}).get("content", "")
        except requests.RequestException as e:
            logger.error(f"Chat response generation failed: {e}")
            raise

    def is_available(self) -> bool:
        """
        Check if the Ollama service is available.

        Returns:
            True if the service is responding, False otherwise.
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def list_models(self) -> list[str]:
        """
        List available models in Ollama.

        Returns:
            A list of model names.
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            response.raise_for_status()
            result = response.json()
            return [model["name"] for model in result.get("models", [])]
        except requests.RequestException as e:
            logger.error(f"Failed to list models: {e}")
            return []
