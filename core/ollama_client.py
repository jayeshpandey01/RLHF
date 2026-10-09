"""Client for interacting with local or remote Ollama inference engine."""

from typing import Any
import httpx
from loguru import logger
from core.config import settings


class OllamaClient:
    """Async and synchronous client for Ollama API."""

    def __init__(self, base_url: str = settings.OLLAMA_BASE_URL, timeout: float = settings.OLLAMA_TIMEOUT_SECONDS):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def check_health(self) -> bool:
        """Check if Ollama server daemon is reachable."""
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
        except Exception as e:
            logger.debug(f"Ollama health check failed: {e}")
            return False

    async def list_models(self) -> list[str]:
        """List all currently downloaded model names in Ollama."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                if response.status_code == 200:
                    models_data = response.json().get("models", [])
                    return [m.get("name") for m in models_data if "name" in m]
        except Exception as e:
            logger.error(f"Failed to fetch model list from Ollama: {e}")
        return []

    async def pull_model(self, model_name: str) -> bool:
        """Download a model into Ollama."""
        logger.info(f"Initiating Ollama pull for model: {model_name}...")
        try:
            async with httpx.AsyncClient(timeout=600.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/pull",
                    json={"name": model_name, "stream": False},
                )
                if response.status_code == 200:
                    logger.info(f"Successfully pulled {model_name}.")
                    return True
                else:
                    logger.error(f"Ollama pull failed ({response.status_code}): {response.text}")
        except Exception as e:
            logger.error(f"Error pulling model {model_name}: {e}")
        return False

    async def chat(
        self,
        messages: list[dict[str, str]],
        model: str = settings.DEFAULT_MODEL,
        temperature: float = 0.7,
    ) -> str:
        """Send chat messages and return assistant response."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "model": model,
                        "messages": messages,
                        "stream": False,
                        "options": {"temperature": temperature},
                    },
                )
                if response.status_code == 200:
                    data = response.json()
                    return data.get("message", {}).get("content", "")
                else:
                    logger.error(f"Chat request failed ({response.status_code}): {response.text}")
                    return ""
        except Exception as e:
            logger.error(f"Exception during Ollama chat: {e}")
            return ""


# Global singleton client
ollama_client = OllamaClient()
