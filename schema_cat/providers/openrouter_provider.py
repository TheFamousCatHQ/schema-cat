import logging
import os
from xml.etree import ElementTree
from typing import List, Dict, Any, Union
import httpx

from schema_cat.providers.openai_compat_provider import OpenAiCompatProvider
from schema_cat.retry import with_retry

logger = logging.getLogger("schema_cat.openrouter")


class OpenRouterProvider(OpenAiCompatProvider):
    """OpenRouter provider implementation."""

    @with_retry()
    async def call(self,
                   model: str,
                   sys_prompt: str,
                   user_prompt: str,
                   xml_schema: str = None,
                   max_tokens: int = 8192,
                   temperature: float = 0.0,
                   max_retries: int = 5,
                   initial_delay: float = 1.0,
                   max_delay: float = 60.0) -> Union[ElementTree.XML, str]:
        api_key = os.getenv("OPENROUTER_API_KEY")
        base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

        return await self._call(base_url, api_key, model, sys_prompt, user_prompt, xml_schema, max_tokens,
                                temperature,
                                max_retries, initial_delay, max_delay)

    async def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Retrieve all available models from OpenRouter's API.

        Returns:
            List of model dictionaries containing model information.
        """
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            logger.warning("OPENROUTER_API_KEY not found, cannot retrieve models")
            return []

        base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER", "https://www.thefamouscat.com"),
            "X-Title": os.getenv("OPENROUTER_X_TITLE", "SchemaCat"),
            "Content-Type": "application/json"
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/models",
                    headers=headers,
                    timeout=30
                )
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as e:
                    logger.error(f"OpenRouter API call failed: {response.text}: {e}")
                    raise e
                data = response.json()

                models = []
                for model in data.get("data", []):
                    model_dict = {
                        'id': model.get('id'),
                        'object': model.get('object', 'model'),
                        'owned_by': model.get('owned_by'),
                        'context_length': model.get('context_length'),
                        'pricing': model.get('pricing'),
                        'top_provider': model.get('top_provider')
                    }
                    models.append(model_dict)

                logger.info(f"Retrieved {len(models)} models from OpenRouter")
                return models
        except Exception as e:
            logger.error(f"Failed to retrieve models from OpenRouter: {e}")
            return []
