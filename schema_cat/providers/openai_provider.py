import logging
import os
from xml.etree import ElementTree
from typing import List, Dict, Any, Union

from schema_cat.base_provider import BaseProvider
from schema_cat.prompt import build_system_prompt
from schema_cat.retry import with_retry
from schema_cat.xml import xml_from_string

logger = logging.getLogger("schema_cat.openai")


class OpenAIProvider(BaseProvider):
    """OpenAI provider implementation."""

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
        import openai
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
        system_prompt = build_system_prompt(sys_prompt, xml_schema)
        messages = [
            {"role": "system",
             "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        content = response.choices[0].message.content.strip()
        logger.info("Successfully received response from OpenAI")
        logger.debug(f"Raw response content: {content}")

        # If no XML schema provided, return raw string response
        if xml_schema is None:
            logger.debug("No XML schema provided, returning raw string response")
            return content

        # Otherwise, parse as XML for structured response
        root = xml_from_string(content)
        logger.debug("Successfully parsed response as XML")
        return root

    async def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Retrieve all available models from OpenAI's API.

        Returns:
            List of model dictionaries containing model information.
        """
        import openai
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not found, cannot retrieve models")
            return []

        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)

        try:
            models_response = await client.models.list()
            models = []
            for model in models_response.data:
                model_dict = {
                    'id': model.id,
                    'object': model.object,
                    'created': getattr(model, 'created', None),
                    'owned_by': getattr(model, 'owned_by', None)
                }
                models.append(model_dict)

            logger.info(f"Retrieved {len(models)} models from OpenAI")
            return models
        except Exception as e:
            logger.error(f"Failed to retrieve models from OpenAI: {e}")
            return []
