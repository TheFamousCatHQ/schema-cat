import logging
import os
from xml.etree import ElementTree

from schema_cat.providers.openai_compat_provider import OpenAiCompatProvider
from schema_cat.retry import with_retry

logger = logging.getLogger("schema_cat.openrouter")


class CometProvider(OpenAiCompatProvider):
    """CometApi provider implementation."""

    @with_retry()
    async def call(self,
                   model: str,
                   sys_prompt: str,
                   user_prompt: str,
                   xml_schema: str,
                   max_tokens: int = 8192,
                   temperature: float = 0.0,
                   max_retries: int = 5,
                   initial_delay: float = 1.0,
                   max_delay: float = 60.0) -> ElementTree.XML:
        api_key = os.getenv("COMET_API_KEY")
        base_url = "https://api.cometapi.com/v1"

        return await self._call(base_url, api_key, model, sys_prompt, user_prompt, xml_schema, max_tokens,
                                temperature,
                                max_retries, initial_delay, max_delay)
