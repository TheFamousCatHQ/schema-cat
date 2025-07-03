import logging
import os
from xml.etree import ElementTree
from typing import List, Dict, Any, Union

from schema_cat.base_provider import BaseProvider
from schema_cat.xml import xml_from_string
from schema_cat.prompt import build_system_prompt
from schema_cat.retry import with_retry

logger = logging.getLogger("schema_cat.anthropic")


class AnthropicProvider(BaseProvider):
    """Anthropic provider implementation."""

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
        import anthropic
        api_key = os.getenv("ANTHROPIC_API_KEY")
        client = anthropic.AsyncAnthropic(api_key=api_key)
        system_prompt = build_system_prompt(sys_prompt, xml_schema)
        response = await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        )
        content = response.content[0].text.strip() if hasattr(response.content[0], 'text') else response.content[0][
            'text'].strip()
        logger.info("Successfully received response from Anthropic")
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
        Retrieve available models from Anthropic.

        Note: Anthropic doesn't provide a public models API endpoint,
        so we return a hardcoded list of known models.

        Returns:
            List of model dictionaries containing model information.
        """
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            logger.warning("ANTHROPIC_API_KEY not found, cannot retrieve models")
            return []

        # Hardcoded list of known Anthropic models
        # This should be updated as new models become available
        known_models = [
            {
                'id': 'claude-3-5-sonnet-20241022',
                'object': 'model',
                'owned_by': 'anthropic',
                'context_length': 200000
            },
            {
                'id': 'claude-3-5-sonnet-latest',
                'object': 'model',
                'owned_by': 'anthropic',
                'context_length': 200000
            },
            {
                'id': 'claude-3-5-haiku-20241022',
                'object': 'model',
                'owned_by': 'anthropic',
                'context_length': 200000
            },
            {
                'id': 'claude-3-5-haiku-latest',
                'object': 'model',
                'owned_by': 'anthropic',
                'context_length': 200000
            },
            {
                'id': 'claude-3-opus-20240229',
                'object': 'model',
                'owned_by': 'anthropic',
                'context_length': 200000
            },
            {
                'id': 'claude-3-sonnet-20240229',
                'object': 'model',
                'owned_by': 'anthropic',
                'context_length': 200000
            },
            {
                'id': 'claude-3-haiku-20240307',
                'object': 'model',
                'owned_by': 'anthropic',
                'context_length': 200000
            },
            {
                'id': 'claude-sonnet-4-20250514',
                'object': 'model',
                'owned_by': 'anthropic',
                'context_length': 200000
            }
        ]

        logger.info(f"Retrieved {len(known_models)} known models from Anthropic")
        return known_models
