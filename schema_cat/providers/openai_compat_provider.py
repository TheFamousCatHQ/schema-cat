import logging
import os
from abc import ABC
from typing import Union
from xml.etree import ElementTree

import httpx

from schema_cat.base_provider import BaseProvider
from schema_cat.prompt import build_system_prompt
from schema_cat.retry import with_retry
from schema_cat.xml import xml_from_string, XMLParsingError

logger = logging.getLogger("schema_cat.openai_compat")


class OpenAiCompatProvider(BaseProvider, ABC):
    """OpenAI compatible provider implementation."""

    async def _call_internal(self,
                             base_url: str,
                             api_key: str,
                             model: str,
                             sys_prompt: str,
                             user_prompt: str,
                             xml_schema: str = None,
                             max_tokens: int = 8192,
                             temperature: float = 0.0) -> Union[ElementTree.XML, str]:
        """Internal call method without retry decorator to avoid conflicts."""
        system_prompt = build_system_prompt(sys_prompt, xml_schema)
        data = {
            "model": model,
            "messages": [
                {"role": "system",
                 "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER", "https://www.thefamouscat.com"),
            "X-Title": os.getenv("OPENROUTER_X_TITLE", "SchemaCat"),
            "Content-Type": "application/json"
        }

        logger.info(f"Calling OpenAI compatible API with model {model}")
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=60
            )
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as e:
                logger.error(f"OpenAI compatible API call failed: {response.text}: {e}")
                raise e
            content = response.json()["choices"][0]["message"]["content"].strip()

        logger.info(f"Successfully received response from {self.__class__.__name__}")
        logger.debug(f"Raw response content: {content}")

        # If no XML schema provided, return raw string response
        if xml_schema is None:
            logger.debug("No XML schema provided, returning raw string response")
            return content

        # Parse the response content as XML
        try:
            root = xml_from_string(content)
            logger.debug("Successfully parsed response as XML")
            return root
        except XMLParsingError as e:
            logger.warning(f"XML parsing failed: {str(e)}")
            # Attach the original content to the exception for potential XML fixing
            e.original_content = content
            raise e

    @with_retry()
    async def _call(self,
                    base_url: str,
                    api_key: str,
                    model: str,
                    sys_prompt: str,
                    user_prompt: str,
                    xml_schema: str = None,
                    max_tokens: int = 8192,
                    temperature: float = 0.0,
                    max_retries: int = 5,
                    initial_delay: float = 1.0,
                    max_delay: float = 60.0,
                    retry_model: str = None) -> Union[ElementTree.XML, str]:

        if retry_model is None:
            retry_model = model

        try:
            # Try the main call first
            return await self._call_internal(
                base_url=base_url,
                api_key=api_key,
                model=model,
                sys_prompt=sys_prompt,
                user_prompt=user_prompt,
                xml_schema=xml_schema,
                max_tokens=max_tokens,
                temperature=temperature
            )
        except XMLParsingError as e:
            # If XML parsing fails and we have retries left, try to fix the XML
            if max_retries > 0 and xml_schema is not None:
                logger.info("XML parsing failed, attempting to fix with retry model")
                try:
                    # Get the original content that failed to parse
                    original_content = getattr(e, 'original_content', 'Invalid XML content')
                    # Use the internal method to avoid triggering the retry decorator again
                    return await self._call_internal(
                        base_url=base_url,
                        api_key=api_key,
                        model=retry_model,
                        sys_prompt="Convert this data into valid XML according to the schema",
                        user_prompt=original_content,
                        xml_schema=xml_schema,
                        max_tokens=max_tokens,
                        temperature=temperature
                    )
                except Exception as retry_error:
                    logger.warning(f"XML fix attempt failed: {str(retry_error)}")
                    # If the fix attempt fails, raise the original error
                    raise e
            else:
                # No retries left or no schema, raise the original error
                raise e
