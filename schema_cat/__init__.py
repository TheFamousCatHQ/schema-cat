"""schema-cat: A Python library for typed prompts."""

import logging
from typing import Type, TypeVar

from pydantic import BaseModel

from schema_cat.anthropic import call_anthropic
from schema_cat.model_providers import MODEL_PROVIDER_MAP
from schema_cat.openai import call_openai
from schema_cat.openrouter import call_openrouter
from schema_cat.provider import get_provider_and_model
from schema_cat.provider_enum import Provider, _provider_api_key_available
from schema_cat.retry import with_retry, retry_with_exponential_backoff
from schema_cat.schema import schema_to_xml, xml_to_string, xml_to_base_model

T = TypeVar("T", bound=BaseModel)


async def prompt_with_schema(
        prompt: str,
        schema: Type[T],
        model: str,
        max_tokens: int = 8192,
        temperature: float = 0.0,
        sys_prompt: str = "",
        max_retries: int = 5,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        provider: Provider = None,
) -> T:
    """
    Automatically selects the best provider and provider-specific model for the given model name.

    Args:
        prompt: The prompt to send to the LLM
        schema: A Pydantic model class defining the expected response structure
        model: The LLM model to use (e.g., "gpt-4-turbo", "claude-3-haiku")
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (0.0 to 1.0)
        sys_prompt: Optional system prompt to prepend
        max_retries: Maximum number of retries for API calls
        initial_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        provider: Optional provider to use. If not specified, the best available provider is used.

    Returns:
        An instance of the Pydantic model
    """

    if provider is None:
        p, provider_model = get_provider_and_model(model)
    else:
        p = provider
        provider_model = model
    logging.info(f"Using provider: {p.value}, model: {provider_model}")
    xml: str = xml_to_string(schema_to_xml(schema))
    xml_elem = await p.call(
        provider_model,
        sys_prompt,
        prompt,
        xml_schema=xml,
        max_tokens=max_tokens,
        temperature=temperature,
        max_retries=max_retries,
        initial_delay=initial_delay,
        max_delay=max_delay,
    )
    return xml_to_base_model(xml_elem, schema)
