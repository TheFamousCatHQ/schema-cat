"""schema-cat: A Python library for typed prompts."""

import logging
from typing import Type, TypeVar

from pydantic import BaseModel

from schema_cat.anthropic import call_anthropic
from schema_cat.openai import call_openai
from schema_cat.openrouter import call_openrouter
from schema_cat.provider import Provider, _provider_api_key_available, MODEL_PROVIDER_MAP, get_provider_and_model
from schema_cat.schema import schema_to_xml, xml_to_string, xml_to_base_model

T = TypeVar("T", bound=BaseModel)


async def prompt_with_schema(
        prompt: str,
        schema: Type[T],
        model: str,
        max_tokens: int = 8192,
        temperature: float = 0.0,
        sys_prompt: str = "",
) -> T:
    """
    Automatically selects the best provider and provider-specific model for the given model name.
    """
    provider, provider_model = get_provider_and_model(model)
    logging.info(f"Using provider: {provider.value}, model: {provider_model}")
    xml: str = xml_to_string(schema_to_xml(schema))
    xml_elem = await provider.call(
        provider_model,
        sys_prompt,
        prompt,
        xml_schema=xml,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return xml_to_base_model(xml_elem, schema)
