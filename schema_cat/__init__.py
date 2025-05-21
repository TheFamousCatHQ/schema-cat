"""schema-cat: A Python library for typed prompts."""
from enum import Enum
from typing import Type, TypeVar

from pydantic import BaseModel

from schema_cat.anthropic import call_anthropic
from schema_cat.openai import call_openai
from schema_cat.openrouter import call_openrouter
from schema_cat.schema import schema_to_xml, xml_to_string, xml_to_base_model

T = TypeVar('T', bound=BaseModel)


class Provider(str, Enum):
    OPENROUTER = "openrouter"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


async def prompt_with_schema(
    prompt: str,
    schema: Type[T],
    model: str,
    provider: Provider,
    max_tokens: int = 8192,
    temperature: float = 0.0,
    sys_prompt: str = ""
) -> T:
    xml: str = xml_to_string(schema_to_xml(schema))
    if provider == Provider.OPENROUTER:
        xml_elem = await call_openrouter(
            model, sys_prompt, prompt, xml_schema=xml,
            max_tokens=max_tokens, temperature=temperature
        )
        return xml_to_base_model(xml_elem, schema)
    elif provider == Provider.OPENAI:
        xml_elem = await call_openai(
            model, sys_prompt, prompt, xml_schema=xml,
            max_tokens=max_tokens, temperature=temperature
        )
        return xml_to_base_model(xml_elem, schema)
    elif provider == Provider.ANTHROPIC:
        xml_elem = await call_anthropic(
            model, sys_prompt, prompt, xml_schema=xml,
            max_tokens=max_tokens, temperature=temperature
        )
        return xml_to_base_model(xml_elem, schema)
    else:
        raise Exception(f"Provider {provider} not supported")
