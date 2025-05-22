"""schema-cat: A Python library for typed prompts."""

import os
from enum import Enum
from typing import Type, TypeVar
import logging

from pydantic import BaseModel

from schema_cat.anthropic import call_anthropic
from schema_cat.openai import call_openai
from schema_cat.openrouter import call_openrouter
from schema_cat.schema import schema_to_xml, xml_to_string, xml_to_base_model

T = TypeVar("T", bound=BaseModel)


class Provider(str, Enum):
    OPENROUTER = "openrouter"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


# Canonical model mapping: maps internal model names to provider/model pairs in order of preference
MODEL_PROVIDER_MAP = {
    # Google Gemini
    "gemini-2.5-flash-preview": [
        (
            Provider.OPENROUTER,
            "google/gemini-2.5-flash-preview",
        ),  # Default, categorize_files_openrouter, etc.
        (Provider.OPENAI, "gpt-4.1-mini"),
        (Provider.ANTHROPIC, "claude-3.5-sonnet"),
    ],
    # OpenAI nano
    "gpt-4.1-nano-2025-04-14": [
        (
            Provider.OPENROUTER,
            "openai/gpt-4.1-nano-2025-04-14",
        ),  # categorize_files_openai_json
        (Provider.OPENAI, "gpt-4.1-nano-2025-04-14"),
        (Provider.ANTHROPIC, "claude-3.5-sonnet"),
    ],
    # OpenAI mini
    "gpt-4.1-mini": [
        (Provider.OPENROUTER, "openai/gpt-4.1-mini"),  # bug_analyzer (schema_cat)
        (Provider.OPENAI, "gpt-4.1-mini"),
        (Provider.ANTHROPIC, "claude-3.7-sonnet"),
    ],
    "openai/gpt-4.1-mini": [
        (Provider.OPENROUTER, "openai/gpt-4.1-mini"),
        (Provider.OPENAI, "gpt-4.1-mini"),
        (Provider.ANTHROPIC, "claude-3.7-sonnet"),
    ],
    # OpenAI gpt-4o-mini
    "gpt-4o-mini": [
        (Provider.OPENROUTER, "openrouter/gpt-4o-mini"),  # validate_complexity_report
        (Provider.OPENAI, "gpt-4o-mini"),
        (Provider.ANTHROPIC, "claude-3.7-sonnet"),
    ],
    "openrouter/gpt-4o-mini": [
        (Provider.OPENROUTER, "openrouter/gpt-4o-mini"),
        (Provider.OPENAI, "gpt-4o-mini"),
        (Provider.ANTHROPIC, "claude-3.7-sonnet"),
    ],
    # Anthropic Claude Sonnet
    "claude-3.5-sonnet": [
        (
            Provider.ANTHROPIC,
            "claude-3.5-sonnet",
        ),  # Docstring reference in create_agent
        (Provider.OPENROUTER, "anthropic/claude-3.5-sonnet"),
        (Provider.OPENAI, "anthropic/gpt-4.1-mini"),
    ],
    "anthropic/claude-3.5-sonnet": [
        (Provider.ANTHROPIC, "claude-3.5-sonnet"),
        (Provider.OPENROUTER, "anthropic/claude-3.5-sonnet"),
        (Provider.OPENAI, "anthropic/gpt-4.1-mini"),
    ],
    # Existing entries
    "claude-haiku": [
        (Provider.ANTHROPIC, "claude-3-haiku-20240307"),
        (Provider.OPENROUTER, "openrouter/claude-3-haiku-20240307"),
        (Provider.OPENAI, "gpt-4.1-nano"),  # fallback to a similar OpenAI model
    ],
}


def _provider_api_key_available(provider: Provider) -> bool:
    if provider == Provider.OPENROUTER:
        return bool(os.getenv("OPENROUTER_API_KEY"))
    elif provider == Provider.OPENAI:
        return bool(os.getenv("OPENAI_API_KEY"))
    elif provider == Provider.ANTHROPIC:
        return bool(os.getenv("ANTHROPIC_API_KEY"))
    return False


def get_provider_and_model(model_name: str) -> tuple[Provider, str]:
    """
    Given a model name (provider-specific or canonical), return the best available (provider, provider_model_name) tuple.
    - If provider-specific (contains '/'), try that provider first, then fall back to canonical mapping.
    - If canonical, use priority: OPENROUTER, OPENAI, ANTHROPIC.
    """
    if "/" in model_name:
        # Provider-specific: extract provider
        provider_str, provider_model = model_name.split("/", 1)
        try:
            provider = Provider(provider_str.lower())
        except ValueError:
            provider = None
        if provider and _provider_api_key_available(provider):
            return provider, model_name
        # Fallback: try canonical mapping if available
        for canonical, candidates in MODEL_PROVIDER_MAP.items():
            for cand_provider, cand_model in candidates:
                if cand_model == model_name and _provider_api_key_available(
                    cand_provider
                ):
                    return cand_provider, cand_model
        # Try canonical fallback by canonical name
        for canonical, candidates in MODEL_PROVIDER_MAP.items():
            for cand_provider, cand_model in candidates:
                if _provider_api_key_available(cand_provider):
                    return cand_provider, cand_model
        raise ValueError(
            f"No available provider for provider-specific model '{model_name}'"
        )
    else:
        # Canonical: use priority order in MODEL_PROVIDER_MAP
        candidates = MODEL_PROVIDER_MAP.get(model_name, [])
        for provider, provider_model in candidates:
            if _provider_api_key_available(provider):
                return provider, provider_model
        raise ValueError(
            f"No available provider/model for canonical model '{model_name}'"
        )


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
    if provider == Provider.OPENROUTER:
        xml_elem = await call_openrouter(
            provider_model,
            sys_prompt,
            prompt,
            xml_schema=xml,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return xml_to_base_model(xml_elem, schema)
    elif provider == Provider.OPENAI:
        xml_elem = await call_openai(
            provider_model,
            sys_prompt,
            prompt,
            xml_schema=xml,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return xml_to_base_model(xml_elem, schema)
    elif provider == Provider.ANTHROPIC:
        xml_elem = await call_anthropic(
            provider_model,
            sys_prompt,
            prompt,
            xml_schema=xml,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return xml_to_base_model(xml_elem, schema)
    else:
        raise Exception(f"Provider {provider} not supported")
