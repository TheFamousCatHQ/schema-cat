import os
from enum import Enum
import re

from schema_cat.anthropic import call_anthropic
from schema_cat.openai import call_openai
from schema_cat.openrouter import call_openrouter


class Provider(str, Enum):
    OPENROUTER = "openrouter"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"

    @property
    def call(self):
        if self == Provider.OPENROUTER:
            return call_openrouter
        elif self == Provider.OPENAI:
            return call_openai
        elif self == Provider.ANTHROPIC:
            return call_anthropic
        else:
            raise NotImplementedError(f"No call method for provider {self}")


def _provider_api_key_available(provider: Provider) -> bool:
    if provider == Provider.OPENROUTER:
        return bool(os.getenv("OPENROUTER_API_KEY"))
    elif provider == Provider.OPENAI:
        return bool(os.getenv("OPENAI_API_KEY"))
    elif provider == Provider.ANTHROPIC:
        return bool(os.getenv("ANTHROPIC_API_KEY"))
    return False


# Canonical model mapping: maps internal model names to provider/model pairs in order of preference
MODEL_PROVIDER_MAP = {
    # Google Gemini
    "gemini-2.5-flash-preview": [
        (
            Provider.OPENROUTER,
            "google/gemini-2.5-flash-preview",
        ),  # Default, categorize_files_openrouter, etc.
        (Provider.OPENAI, "gpt-4.1-mini"),
        (Provider.ANTHROPIC, "claude-3.5-sonnet-latest"),
    ],
    # OpenAI nano
    "gpt-4.1-nano-2025-04-14": [
        (
            Provider.OPENROUTER,
            "openai/gpt-4.1-nano-2025-04-14",
        ),  # categorize_files_openai_json
        (Provider.OPENAI, "gpt-4.1-nano-2025-04-14"),
        (Provider.ANTHROPIC, "claude-3.5-sonnet-latest"),
    ],
    # OpenAI mini
    "gpt-4.1-mini": [
        (Provider.OPENROUTER, "openai/gpt-4.1-mini"),  # bug_analyzer (schema_cat)
        (Provider.OPENAI, "gpt-4.1-mini"),
        (Provider.ANTHROPIC, "claude-3.7-sonnet-latest"),
    ],
    "openai/gpt-4.1-mini": [
        (Provider.OPENROUTER, "openai/gpt-4.1-mini"),
        (Provider.OPENAI, "gpt-4.1-mini"),
        (Provider.ANTHROPIC, "claude-3.7-sonnet-latest"),
    ],
    # OpenAI gpt-4o-mini
    "gpt-4o-mini": [
        (Provider.OPENROUTER, "openrouter/gpt-4o-mini"),  # validate_complexity_report
        (Provider.OPENAI, "gpt-4o-mini"),
        (Provider.ANTHROPIC, "claude-3.7-sonnet-latest"),
    ],
    "openrouter/gpt-4o-mini": [
        (Provider.OPENROUTER, "openrouter/gpt-4o-mini"),
        (Provider.OPENAI, "gpt-4o-mini"),
        (Provider.ANTHROPIC, "claude-3.7-sonnet-latest"),
    ],
    # Anthropic Claude Sonnet
    "claude-3.5-sonnet": [
        (
            Provider.ANTHROPIC,
            "claude-3.5-sonnet-latest",
        ),  # Docstring reference in create_agent
        (Provider.OPENROUTER, "anthropic/claude-3.5-sonnet"),
        (Provider.OPENAI, "anthropic/gpt-4.1-mini"),
    ],
    "anthropic/claude-3.5-sonnet": [
        (Provider.ANTHROPIC, "claude-3.5-sonnet-latest"),
        (Provider.OPENROUTER, "anthropic/claude-3.5-sonnet"),
        (Provider.OPENAI, "anthropic/gpt-4.1-mini"),
    ],
    # Existing entries
    "claude-haiku": [
        (Provider.ANTHROPIC, "claude-3-haiku-20240307"),
        (Provider.OPENROUTER, "openrouter/claude-3-haiku-20240307"),
        (Provider.OPENAI, "gpt-4.1-nano"),  # fallback to a similar OpenAI model
    ],
    "anthropic/claude-3.5-haiku": [
        (Provider.ANTHROPIC, "claude-3-5-haiku-latest"),
        (Provider.OPENROUTER, "anthropic/claude-3.5-haiku"),
        (Provider.OPENAI, "anthropic/gpt-4.1-nano"),
    ],
    "gemma": [
        (Provider.OPENROUTER, "google/gemma-3-4b-it"),
        (Provider.OPENROUTER, "anthropic/claude-3.5-haiku"),
        (Provider.ANTHROPIC, "claude-3-5-haiku-latest"),
    ]
}


def _normalize_model_name(name: str) -> str:
    # Remove all non-alphanumeric characters and lowercase
    return re.sub(r"[^a-zA-Z0-9]", "", name).lower()


def get_provider_and_model(model_name: str) -> tuple[Provider, str]:
    """
    Given a model name (provider-specific or canonical), return the best available (provider, provider_model_name) tuple.
    - If provider-specific (contains '/'), try that provider first, then fall back to canonical mapping.
    - If canonical, use priority: OPENROUTER, OPENAI, ANTHROPIC.
    - If not found as a key, search all values for a matching model name (deep search, normalized).
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
        norm_model_name = _normalize_model_name(model_name)
        # Canonical: use priority order in MODEL_PROVIDER_MAP
        for key, candidates in MODEL_PROVIDER_MAP.items():
            if _normalize_model_name(key) == norm_model_name:
                for provider, provider_model in candidates:
                    if _provider_api_key_available(provider):
                        return provider, provider_model
        # Deep search: look for normalized model_name in all values
        for candidates in MODEL_PROVIDER_MAP.values():
            for provider, provider_model in candidates:
                if _normalize_model_name(provider_model) == norm_model_name and _provider_api_key_available(provider):
                    return provider, provider_model
        raise ValueError(
            f"No available provider/model for canonical model '{model_name}'"
        )
