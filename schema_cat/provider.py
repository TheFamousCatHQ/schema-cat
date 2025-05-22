import re

from schema_cat.provider_enum import Provider, _provider_api_key_available
from schema_cat.model_providers import MODEL_PROVIDER_MAP


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
