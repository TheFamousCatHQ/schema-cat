import os
from enum import Enum

from schema_cat.providers import OpenRouterProvider, OpenAIProvider, AnthropicProvider
from schema_cat.providers.comet_provider import CometProvider


class Provider(str, Enum):
    OPENROUTER = "openrouter"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    COMET = "comet"

    @property
    def call(self):
        if self == Provider.OPENROUTER:
            return OpenRouterProvider().call
        elif self == Provider.OPENAI:
            return OpenAIProvider().call
        elif self == Provider.ANTHROPIC:
            return AnthropicProvider().call
        elif self == Provider.COMET:
            return CometProvider().call
        else:
            raise NotImplementedError(f"No call method for provider {self}")


def _provider_api_key_available(provider: Provider) -> bool:
    if provider == Provider.OPENROUTER:
        return bool(os.getenv("OPENROUTER_API_KEY"))
    elif provider == Provider.OPENAI:
        return bool(os.getenv("OPENAI_API_KEY"))
    elif provider == Provider.ANTHROPIC:
        return bool(os.getenv("ANTHROPIC_API_KEY"))
    elif provider == Provider.COMET:
        return bool(os.getenv("COMET_API_KEY"))
    return False
