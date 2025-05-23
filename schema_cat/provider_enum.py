import os
from enum import Enum

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