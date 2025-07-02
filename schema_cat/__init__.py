"""schema-cat: A Python library for typed prompts."""

import logging
from typing import Type, TypeVar, List, Optional

from pydantic import BaseModel

from schema_cat.model_providers import MODEL_PROVIDER_MAP
from schema_cat.provider import get_provider_and_model
from schema_cat.provider_enum import Provider, _provider_api_key_available
from schema_cat.retry import with_retry, retry_with_exponential_backoff
from schema_cat.schema import schema_to_xml, xml_to_string, xml_to_base_model
from schema_cat.model_registry import (
    ModelRequirements, RoutingStrategy, RequestContext, get_global_registry, get_global_matcher,
    discover_and_register_models
)
from schema_cat.model_router import get_global_router, RouterConfig, configure_global_router

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
        # New enhanced parameters
        model_requirements: ModelRequirements = None,
        routing_strategy: RoutingStrategy = None,
        preferred_providers: List[Provider] = None,
        use_smart_routing: bool = True,
) -> T:
    """
    Automatically selects the best provider and provider-specific model for the given model name.

    Enhanced with intelligent model routing that supports:
    - Simple names: 'gpt4', 'claude', 'gemini'
    - Exact names: 'openai/gpt-4-turbo', 'anthropic/claude-3-sonnet'
    - Fuzzy matching: 'gpt4turbo' -> 'gpt-4-turbo'
    - Configuration-based overrides and routing strategies

    Args:
        prompt: The prompt to send to the LLM
        schema: A Pydantic model class defining the expected response structure
        model: The LLM model to use (e.g., "gpt4", "claude", "gpt-4-turbo", "openai/gpt-4-turbo")
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (0.0 to 1.0)
        sys_prompt: Optional system prompt to prepend
        max_retries: Maximum number of retries for API calls
        initial_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        provider: Optional provider to use. If specified, bypasses smart routing.
        model_requirements: Optional requirements for model selection (context length, capabilities, etc.)
        routing_strategy: Optional strategy for model selection (cheapest, fastest, highest_quality, etc.)
        preferred_providers: Optional list of preferred providers in order of preference
        use_smart_routing: Whether to use the new smart routing system (default: True)

    Returns:
        An instance of the Pydantic model
    """

    # Backward compatibility: if provider is specified, use legacy routing
    if provider is not None:
        p = provider
        provider_model = model
        logging.info(f"Using legacy routing with provider: {p.value}, model: {provider_model}")
    elif not use_smart_routing:
        # Use legacy routing system
        p, provider_model = get_provider_and_model(model)
        logging.info(f"Using legacy routing - provider: {p.value}, model: {provider_model}")
    else:
        # Use new smart routing system
        router = get_global_router()

        # Create request context
        context = RequestContext(
            requirements=model_requirements,
            strategy=routing_strategy,
            preferred_providers=preferred_providers
        )

        # Route the model
        route_result = router.route_model(model, context)

        if route_result is None:
            # Fallback to legacy system if smart routing fails
            logging.warning(f"Smart routing failed for model '{model}', falling back to legacy routing")
            try:
                p, provider_model = get_provider_and_model(model)
                logging.info(f"Fallback routing - provider: {p.value}, model: {provider_model}")
            except ValueError as e:
                raise ValueError(f"No available provider for model '{model}'. {str(e)}")
        else:
            p = route_result.resolution.provider
            provider_model = route_result.resolution.model_name
            logging.info(f"Smart routing - provider: {p.value}, model: {provider_model}, "
                        f"canonical: {route_result.resolution.canonical_name}, "
                        f"reason: {route_result.routing_reason}, "
                        f"confidence: {route_result.resolution.confidence:.2f}")

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


# Utility functions for the enhanced API
def get_available_models(provider: Provider = None) -> List[str]:
    """
    Get list of available models, optionally filtered by provider.

    Args:
        provider: Optional provider to filter by

    Returns:
        List of available model names
    """
    router = get_global_router()
    return router.get_available_models(provider)


def get_provider_for_model(model: str) -> Optional[Provider]:
    """
    Get the provider that would be used for a given model input.

    Args:
        model: Model name or alias

    Returns:
        Provider that would be selected, or None if not available
    """
    router = get_global_router()
    return router.get_provider_for_model(model)


def validate_model_availability(model: str) -> bool:
    """
    Check if a model is available with current API keys.

    Args:
        model: Model name or alias

    Returns:
        True if model is available, False otherwise
    """
    router = get_global_router()
    return router.validate_model_availability(model)


def configure_routing(config: RouterConfig):
    """
    Configure the global router with custom configuration.

    Args:
        config: RouterConfig instance with custom settings
    """
    configure_global_router(config)


def load_config_from_file(config_path: str):
    """
    Load routing configuration from a YAML or JSON file.

    Args:
        config_path: Path to configuration file
    """
    config = RouterConfig.load_from_file(config_path)
    configure_global_router(config)


# Export public API
__all__ = [
    # Core functions
    'prompt_with_schema',

    # Utility functions
    'get_available_models',
    'get_provider_for_model', 
    'validate_model_availability',
    'configure_routing',
    'load_config_from_file',
    'discover_and_register_models',

    # Classes and enums
    'Provider',
    'ModelRequirements',
    'RoutingStrategy',
    'RouterConfig',

    # Schema functions
    'schema_to_xml',
    'xml_to_string', 
    'xml_to_base_model',

    # Legacy exports for backward compatibility
    'MODEL_PROVIDER_MAP',
    'get_provider_and_model',
    'with_retry',
    'retry_with_exponential_backoff',
]
