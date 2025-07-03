"""schema-cat: A Python library for typed prompts."""

import logging
from typing import Type, TypeVar, List, Optional

from pydantic import BaseModel

from schema_cat.model_registry import (
    ModelRequirements, RoutingStrategy, RequestContext, ModelResolution, get_global_registry, get_global_matcher,
    discover_and_register_models
)
from schema_cat.model_router import get_global_router, RouterConfig, configure_global_router
from schema_cat.provider_enum import Provider, _provider_api_key_available
from schema_cat.retry import with_retry, retry_with_exponential_backoff
from schema_cat.schema import schema_to_xml, xml_to_string, xml_to_base_model

T = TypeVar("T", bound=BaseModel)


async def _resolve_provider_and_model(
        model: str,
        provider: Provider = None,
        model_requirements: ModelRequirements = None,
        routing_strategy: RoutingStrategy = None,
        preferred_providers: List[Provider] = None,
        use_smart_routing: bool = True,
) -> tuple[Provider, str]:
    """
    Common routing logic to resolve provider and model name.

    Returns:
        Tuple of (provider, provider_model)
    """
    # If provider is specified, use smart routing but constrain to that provider
    if provider is not None and use_smart_routing:
        # Use smart routing with the specified provider as the only preferred provider
        router = get_global_router()

        # Create request context with the specified provider as the only option
        context = RequestContext(
            requirements=model_requirements,
            strategy=routing_strategy,
            preferred_providers=[provider]
        )

        # Route the model
        route_result = await router.route_model(model, context)

        if route_result is None:
            # Fallback to legacy system if smart routing fails
            logging.warning(
                f"Smart routing failed for model '{model}' with provider {provider.value}, falling back to legacy routing")
            p = provider
            provider_model = model
            logging.info(f"Fallback routing - provider: {p.value}, model: {provider_model}")
        else:
            p = route_result.resolution.provider
            provider_model = route_result.resolution.model_name
            logging.info(f"Smart routing with specified provider - provider: {p.value}, model: {provider_model}, "
                         f"canonical: {route_result.resolution.canonical_name}, "
                         f"reason: {route_result.routing_reason}, "
                         f"confidence: {route_result.resolution.confidence:.2f}")
    elif provider is not None:
        # Legacy routing when smart routing is disabled
        p = provider
        provider_model = model
        logging.info(f"Using legacy routing with provider: {p.value}, model: {provider_model}")
    elif not use_smart_routing:
        # Use new pipeline system even when smart routing is disabled
        resolution = await resolve_model(model, preferred_providers, routing_strategy, model_requirements)
        if resolution is None:
            raise ValueError(f"No available provider for model '{model}'")
        p = resolution.provider
        provider_model = resolution.model_name
        logging.info(
            f"Using pipeline routing - provider: {p.value}, model: {provider_model}, canonical: {resolution.canonical_name}")
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
        route_result = await router.route_model(model, context)

        if route_result is None:
            # Fallback to direct pipeline resolution if smart routing fails
            logging.warning(f"Smart routing failed for model '{model}', falling back to direct pipeline resolution")
            resolution = await resolve_model(model, preferred_providers, routing_strategy, model_requirements)
            if resolution is None:
                raise ValueError(f"No available provider for model '{model}'")
            p = resolution.provider
            provider_model = resolution.model_name
            logging.info(
                f"Fallback pipeline routing - provider: {p.value}, model: {provider_model}, canonical: {resolution.canonical_name}")
        else:
            p = route_result.resolution.provider
            provider_model = route_result.resolution.model_name
            logging.info(f"Smart routing - provider: {p.value}, model: {provider_model}, "
                         f"canonical: {route_result.resolution.canonical_name}, "
                         f"reason: {route_result.routing_reason}, "
                         f"confidence: {route_result.resolution.confidence:.2f}")

    return p, provider_model


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

    p, provider_model = await _resolve_provider_and_model(
        model=model,
        provider=provider,
        model_requirements=model_requirements,
        routing_strategy=routing_strategy,
        preferred_providers=preferred_providers,
        use_smart_routing=use_smart_routing
    )

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


async def prompt_without_schema(
        prompt: str,
        model: str,
        max_tokens: int = 8192,
        temperature: float = 0.0,
        sys_prompt: str = "",
        max_retries: int = 5,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        provider: Provider = None,
        # Enhanced parameters
        model_requirements: ModelRequirements = None,
        routing_strategy: RoutingStrategy = None,
        preferred_providers: List[Provider] = None,
        use_smart_routing: bool = True,
) -> str:
    """
    Send a freeform prompt to an LLM without requiring a structured response schema.

    This function provides the same intelligent model routing as prompt_with_schema
    but returns the raw string response from the LLM instead of parsing it into
    a structured format.

    Enhanced with intelligent model routing that supports:
    - Simple names: 'gpt4', 'claude', 'gemini'
    - Exact names: 'openai/gpt-4-turbo', 'anthropic/claude-3-sonnet'
    - Fuzzy matching: 'gpt4turbo' -> 'gpt-4-turbo'
    - Configuration-based overrides and routing strategies

    Args:
        prompt: The prompt to send to the LLM
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
        Raw string response from the LLM
    """

    p, provider_model = await _resolve_provider_and_model(
        model=model,
        provider=provider,
        model_requirements=model_requirements,
        routing_strategy=routing_strategy,
        preferred_providers=preferred_providers,
        use_smart_routing=use_smart_routing
    )

    # Call provider without XML schema to get raw string response
    response = await p.call(
        provider_model,
        sys_prompt,
        prompt,
        xml_schema=None,  # No schema for freeform response
        max_tokens=max_tokens,
        temperature=temperature,
        max_retries=max_retries,
        initial_delay=initial_delay,
        max_delay=max_delay,
    )
    return response


prompt = prompt_without_schema


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


async def get_provider_for_model(model: str) -> Optional[Provider]:
    """
    Get the provider that would be used for a given model input.

    Args:
        model: Model name or alias

    Returns:
        Provider that would be selected, or None if not available
    """
    router = get_global_router()
    return await router.get_provider_for_model(model)


async def validate_model_availability(model: str) -> bool:
    """
    Check if a model is available with current API keys.

    Args:
        model: Model name or alias

    Returns:
        True if model is available, False otherwise
    """
    router = get_global_router()
    return await router.validate_model_availability(model)


async def resolve_model(
        model: str,
        preferred_providers: List[Provider] = None,
        routing_strategy: RoutingStrategy = RoutingStrategy.BEST_AVAILABLE,
        model_requirements: ModelRequirements = None
) -> Optional[ModelResolution]:
    """
    Resolve a model input to the best available provider/model combination.

    This function exposes the core model resolution logic for testing and inspection.
    It supports:
    - Simple names: 'gpt4', 'claude', 'gemini'
    - Exact names: 'openai/gpt-4-turbo', 'anthropic/claude-3-sonnet'
    - Fuzzy matching: 'gpt4turbo' -> 'gpt-4-turbo'
    - Dynamic model discovery from provider APIs

    Args:
        model: The model name or alias to resolve
        preferred_providers: Optional list of preferred providers in order of preference
        routing_strategy: Strategy for model selection (cheapest, fastest, highest_quality, etc.)
        model_requirements: Optional requirements for model selection (context length, capabilities, etc.)

    Returns:
        ModelResolution object containing provider, model name, canonical name, and confidence,
        or None if no suitable model is found

    Example:
        >>> resolution = await resolve_model("gpt4")
        >>> if resolution:
        ...     print(f"Provider: {resolution.provider.value}")
        ...     print(f"Model: {resolution.model_name}")
        ...     print(f"Canonical: {resolution.canonical_name}")
        ...     print(f"Confidence: {resolution.confidence}")
    """
    matcher = get_global_matcher()
    return await matcher.resolve_model(
        model_input=model,
        preferred_providers=preferred_providers,
        fallback_strategy=routing_strategy,
        requirements=model_requirements
    )


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
    'prompt_without_schema',

    # Utility functions
    'get_available_models',
    'get_provider_for_model',
    'validate_model_availability',
    'resolve_model',
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
    'with_retry',
    'retry_with_exponential_backoff',
]
