from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import os
import json
from pathlib import Path

# Optional YAML support
try:
    import yaml
    HAS_YAML = True
except ImportError:
    yaml = None
    HAS_YAML = False

from schema_cat.model_registry import (
    ModelRegistry, ModelMatcher, ModelResolution, ModelRequirements, 
    RoutingStrategy, RequestContext, get_global_registry, get_global_matcher
)
from schema_cat.provider_enum import Provider


@dataclass
class RouterConfig:
    """Configuration for the smart model router."""
    default_strategy: RoutingStrategy = RoutingStrategy.BEST_AVAILABLE
    aliases: Dict[str, str] = field(default_factory=dict)
    provider_priority: List[Provider] = field(default_factory=list)
    overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    provider_settings: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'RouterConfig':
        """Create RouterConfig from dictionary."""
        # Convert string strategy to enum
        default_strategy = config_dict.get('default_strategy', 'best_available')
        if isinstance(default_strategy, str):
            default_strategy = RoutingStrategy(default_strategy)

        # Convert provider priority strings to enums
        provider_priority = []
        for provider_str in config_dict.get('provider_priority', []):
            try:
                provider_priority.append(Provider(provider_str))
            except ValueError:
                continue  # Skip invalid providers

        return cls(
            default_strategy=default_strategy,
            aliases=config_dict.get('aliases', {}),
            provider_priority=provider_priority,
            overrides=config_dict.get('overrides', {}),
            provider_settings=config_dict.get('provider_settings', {})
        )

    @classmethod
    def load_from_file(cls, config_path: str) -> 'RouterConfig':
        """Load configuration from YAML or JSON file."""
        path = Path(config_path)

        if not path.exists():
            return cls()  # Return default config

        with open(path, 'r') as f:
            if path.suffix.lower() in ['.yaml', '.yml']:
                if not HAS_YAML:
                    raise ImportError(
                        "PyYAML is required to load YAML configuration files. "
                        "Install it with: pip install pyyaml"
                    )
                config_dict = yaml.safe_load(f) or {}
            elif path.suffix.lower() == '.json':
                config_dict = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {path.suffix}")

        # Extract model_routing section if it exists
        model_routing = config_dict.get('model_routing', config_dict)
        return cls.from_dict(model_routing)

    @classmethod
    def load_default(cls) -> 'RouterConfig':
        """Load default configuration from environment or config files."""
        # Try to load from environment variable
        config_path = os.getenv('SCHEMA_CAT_CONFIG')
        if config_path:
            return cls.load_from_file(config_path)

        # Try common config file locations
        possible_paths = [
            'schema_cat_config.json',  # Try JSON first (no extra dependency)
            '.schema_cat.json',
        ]

        # Add YAML paths only if YAML is available
        if HAS_YAML:
            possible_paths.extend([
                'schema_cat_config.yaml',
                'schema_cat_config.yml', 
                '.schema_cat.yaml',
                '.schema_cat.yml',
            ])

        for path in possible_paths:
            if Path(path).exists():
                try:
                    return cls.load_from_file(path)
                except ImportError:
                    # Skip YAML files if YAML is not available
                    continue

        return cls()  # Return default config


@dataclass
class RouteResult:
    """Result of model routing."""
    resolution: ModelResolution
    config_override: Optional[Dict[str, Any]] = None
    routing_reason: str = "default"


class SmartModelRouter:
    """Smart router with configuration and override support."""

    def __init__(self, registry: ModelRegistry = None, config: RouterConfig = None):
        self.registry = registry or get_global_registry()
        self.matcher = ModelMatcher(self.registry)
        self.config = config or RouterConfig.load_default()

        # Apply configuration aliases to registry
        self._apply_config_aliases()

    def _apply_config_aliases(self):
        """Apply aliases from configuration to the registry."""
        for alias, canonical in self.config.aliases.items():
            self.registry.register_alias(alias, canonical)

    async def route_model(self, model_input: str, context: RequestContext = None) -> Optional[RouteResult]:
        """
        Route model request to best provider considering:
        - API key availability
        - Provider health/latency
        - Cost optimization
        - User preferences/overrides
        - Model capabilities (context length, etc.)
        """
        context = context or RequestContext()

        # Check for model-specific overrides
        override_config = self._get_model_override(model_input)

        # Determine preferred providers
        preferred_providers = context.preferred_providers
        if not preferred_providers:
            if override_config and 'preferred_provider' in override_config:
                preferred_provider = Provider(override_config['preferred_provider'])
                preferred_providers = [preferred_provider]
            elif override_config and 'fallback_providers' in override_config:
                preferred_providers = [Provider(p) for p in override_config['fallback_providers']]
            else:
                preferred_providers = self.config.provider_priority

        # Determine routing strategy
        strategy = context.strategy or self.config.default_strategy
        if override_config and 'strategy' in override_config:
            strategy = RoutingStrategy(override_config['strategy'])

        # Apply cost threshold if specified
        requirements = context.requirements
        if override_config and 'cost_threshold' in override_config:
            if not requirements:
                requirements = ModelRequirements()
            requirements.max_cost_per_1k_tokens = override_config['cost_threshold']

        # Resolve the model
        resolution = await self.matcher.resolve_model(
            model_input=model_input,
            preferred_providers=preferred_providers,
            fallback_strategy=strategy,
            requirements=requirements
        )

        if resolution:
            routing_reason = self._determine_routing_reason(
                model_input, resolution, override_config, preferred_providers
            )

            return RouteResult(
                resolution=resolution,
                config_override=override_config,
                routing_reason=routing_reason
            )

        return None

    def _get_model_override(self, model_input: str) -> Optional[Dict[str, Any]]:
        """Get model-specific override configuration."""
        # Check direct model name
        if model_input in self.config.overrides:
            return self.config.overrides[model_input]

        # Check canonical name
        canonical_name = self.registry.get_canonical_name(model_input)
        if canonical_name and canonical_name in self.config.overrides:
            return self.config.overrides[canonical_name]

        return None

    def _determine_routing_reason(self, model_input: str, resolution: ModelResolution,
                                override_config: Optional[Dict[str, Any]], 
                                preferred_providers: List[Provider]) -> str:
        """Determine the reason for the routing decision."""
        if override_config:
            if 'preferred_provider' in override_config:
                return f"model_override_preferred_provider"
            elif 'fallback_providers' in override_config:
                return f"model_override_fallback"

        if preferred_providers and resolution.provider in preferred_providers:
            return f"preferred_provider_{resolution.provider.value}"

        return "default_priority"

    def get_available_models(self, provider: Provider = None) -> List[str]:
        """Get list of available models, optionally filtered by provider."""
        from schema_cat.provider_enum import _provider_api_key_available

        available_models = []

        for canonical_name in self.registry.get_all_canonical_names():
            models = self.registry.get_models_for_canonical(canonical_name)

            for model_info in models:
                if provider and model_info.provider != provider:
                    continue

                if _provider_api_key_available(model_info.provider):
                    available_models.append(canonical_name)
                    break  # Only add canonical name once

        return sorted(available_models)

    async def get_provider_for_model(self, model_input: str) -> Optional[Provider]:
        """Get the provider that would be used for a given model input."""
        route_result = await self.route_model(model_input)
        return route_result.resolution.provider if route_result else None

    async def validate_model_availability(self, model_input: str) -> bool:
        """Check if a model is available with current API keys."""
        route_result = await self.route_model(model_input)
        return route_result is not None


# Global router instance
_global_router = None


def get_global_router() -> SmartModelRouter:
    """Get the global smart model router instance."""
    global _global_router
    if _global_router is None:
        _global_router = SmartModelRouter()
    return _global_router


def configure_global_router(config: RouterConfig):
    """Configure the global router with custom configuration."""
    global _global_router
    _global_router = SmartModelRouter(config=config)


def reset_global_router():
    """Reset the global router to default configuration."""
    global _global_router
    _global_router = None
