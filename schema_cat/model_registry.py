from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import re
import os
from schema_cat.provider_enum import Provider


@dataclass
class ModelRequirements:
    """Requirements for model selection."""
    min_context_length: Optional[int] = None
    supports_function_calling: Optional[bool] = None
    max_cost_per_1k_tokens: Optional[float] = None
    min_quality_score: Optional[float] = None


@dataclass
class ModelCapabilities:
    """Capabilities of a specific model."""
    context_length: int = 4096
    supports_function_calling: bool = False
    cost_per_1k_tokens: float = 0.0
    quality_score: float = 0.0
    supports_streaming: bool = True


@dataclass
class ModelInfo:
    """Information about a registered model."""
    canonical_name: str
    provider: Provider
    provider_model_name: str
    priority: int = 0
    aliases: List[str] = field(default_factory=list)
    capabilities: ModelCapabilities = field(default_factory=ModelCapabilities)


@dataclass
class ModelResolution:
    """Result of model resolution."""
    provider: Provider
    model_name: str
    canonical_name: str
    confidence: float = 1.0


class RoutingStrategy(str, Enum):
    """Available routing strategies."""
    BEST_AVAILABLE = "best_available"
    CHEAPEST = "cheapest"
    FASTEST = "fastest"
    MOST_RELIABLE = "most_reliable"
    HIGHEST_QUALITY = "highest_quality"
    MAX_CONTEXT = "max_context"


@dataclass
class RequestContext:
    """Context for model routing requests."""
    requirements: Optional[ModelRequirements] = None
    strategy: Optional[RoutingStrategy] = None
    preferred_providers: Optional[List[Provider]] = None


class ModelRegistry:
    """Dynamic registry for models and their capabilities."""

    def __init__(self):
        self._models: Dict[str, List[ModelInfo]] = {}
        self._aliases: Dict[str, str] = {}
        self._provider_capabilities: Dict[Provider, Dict[str, ModelCapabilities]] = {}
        self._initialize_default_models()

    def _initialize_default_models(self):
        """Initialize with common aliases only - models will be discovered dynamically."""
        # Only register common aliases - no hardcoded models
        self._register_common_aliases()

    def _register_common_aliases(self):
        """Register common model aliases."""
        common_aliases = {
            # GPT-4 variants
            "gpt4": "gpt-4.1-mini",
            "gpt4o": "gpt-4o-mini", 
            "gpt4-mini": "gpt-4.1-mini",
            "gpt4-turbo": "gpt-4.1-mini",

            # Claude variants
            "claude": "claude-3.5-sonnet",
            "claude-sonnet": "claude-3.5-sonnet",
            "claude-haiku": "claude-haiku",

            # Gemini variants
            "gemini": "google/gemini-2.5-flash",
            "gemini-flash": "google/gemini-2.5-flash",
        }

        for alias, canonical in common_aliases.items():
            self.register_alias(alias, canonical)

    def register_model(self, canonical_name: str, provider: Provider, 
                      provider_model_name: str, priority: int = 0,
                      aliases: List[str] = None, 
                      capabilities: ModelCapabilities = None) -> None:
        """Register a model with the registry."""
        if canonical_name not in self._models:
            self._models[canonical_name] = []

        model_info = ModelInfo(
            canonical_name=canonical_name,
            provider=provider,
            provider_model_name=provider_model_name,
            priority=priority,
            aliases=aliases or [],
            capabilities=capabilities or ModelCapabilities()
        )

        self._models[canonical_name].append(model_info)

        # Sort by priority (lower number = higher priority)
        self._models[canonical_name].sort(key=lambda x: x.priority)

        # Register aliases
        if aliases:
            for alias in aliases:
                self.register_alias(alias, canonical_name)

    def register_alias(self, alias: str, canonical_name: str) -> None:
        """Register an alias for a canonical model name."""
        self._aliases[alias.lower()] = canonical_name

    def get_canonical_name(self, model_input: str) -> Optional[str]:
        """Get canonical name from model input (handles aliases)."""
        # Check direct match first
        if model_input in self._models:
            return model_input

        # Check aliases
        normalized_input = model_input.lower()
        if normalized_input in self._aliases:
            return self._aliases[normalized_input]

        # Check if it's a provider-specific name
        if "/" in model_input:
            provider_str, model_name = model_input.split("/", 1)
            # Try to find canonical name by provider model name
            for canonical, models in self._models.items():
                for model_info in models:
                    if model_info.provider_model_name == model_input:
                        return canonical

        return None

    def get_models_for_canonical(self, canonical_name: str) -> List[ModelInfo]:
        """Get all registered models for a canonical name."""
        return self._models.get(canonical_name, [])

    def get_all_canonical_names(self) -> List[str]:
        """Get all registered canonical model names."""
        return list(self._models.keys())

    def get_all_aliases(self) -> Dict[str, str]:
        """Get all registered aliases."""
        return self._aliases.copy()


class FuzzyMatcher:
    """Fuzzy matching for model names."""

    @staticmethod
    def normalize_name(name: str) -> str:
        """Normalize a model name for fuzzy matching."""
        # Remove special characters, convert to lowercase
        return re.sub(r'[^a-zA-Z0-9]', '', name).lower()

    @staticmethod
    def calculate_similarity(input_name: str, target_name: str) -> float:
        """Calculate similarity score between two names."""
        input_norm = FuzzyMatcher.normalize_name(input_name)
        target_norm = FuzzyMatcher.normalize_name(target_name)

        if input_norm == target_norm:
            return 1.0

        if input_norm in target_norm or target_norm in input_norm:
            return 0.8

        # Simple character overlap scoring
        common_chars = set(input_norm) & set(target_norm)
        total_chars = set(input_norm) | set(target_norm)

        if not total_chars:
            return 0.0

        return len(common_chars) / len(total_chars)

    def find_best_matches(self, input_name: str, candidates: List[str], 
                         threshold: float = 0.6) -> List[Tuple[str, float]]:
        """Find best matching candidates with scores."""
        matches = []

        for candidate in candidates:
            score = self.calculate_similarity(input_name, candidate)
            if score >= threshold:
                matches.append((candidate, score))

        # Sort by score descending
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches


class ModelMatcher:
    """Intelligent model matcher with fuzzy matching and resolution."""

    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self.fuzzy_matcher = FuzzyMatcher()

    async def resolve_model(self, model_input: str, 
                     preferred_providers: List[Provider] = None,
                     fallback_strategy: RoutingStrategy = RoutingStrategy.BEST_AVAILABLE,
                     requirements: ModelRequirements = None) -> Optional[ModelResolution]:
        """
        Resolve model input to best available provider/model combination.

        Supports:
        - Simple names: 'gpt4', 'claude', 'gemini'
        - Exact names: 'openai/gpt-4-turbo', 'anthropic/claude-3-sonnet'
        - Fuzzy matching: 'gpt4turbo' -> 'gpt-4-turbo'
        - Version resolution: 'gpt4' -> latest available GPT-4 variant
        """

        # Step 1: Try exact canonical name match
        canonical_name = self.registry.get_canonical_name(model_input)

        if canonical_name:
            resolution = self._resolve_canonical_model(
                canonical_name, preferred_providers, fallback_strategy, requirements
            )
            if resolution:
                return resolution

        # Step 2: Try fuzzy matching
        all_canonical_names = self.registry.get_all_canonical_names()
        all_aliases = list(self.registry.get_all_aliases().keys())
        all_candidates = all_canonical_names + all_aliases

        matches = self.fuzzy_matcher.find_best_matches(model_input, all_candidates)

        for match_name, confidence in matches:
            # Try to resolve the match
            canonical_name = self.registry.get_canonical_name(match_name)
            if canonical_name:
                resolution = self._resolve_canonical_model(
                    canonical_name, preferred_providers, fallback_strategy, requirements
                )
                if resolution:
                    resolution.confidence = confidence
                    return resolution

        # Step 3: Dynamic discovery - discover models from providers if not found
        await self._discover_and_register_missing_model(model_input, preferred_providers)

        # Step 4: Try resolution again after discovery
        canonical_name = self.registry.get_canonical_name(model_input)
        if canonical_name:
            resolution = self._resolve_canonical_model(
                canonical_name, preferred_providers, fallback_strategy, requirements
            )
            if resolution:
                return resolution

        # Step 5: Try fuzzy matching again after discovery
        all_canonical_names = self.registry.get_all_canonical_names()
        all_aliases = list(self.registry.get_all_aliases().keys())
        all_candidates = all_canonical_names + all_aliases

        matches = self.fuzzy_matcher.find_best_matches(model_input, all_candidates)

        for match_name, confidence in matches:
            canonical_name = self.registry.get_canonical_name(match_name)
            if canonical_name:
                resolution = self._resolve_canonical_model(
                    canonical_name, preferred_providers, fallback_strategy, requirements
                )
                if resolution:
                    resolution.confidence = confidence
                    return resolution

        return None

    async def _discover_and_register_missing_model(self, model_input: str, 
                                                  preferred_providers: List[Provider] = None) -> None:
        """
        Discover and register models from providers when a model is not found in the registry.

        Args:
            model_input: The model name/input that wasn't found
            preferred_providers: Optional list of preferred providers to check first
        """
        from schema_cat.provider_enum import Provider, _provider_api_key_available
        import logging

        logger = logging.getLogger("schema_cat.model_registry")

        # Determine which providers to check
        providers_to_check = []

        # If model_input contains a provider prefix (e.g., "openai/gpt-4"), prioritize that provider
        if "/" in model_input:
            provider_str, _ = model_input.split("/", 1)
            try:
                specific_provider = Provider(provider_str.lower())
                if _provider_api_key_available(specific_provider):
                    providers_to_check.append(specific_provider)
            except ValueError:
                pass  # Invalid provider name

        # Add preferred providers
        if preferred_providers:
            for provider in preferred_providers:
                if provider not in providers_to_check and _provider_api_key_available(provider):
                    providers_to_check.append(provider)

        # Add all available providers as fallback
        for provider in Provider:
            if provider not in providers_to_check and _provider_api_key_available(provider):
                providers_to_check.append(provider)

        # Discover models from each provider
        for provider in providers_to_check:
            try:
                logger.info(f"Discovering models from {provider.value} for '{model_input}'")
                models = await provider.init_models()

                # Register discovered models
                for model_data in models:
                    model_id = model_data.get('id')
                    if not model_id:
                        continue

                    # Check if this model matches what we're looking for
                    if self._model_matches_input(model_id, model_input):
                        # Normalize the canonical name to ensure consistency across providers
                        canonical_name = _normalize_canonical_name(model_id, provider)

                        # Create model capabilities from discovered data
                        capabilities = _estimate_model_capabilities(model_id, model_data)

                        # Register the discovered model
                        self.registry.register_model(
                            canonical_name=canonical_name,
                            provider=provider,
                            provider_model_name=model_id,
                            priority=100,  # Lower priority than hardcoded models
                            capabilities=capabilities
                        )

                        logger.info(f"Registered discovered model: {model_id} from {provider.value}")

                        # If this is an exact match, we can stop here
                        if model_id.lower() == model_input.lower():
                            return

            except Exception as e:
                logger.warning(f"Failed to discover models from {provider.value}: {e}")
                continue

    def _model_matches_input(self, model_id: str, model_input: str) -> bool:
        """
        Check if a discovered model ID matches the input we're looking for.

        Args:
            model_id: The model ID from the provider
            model_input: The input model name we're trying to resolve

        Returns:
            True if the model matches the input
        """
        # Normalize both strings for comparison
        normalized_id = self.fuzzy_matcher.normalize_name(model_id)
        normalized_input = self.fuzzy_matcher.normalize_name(model_input)

        # Check for exact match
        if normalized_id == normalized_input:
            return True

        # Check if input is contained in model ID
        if normalized_input in normalized_id:
            return True

        # Check for provider-specific format (e.g., "openai/gpt-4" matches "gpt-4")
        if "/" in model_input:
            _, model_part = model_input.split("/", 1)
            normalized_model_part = self.fuzzy_matcher.normalize_name(model_part)
            if normalized_model_part == normalized_id or normalized_model_part in normalized_id:
                return True

        # Check similarity score
        similarity = self.fuzzy_matcher.calculate_similarity(model_input, model_id)
        return similarity >= 0.7  # High threshold for automatic registration

    def _resolve_canonical_model(self, canonical_name: str,
                               preferred_providers: List[Provider] = None,
                               fallback_strategy: RoutingStrategy = RoutingStrategy.BEST_AVAILABLE,
                               requirements: ModelRequirements = None) -> Optional[ModelResolution]:
        """Resolve a canonical model name to a specific provider/model."""
        from schema_cat.provider_enum import _provider_api_key_available

        models = self.registry.get_models_for_canonical(canonical_name)
        if not models:
            return None

        # Filter by available providers (have API keys)
        available_models = [
            model for model in models 
            if _provider_api_key_available(model.provider)
        ]

        if not available_models:
            return None

        # Filter by requirements if specified
        if requirements:
            available_models = [
                model for model in available_models
                if self._meets_requirements(model, requirements)
            ]

            if not available_models:
                return None

        # Apply provider preferences and routing strategy
        if preferred_providers:
            # Filter models by preferred providers
            preferred_models = []
            for preferred_provider in preferred_providers:
                for model in available_models:
                    if model.provider == preferred_provider:
                        preferred_models.append(model)

            # Apply routing strategy to preferred models
            if preferred_models:
                selected_model = self._apply_strategy(preferred_models, fallback_strategy)
            else:
                # No models found in preferred providers, fall back to all available models
                selected_model = self._apply_strategy(available_models, fallback_strategy)
        else:
            # Apply fallback strategy to all available models
            selected_model = self._apply_strategy(available_models, fallback_strategy)

        if selected_model:
            return ModelResolution(
                provider=selected_model.provider,
                model_name=selected_model.provider_model_name,
                canonical_name=canonical_name
            )

        return None

    def _meets_requirements(self, model: ModelInfo, requirements: ModelRequirements) -> bool:
        """Check if a model meets the specified requirements."""
        caps = model.capabilities

        if requirements.min_context_length and caps.context_length < requirements.min_context_length:
            return False

        if requirements.supports_function_calling and not caps.supports_function_calling:
            return False

        if requirements.max_cost_per_1k_tokens and caps.cost_per_1k_tokens > requirements.max_cost_per_1k_tokens:
            return False

        if requirements.min_quality_score and caps.quality_score < requirements.min_quality_score:
            return False

        return True

    def _apply_strategy(self, models: List[ModelInfo], strategy: RoutingStrategy) -> Optional[ModelInfo]:
        """Apply routing strategy to select best model."""
        if not models:
            return None

        if strategy == RoutingStrategy.BEST_AVAILABLE:
            # Return highest priority (lowest priority number)
            return min(models, key=lambda x: x.priority)

        elif strategy == RoutingStrategy.CHEAPEST:
            # Return cheapest model
            return min(models, key=lambda x: x.capabilities.cost_per_1k_tokens)

        elif strategy == RoutingStrategy.HIGHEST_QUALITY:
            # Return highest quality model
            return max(models, key=lambda x: x.capabilities.quality_score)

        elif strategy == RoutingStrategy.MAX_CONTEXT:
            # Return model with the largest context window
            return max(models, key=lambda x: x.capabilities.context_length)

        else:
            # Default to best available
            return min(models, key=lambda x: x.priority)


# Global registry instance
_global_registry = None


def get_global_registry() -> ModelRegistry:
    """Get the global model registry instance."""
    global _global_registry
    if _global_registry is None:
        _global_registry = ModelRegistry()
    return _global_registry


def get_global_matcher() -> ModelMatcher:
    """Get the global model matcher instance."""
    return ModelMatcher(get_global_registry())


def _estimate_model_capabilities(model_id: str, model_data: dict) -> 'ModelCapabilities':
    """
    Estimate model capabilities based on model ID and available data.

    Args:
        model_id: The model ID from the provider
        model_data: Raw model data from the provider API

    Returns:
        ModelCapabilities with estimated values
    """
    # Start with provided data or defaults
    context_length = model_data.get('context_length', 4096)
    cost_per_1k_tokens = 0.0  # Default, would need pricing API for accurate costs
    quality_score = 0.5  # Default neutral score

    # Estimate capabilities based on model name patterns
    model_lower = model_id.lower()

    # Gemini model capabilities estimation
    if 'gemini' in model_lower:
        # Context length estimation based on model type
        if 'pro' in model_lower:
            if '2.0' in model_lower or '2.5' in model_lower:
                context_length = 2000000  # Gemini 2.0/2.5 Pro has very high context
                quality_score = 0.95
                cost_per_1k_tokens = 0.075
            else:
                context_length = 1000000  # Gemini 1.5 Pro
                quality_score = 0.9
                cost_per_1k_tokens = 0.05
        elif 'flash' in model_lower:
            if '2.0' in model_lower or '2.5' in model_lower:
                context_length = 1000000  # Gemini 2.0/2.5 Flash
                quality_score = 0.85
                cost_per_1k_tokens = 0.02
            else:
                context_length = 1000000  # Gemini 1.5 Flash
                quality_score = 0.8
                cost_per_1k_tokens = 0.015
        elif 'thinking' in model_lower:
            context_length = 300000  # Thinking models have good context but not as high as Pro
            quality_score = 0.9
            cost_per_1k_tokens = 0.04
        elif 'exp' in model_lower:
            # Experimental models - assume high quality but variable context
            context_length = 500000
            quality_score = 0.88
            cost_per_1k_tokens = 0.03
        else:
            # Default gemini model
            context_length = 32000
            quality_score = 0.75
            cost_per_1k_tokens = 0.01

    # GPT model capabilities estimation
    elif 'gpt' in model_lower:
        if 'gpt-4' in model_lower:
            if 'turbo' in model_lower:
                context_length = 128000
                quality_score = 0.95
                cost_per_1k_tokens = 0.06
            elif 'o1' in model_lower:
                context_length = 200000
                quality_score = 0.98
                cost_per_1k_tokens = 0.15
            else:
                context_length = 8192
                quality_score = 0.9
                cost_per_1k_tokens = 0.08
        elif 'gpt-3.5' in model_lower:
            context_length = 16385
            quality_score = 0.8
            cost_per_1k_tokens = 0.002

    # Claude model capabilities estimation
    elif 'claude' in model_lower:
        if 'opus' in model_lower:
            context_length = 200000
            quality_score = 0.95
            cost_per_1k_tokens = 0.075
        elif 'sonnet' in model_lower:
            context_length = 200000
            quality_score = 0.9
            cost_per_1k_tokens = 0.015
        elif 'haiku' in model_lower:
            context_length = 200000
            quality_score = 0.8
            cost_per_1k_tokens = 0.0025

    return ModelCapabilities(
        context_length=context_length,
        supports_function_calling=True,  # Assume true for modern models
        cost_per_1k_tokens=cost_per_1k_tokens,
        quality_score=quality_score
    )


def _normalize_canonical_name(model_id: str, provider: 'Provider') -> str:
    """
    Normalize model names to ensure similar models from different providers 
    use the same canonical name.

    Args:
        model_id: The model ID from the provider
        provider: The provider enum

    Returns:
        Normalized canonical name
    """
    from schema_cat.provider_enum import Provider

    # For COMET models, add google/ prefix to match OPENROUTER naming
    if provider == Provider.COMET:
        # If it's a gemini model without google/ prefix, add it
        if model_id.startswith('gemini-') and not model_id.startswith('google/'):
            model_id = f"google/{model_id}"

    # Group similar gemini models under the same canonical name for better routing
    if 'gemini' in model_id.lower():
        # Group all gemini models under the main canonical name that "gemini" alias points to
        # This allows routing strategies to choose between different gemini variants
        return "google/gemini-2.5-flash"

    # For other providers, use the model_id as-is
    return model_id


async def discover_and_register_models(provider: 'Provider' = None) -> Dict[str, int]:
    """
    Discover available models from provider APIs and register them in the global registry.

    Args:
        provider: Optional specific provider to discover models from. 
                 If None, discovers from all available providers.

    Returns:
        Dictionary mapping provider names to number of models discovered.
    """
    from schema_cat.provider_enum import Provider, _provider_api_key_available

    registry = get_global_registry()
    results = {}

    providers_to_check = [provider] if provider else list(Provider)

    for prov in providers_to_check:
        if not _provider_api_key_available(prov):
            results[prov.value] = 0
            continue

        try:
            models = await prov.init_models()
            count = 0

            for model_data in models:
                model_id = model_data.get('id')
                if not model_id:
                    continue

                # Normalize the canonical name to ensure consistency across providers
                canonical_name = _normalize_canonical_name(model_id, prov)

                # Create model capabilities from discovered data
                capabilities = _estimate_model_capabilities(model_id, model_data)

                # Register the discovered model
                registry.register_model(
                    canonical_name=canonical_name,
                    provider=prov,
                    provider_model_name=model_id,
                    priority=0,  # Default priority
                    capabilities=capabilities
                )
                count += 1

            results[prov.value] = count

        except Exception as e:
            import logging
            logger = logging.getLogger("schema_cat.model_registry")
            logger.warning(f"Failed to discover models from {prov.value}: {e}")
            results[prov.value] = 0

    return results
