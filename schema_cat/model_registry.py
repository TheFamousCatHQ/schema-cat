import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from schema_cat.provider_enum import Provider


@dataclass
class ModelRequirements:
    """Requirements for model selection."""
    min_context_length: Optional[int] = None
    supports_function_calling: Optional[bool] = None
    max_cost_per_1k_tokens: Optional[float] = None
    min_quality_score: Optional[float] = None

    def to_dict(self) -> Dict[str, any]:
        """Convert ModelRequirements to dictionary."""
        return {
            'min_context_length': self.min_context_length,
            'supports_function_calling': self.supports_function_calling,
            'max_cost_per_1k_tokens': self.max_cost_per_1k_tokens,
            'min_quality_score': self.min_quality_score
        }

    @classmethod
    def from_dict(cls, req_dict: Dict[str, any]) -> 'ModelRequirements':
        """Create ModelRequirements from dictionary."""
        return cls(
            min_context_length=req_dict.get('min_context_length'),
            supports_function_calling=req_dict.get('supports_function_calling'),
            max_cost_per_1k_tokens=req_dict.get('max_cost_per_1k_tokens'),
            min_quality_score=req_dict.get('min_quality_score')
        )


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
        # Use the new ModelCatalog internally
        self.catalog = ModelCatalog()
        # Keep old interface for backward compatibility
        self._models: Dict[str, List[ModelInfo]] = {}
        self._aliases: Dict[str, str] = {}
        self._provider_capabilities: Dict[Provider, Dict[str, ModelCapabilities]] = {}
        self._initialize_default_models()

    def clear(self):
        """Clear all model registry."""
        self._models.clear()
        self._aliases.clear()
        self.catalog.clear()
        self._provider_capabilities.clear()
        self._initialize_default_models()

    def _initialize_default_models(self):
        """Initialize registry - models will be discovered dynamically from providers."""
        # No static models or aliases - everything will be discovered dynamically
        pass

    def register_model(self, canonical_name: str, provider: Provider,
                       provider_model_name: str, priority: int = 0,
                       aliases: List[str] = None,
                       capabilities: ModelCapabilities = None) -> None:
        """Register a model with the registry."""
        # Register in new catalog
        self.catalog.register_model(
            canonical_name=canonical_name,
            provider=provider,
            provider_model_name=provider_model_name,
            priority=priority,
            capabilities=capabilities or ModelCapabilities()
        )

        # Keep old registry for backward compatibility
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
        # Register in new catalog
        self.catalog.register_alias(alias, canonical_name)

        # Keep old registry for backward compatibility
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
        # Remove everything up to and including the first slash
        name = re.sub(r'^.*?/', '', name)

        # Remove everything after and including the last colon
        name = re.sub(r':[^:]*$', '', name)

        # Remove special characters and convert to lowercase
        return re.sub(r'[^a-zA-Z0-9]', '', name).lower()

    @staticmethod
    def calculate_similarity(input_name: str, target_name: str) -> float:
        """Calculate similarity score between two names using improved algorithm."""
        input_norm = FuzzyMatcher.normalize_name(input_name)
        target_norm = FuzzyMatcher.normalize_name(target_name)

        # Exact match
        if input_norm == target_norm:
            return 1.0

        # Empty strings
        if not input_norm or not target_norm:
            return 0.0

        # Prefix matching - very important for model names
        if target_norm.startswith(input_norm):
            # Higher score for longer prefixes relative to target length
            return 0.95 - (0.1 * (len(target_norm) - len(input_norm)) / len(target_norm))

        if input_norm.startswith(target_norm):
            return 0.95 - (0.1 * (len(input_norm) - len(target_norm)) / len(input_norm))

        # Edit distance similarity (Levenshtein)
        edit_distance = FuzzyMatcher._levenshtein_distance(input_norm, target_norm)
        max_len = max(len(input_norm), len(target_norm))
        edit_similarity = 1.0 - (edit_distance / max_len)

        # Token-based similarity for hyphenated/underscore separated names
        input_tokens = set(re.split(r'[-_/]', input_norm))
        target_tokens = set(re.split(r'[-_/]', target_norm))

        if input_tokens and target_tokens:
            # Check if any input token matches any target token exactly
            common_tokens = input_tokens & target_tokens
            if common_tokens:
                token_similarity = len(common_tokens) / len(input_tokens | target_tokens)
                # Boost score if we have exact token matches
                token_similarity = 0.7 + (0.2 * token_similarity)
            else:
                token_similarity = 0.0
        else:
            token_similarity = 0.0

        # Character-based similarity (improved Jaccard)
        common_chars = set(input_norm) & set(target_norm)
        total_chars = set(input_norm) | set(target_norm)
        char_similarity = len(common_chars) / len(total_chars) if total_chars else 0.0

        # Return the best similarity score from different methods
        return max(edit_similarity, token_similarity, char_similarity)

    @staticmethod
    def _levenshtein_distance(s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings."""
        if len(s1) < len(s2):
            return FuzzyMatcher._levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def find_best_matches(self, input_name: str, candidates: List[str],
                          threshold: float = 0.7) -> List[Tuple[str, float]]:
        """Find best matching candidates with scores."""
        matches = []

        for candidate in candidates:
            score = self.calculate_similarity(input_name, candidate)
            if score >= threshold:
                matches.append((candidate, score))

        # Sort by score descending
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches


# New Resolution Pipeline Architecture

@dataclass
class ResolutionRequest:
    """Request object for model resolution pipeline."""
    model_input: str
    preferred_providers: Optional[List[Provider]] = None
    routing_strategy: RoutingStrategy = RoutingStrategy.BEST_AVAILABLE
    requirements: Optional[ModelRequirements] = None
    context: Optional['RequestContext'] = None


@dataclass
class ResolutionConfig:
    """Configuration for model resolution."""
    preferred_providers: List[Provider] = field(default_factory=list)
    routing_strategy: RoutingStrategy = RoutingStrategy.BEST_AVAILABLE
    requirements: Optional[ModelRequirements] = None
    overrides: Dict[str, Dict] = field(default_factory=dict)

    def apply_overrides(self, model_input: str) -> 'ResolutionConfig':
        """Return new config with overrides applied for specific model."""
        if model_input not in self.overrides:
            return self

        override = self.overrides[model_input]
        new_config = ResolutionConfig(
            preferred_providers=self.preferred_providers.copy(),
            routing_strategy=self.routing_strategy,
            requirements=self.requirements,
            overrides=self.overrides.copy()
        )

        if 'preferred_providers' in override:
            new_config.preferred_providers = [Provider(p) for p in override['preferred_providers']]
        if 'routing_strategy' in override:
            new_config.routing_strategy = RoutingStrategy(override['routing_strategy'])
        if 'requirements' in override:
            new_config.requirements = ModelRequirements(**override['requirements'])

        return new_config


class ModelCatalog:
    """Unified catalog for all model information."""

    def __init__(self):
        self.models: Dict[str, List[ModelInfo]] = {}
        self.aliases: Dict[str, str] = {}
        self.fuzzy_index = FuzzyMatcher()

    def clear(self):
        """Clear all model information."""
        self.models = {}
        self.aliases = {}
        self.fuzzy_index = FuzzyMatcher()

    def register_model(self, canonical_name: str, provider: Provider,
                       provider_model_name: str, priority: int = 50,
                       capabilities: Optional['ModelCapabilities'] = None):
        """Register a model in the catalog."""
        if canonical_name not in self.models:
            self.models[canonical_name] = []

        model_info = ModelInfo(
            canonical_name=canonical_name,
            provider=provider,
            provider_model_name=provider_model_name,
            priority=priority,
            capabilities=capabilities or ModelCapabilities()
        )

        self.models[canonical_name].append(model_info)
        # Sort by priority (lower number = higher priority)
        self.models[canonical_name].sort(key=lambda x: x.priority)

    def register_alias(self, alias: str, canonical_name: str):
        """Register an alias for a canonical model name."""
        self.aliases[alias] = canonical_name

    def find_exact(self, name: str) -> List[ModelInfo]:
        """Find models by exact canonical name or alias."""
        # Check direct canonical name
        if name in self.models:
            return self.models[name].copy()

        # Check aliases
        if name in self.aliases:
            canonical = self.aliases[name]
            if canonical in self.models:
                return self.models[canonical].copy()

        return []

    def find_fuzzy(self, name: str, threshold: float = 0.7) -> List[Tuple[ModelInfo, float]]:
        """Find models using fuzzy matching with confidence scores."""
        results = []

        # Get all possible candidates
        all_names = list(self.models.keys()) + list(self.aliases.keys())
        matches = self.fuzzy_index.find_best_matches(name, all_names, threshold)

        for match_name, confidence in matches:
            models = self.find_exact(match_name)
            for model in models:
                results.append((model, confidence))

        # Sort by confidence descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def get_all_canonical_names(self) -> List[str]:
        """Get all canonical model names."""
        return list(self.models.keys())

    def get_all_aliases(self) -> Dict[str, str]:
        """Get all aliases."""
        return self.aliases.copy()


class BaseResolver(ABC):
    """Base class for model resolvers."""

    def __init__(self, catalog: ModelCatalog):
        self.catalog = catalog

    @abstractmethod
    async def resolve(self, request: ResolutionRequest) -> Optional[ModelResolution]:
        """Attempt to resolve the model request."""
        pass


class ModelSelector:
    """Handles model selection based on strategy and requirements."""

    def select_best(self, candidates: List[ModelInfo],
                    strategy: RoutingStrategy,
                    requirements: Optional[ModelRequirements] = None) -> Optional[ModelInfo]:
        """Select the best model from candidates based on strategy and requirements."""
        if not candidates:
            return None

        # Filter by requirements first
        filtered = self._filter_by_requirements(candidates, requirements)
        if not filtered:
            return None

        # Apply strategy
        return self._apply_strategy(filtered, strategy)

    def _filter_by_requirements(self, models: List[ModelInfo],
                                requirements: Optional[ModelRequirements]) -> List[ModelInfo]:
        """Filter models by requirements."""
        if not requirements:
            return models

        filtered = []
        for model in models:
            if self._meets_requirements(model, requirements):
                filtered.append(model)

        return filtered

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
            return min(models, key=lambda x: x.priority)
        elif strategy == RoutingStrategy.CHEAPEST:
            return min(models, key=lambda x: x.capabilities.cost_per_1k_tokens)
        elif strategy == RoutingStrategy.HIGHEST_QUALITY:
            return max(models, key=lambda x: x.capabilities.quality_score)
        elif strategy == RoutingStrategy.MAX_CONTEXT:
            return max(models, key=lambda x: x.capabilities.context_length)
        else:
            return min(models, key=lambda x: x.priority)


class ExactMatchResolver(BaseResolver):
    """Resolves exact canonical names and provider/model format."""

    def __init__(self, catalog: ModelCatalog, selector: ModelSelector):
        super().__init__(catalog)
        self.selector = selector

    async def resolve(self, request: ResolutionRequest) -> Optional[ModelResolution]:
        """Try exact matching first."""
        from schema_cat.provider_enum import _provider_api_key_available

        # Handle provider/model format (e.g., "openai/gpt-4")
        if "/" in request.model_input:
            provider_str, model_name = request.model_input.split("/", 1)
            try:
                provider = Provider(provider_str.lower())
                if _provider_api_key_available(provider):
                    # Look for exact match in catalog
                    candidates = self.catalog.find_exact(request.model_input)
                    if candidates:
                        # Filter by the specific provider
                        provider_candidates = [c for c in candidates if c.provider == provider]
                        if provider_candidates:
                            selected = self.selector.select_best(
                                provider_candidates,
                                request.routing_strategy,
                                request.requirements
                            )
                            if selected:
                                return ModelResolution(
                                    provider=selected.provider,
                                    model_name=selected.provider_model_name,
                                    canonical_name=selected.canonical_name,
                                    confidence=1.0
                                )
            except ValueError:
                pass  # Invalid provider name

        # Try exact canonical name or alias match
        candidates = self.catalog.find_exact(request.model_input)
        if candidates:
            # Filter by available providers
            available_candidates = [
                c for c in candidates
                if _provider_api_key_available(c.provider)
            ]

            if available_candidates:
                # Apply provider preferences
                if request.preferred_providers:
                    preferred_candidates = [
                        c for c in available_candidates
                        if c.provider in request.preferred_providers
                    ]
                    if preferred_candidates:
                        available_candidates = preferred_candidates

                selected = self.selector.select_best(
                    available_candidates,
                    request.routing_strategy,
                    request.requirements
                )

                if selected:
                    return ModelResolution(
                        provider=selected.provider,
                        model_name=selected.provider_model_name,
                        canonical_name=selected.canonical_name,
                        confidence=1.0
                    )

        return None


class FuzzyMatchResolver(BaseResolver):
    """Resolves models using fuzzy matching."""

    def __init__(self, catalog: ModelCatalog, selector: ModelSelector, threshold: float = 0.7):
        super().__init__(catalog)
        self.selector = selector
        self.threshold = threshold

    async def resolve(self, request: ResolutionRequest) -> Optional[ModelResolution]:
        """Try fuzzy matching."""
        from schema_cat.provider_enum import _provider_api_key_available

        fuzzy_matches = self.catalog.find_fuzzy(request.model_input, self.threshold)

        # Collect all valid matching models
        valid_models = []
        for model, confidence in fuzzy_matches:
            if _provider_api_key_available(model.provider):
                # Check if this model meets requirements
                if request.requirements and not self.selector._meets_requirements(model, request.requirements):
                    continue

                # Add confidence to model for later reference
                model._confidence = confidence
                valid_models.append(model)

        if not valid_models:
            return None

        # Apply provider preferences if specified
        if request.preferred_providers:
            preferred_models = [
                model for model in valid_models
                if model.provider in request.preferred_providers
            ]
            if preferred_models:
                valid_models = preferred_models

        # Apply routing strategy to select the best model
        selected_model = self.selector._apply_strategy(valid_models, request.routing_strategy)

        if selected_model:
            return ModelResolution(
                provider=selected_model.provider,
                model_name=selected_model.provider_model_name,
                canonical_name=selected_model.canonical_name,
                confidence=getattr(selected_model, '_confidence', 1.0)
            )

        return None


class DiscoveryResolver(BaseResolver):
    """Discovers models dynamically from provider APIs."""

    def __init__(self, catalog: ModelCatalog, selector: ModelSelector):
        super().__init__(catalog)
        self.selector = selector

    async def resolve(self, request: ResolutionRequest) -> Optional[ModelResolution]:
        """Try dynamic model discovery."""
        await self._discover_and_register_missing_model(request)

        # After discovery, try exact match again
        exact_resolver = ExactMatchResolver(self.catalog, self.selector)
        return await exact_resolver.resolve(request)

    async def _discover_and_register_missing_model(self, request: ResolutionRequest) -> None:
        """Discover and register models from providers."""
        from schema_cat.provider_enum import Provider, _provider_api_key_available
        import logging

        logger = logging.getLogger("schema_cat.model_registry")
        model_input = request.model_input

        # Determine which providers to check
        providers_to_check = []

        # If model_input contains a provider prefix, prioritize that provider
        if "/" in model_input:
            provider_str, _ = model_input.split("/", 1)
            try:
                specific_provider = Provider(provider_str.lower())
                if _provider_api_key_available(specific_provider):
                    providers_to_check.append(specific_provider)
            except ValueError:
                pass

        # Add preferred providers
        if request.preferred_providers:
            for provider in request.preferred_providers:
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
                        # Normalize the canonical name
                        canonical_name = _normalize_canonical_name(model_id)

                        # Create model capabilities
                        capabilities = _estimate_model_capabilities(model_id, model_data)

                        # Register the discovered model
                        self.catalog.register_model(
                            canonical_name=canonical_name,
                            provider=provider,
                            provider_model_name=model_id,
                            priority=100,  # Lower priority than hardcoded models
                            capabilities=capabilities
                        )

                        logger.debug(f"Registered discovered model: {model_id} from {provider.value}")

                        # If exact match, we can stop here
                        if model_id.lower() == model_input.lower():
                            return

            except Exception as e:
                logger.warning(f"Failed to discover models from {provider.value}: {e}")
                continue

    def _model_matches_input(self, model_id: str, model_input: str) -> bool:
        """Check if a discovered model ID matches the input."""
        # Use the catalog's fuzzy matcher for consistency
        similarity = self.catalog.fuzzy_index.calculate_similarity(model_input, model_id)
        return similarity >= 0.7  # High threshold for automatic registration


class FallbackResolver(BaseResolver):
    """Fallback resolver that handles edge cases."""

    def __init__(self, catalog: ModelCatalog, selector: ModelSelector):
        super().__init__(catalog)
        self.selector = selector

    async def resolve(self, request: ResolutionRequest) -> Optional[ModelResolution]:
        """Last resort fallback resolution."""
        # Could implement fallback to default models, etc.
        return None


class ModelResolutionPipeline:
    """Pipeline of specialized resolvers for model resolution."""

    def __init__(self, catalog: ModelCatalog):
        self.catalog = catalog
        self.selector = ModelSelector()
        self.resolvers = [
            ExactMatchResolver(catalog, self.selector),
            FuzzyMatchResolver(catalog, self.selector),
            DiscoveryResolver(catalog, self.selector),
            FallbackResolver(catalog, self.selector)
        ]

    async def resolve(self, request: ResolutionRequest) -> Optional[ModelResolution]:
        """Resolve model through the pipeline."""
        for resolver in self.resolvers:
            result = await resolver.resolve(request)
            if result:
                return result
        return None


class PipelineModelMatcher:
    """New pipeline-based model matcher that replaces the old complex ModelMatcher."""

    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self.pipeline = ModelResolutionPipeline(registry.catalog)

    async def resolve_model(self, model_input: str,
                            preferred_providers: List[Provider] = None,
                            fallback_strategy: RoutingStrategy = RoutingStrategy.BEST_AVAILABLE,
                            requirements: ModelRequirements = None) -> Optional[ModelResolution]:
        """
        Resolve model input to best available provider/model combination using the new pipeline.

        This is the new, simplified implementation that replaces the complex 5-step process
        in the old ModelMatcher with a clean pipeline approach.
        """
        request = ResolutionRequest(
            model_input=model_input,
            preferred_providers=preferred_providers,
            routing_strategy=fallback_strategy,
            requirements=requirements
        )

        return await self.pipeline.resolve(request)


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

        # Collect all matching canonical models
        matching_canonical_names = []
        for match_name, confidence in matches:
            canonical_name = self.registry.get_canonical_name(match_name)
            if canonical_name and canonical_name not in matching_canonical_names:
                matching_canonical_names.append(canonical_name)

        # If we have multiple matches, we need to apply strategy across all of them
        if len(matching_canonical_names) > 1:
            # Collect all models from all matching canonical names
            all_matching_models = []
            for canonical_name in matching_canonical_names:
                models = self.registry.get_models_for_canonical(canonical_name)
                if models:
                    # Filter by available providers and requirements
                    from schema_cat.provider_enum import _provider_api_key_available
                    available_models = [
                        model for model in models
                        if _provider_api_key_available(model.provider)
                    ]

                    if requirements:
                        available_models = [
                            model for model in available_models
                            if self._meets_requirements(model, requirements)
                        ]

                    # Add canonical name to each model for later reference
                    for model in available_models:
                        model._canonical_name = canonical_name
                        all_matching_models.append(model)

            if all_matching_models:
                # Apply provider preferences
                if preferred_providers:
                    preferred_models = []
                    for preferred_provider in preferred_providers:
                        for model in all_matching_models:
                            if model.provider == preferred_provider:
                                preferred_models.append(model)

                    if preferred_models:
                        selected_model = self._apply_strategy(preferred_models, fallback_strategy)
                    else:
                        selected_model = self._apply_strategy(all_matching_models, fallback_strategy)
                else:
                    selected_model = self._apply_strategy(all_matching_models, fallback_strategy)

                if selected_model:
                    return ModelResolution(
                        provider=selected_model.provider,
                        model_name=selected_model.provider_model_name,
                        canonical_name=selected_model._canonical_name
                    )
        else:
            # Single match or no matches - use original logic
            for match_name, confidence in matches:
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
                        canonical_name = _normalize_canonical_name(model_id)

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

                        logger.debug(f"Registered discovered model: {model_id} from {provider.value}")

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


def get_global_matcher() -> PipelineModelMatcher:
    """Get the global model matcher instance using the new pipeline architecture."""
    return PipelineModelMatcher(get_global_registry())


def get_legacy_matcher() -> ModelMatcher:
    """Get the legacy model matcher instance for backward compatibility."""
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
    # TODO WHAT THE FUCK
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


def _normalize_canonical_name(model_id: str) -> str:
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
                canonical_name = _normalize_canonical_name(model_id)

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
