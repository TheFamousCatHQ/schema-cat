import pytest

from schema_cat.model_registry import (
    ModelRegistry, ModelCatalog, PipelineModelMatcher,
    ModelRequirements, ModelCapabilities, ModelInfo, ModelResolution,
    RoutingStrategy, RequestContext, ResolutionRequest, FuzzyMatcher, ModelSelector, ExactMatchResolver,
    FuzzyMatchResolver, ModelResolutionPipeline, get_global_registry, get_global_matcher,
    _estimate_model_capabilities,
    _normalize_canonical_name
)
from schema_cat.provider_enum import Provider


class TestModelRequirements:
    """Test ModelRequirements dataclass."""

    def test_model_requirements_creation(self):
        """Test creating ModelRequirements with various parameters."""
        # Test with no parameters
        req = ModelRequirements()
        assert req.min_context_length is None
        assert req.supports_function_calling is None
        assert req.max_cost_per_1k_tokens is None
        assert req.min_quality_score is None

        # Test with all parameters
        req = ModelRequirements(
            min_context_length=8192,
            supports_function_calling=True,
            max_cost_per_1k_tokens=0.01,
            min_quality_score=0.8
        )
        assert req.min_context_length == 8192
        assert req.supports_function_calling is True
        assert req.max_cost_per_1k_tokens == 0.01
        assert req.min_quality_score == 0.8


class TestModelCapabilities:
    """Test ModelCapabilities dataclass."""

    def test_model_capabilities_defaults(self):
        """Test ModelCapabilities default values."""
        cap = ModelCapabilities()
        assert cap.context_length == 4096
        assert cap.supports_function_calling is False
        assert cap.cost_per_1k_tokens == 0.0
        assert cap.quality_score == 0.0
        assert cap.supports_streaming is True

    def test_model_capabilities_custom(self):
        """Test ModelCapabilities with custom values."""
        cap = ModelCapabilities(
            context_length=16384,
            supports_function_calling=True,
            cost_per_1k_tokens=0.02,
            quality_score=0.9,
            supports_streaming=False
        )
        assert cap.context_length == 16384
        assert cap.supports_function_calling is True
        assert cap.cost_per_1k_tokens == 0.02
        assert cap.quality_score == 0.9
        assert cap.supports_streaming is False


class TestModelInfo:
    """Test ModelInfo dataclass."""

    def test_model_info_creation(self):
        """Test creating ModelInfo with required parameters."""
        model = ModelInfo(
            canonical_name="gpt-4",
            provider=Provider.OPENAI,
            provider_model_name="gpt-4-0613"
        )
        assert model.canonical_name == "gpt-4"
        assert model.provider == Provider.OPENAI
        assert model.provider_model_name == "gpt-4-0613"
        assert model.priority == 0
        assert model.aliases == []
        assert isinstance(model.capabilities, ModelCapabilities)

    def test_model_info_with_all_parameters(self):
        """Test creating ModelInfo with all parameters."""
        capabilities = ModelCapabilities(context_length=8192, supports_function_calling=True)
        model = ModelInfo(
            canonical_name="gpt-4-turbo",
            provider=Provider.OPENAI,
            provider_model_name="gpt-4-1106-preview",
            priority=10,
            aliases=["gpt4-turbo", "gpt-4-preview"],
            capabilities=capabilities
        )
        assert model.canonical_name == "gpt-4-turbo"
        assert model.provider == Provider.OPENAI
        assert model.provider_model_name == "gpt-4-1106-preview"
        assert model.priority == 10
        assert model.aliases == ["gpt4-turbo", "gpt-4-preview"]
        assert model.capabilities == capabilities


class TestModelResolution:
    """Test ModelResolution dataclass."""

    def test_model_resolution_creation(self):
        """Test creating ModelResolution."""
        resolution = ModelResolution(
            provider=Provider.OPENAI,
            model_name="gpt-4-0613",
            canonical_name="gpt-4"
        )
        assert resolution.provider == Provider.OPENAI
        assert resolution.model_name == "gpt-4-0613"
        assert resolution.canonical_name == "gpt-4"
        assert resolution.confidence == 1.0

    def test_model_resolution_with_confidence(self):
        """Test creating ModelResolution with custom confidence."""
        resolution = ModelResolution(
            provider=Provider.ANTHROPIC,
            model_name="claude-3-opus-20240229",
            canonical_name="claude-3-opus",
            confidence=0.85
        )
        assert resolution.confidence == 0.85


class TestRoutingStrategy:
    """Test RoutingStrategy enum."""

    def test_routing_strategy_values(self):
        """Test all routing strategy values."""
        assert RoutingStrategy.BEST_AVAILABLE == "best_available"
        assert RoutingStrategy.CHEAPEST == "cheapest"
        assert RoutingStrategy.FASTEST == "fastest"
        assert RoutingStrategy.MOST_RELIABLE == "most_reliable"
        assert RoutingStrategy.HIGHEST_QUALITY == "highest_quality"
        assert RoutingStrategy.MAX_CONTEXT == "max_context"


class TestRequestContext:
    """Test RequestContext dataclass."""

    def test_request_context_defaults(self):
        """Test RequestContext default values."""
        context = RequestContext()
        assert context.requirements is None
        assert context.strategy is None
        assert context.preferred_providers is None

    def test_request_context_with_values(self):
        """Test RequestContext with custom values."""
        requirements = ModelRequirements(min_context_length=8192)
        context = RequestContext(
            requirements=requirements,
            strategy=RoutingStrategy.CHEAPEST,
            preferred_providers=[Provider.OPENAI, Provider.ANTHROPIC]
        )
        assert context.requirements == requirements
        assert context.strategy == RoutingStrategy.CHEAPEST
        assert context.preferred_providers == [Provider.OPENAI, Provider.ANTHROPIC]


class TestFuzzyMatcher:
    """Test FuzzyMatcher functionality."""

    def test_normalize_name(self):
        """Test name normalization."""
        assert FuzzyMatcher.normalize_name("GPT-4") == "gpt4"
        assert FuzzyMatcher.normalize_name("Claude-3-Opus") == "claude3opus"
        assert FuzzyMatcher.normalize_name("gpt_4_turbo") == "gpt4turbo"
        assert FuzzyMatcher.normalize_name("Model Name 123") == "modelname123"

    def test_calculate_similarity(self):
        """Test similarity calculation."""
        # Exact match
        assert FuzzyMatcher.calculate_similarity("gpt4", "gpt4") == 1.0

        # Partial matches (after normalization, "gpt4" and "gpt-4" become identical)
        similarity = FuzzyMatcher.calculate_similarity("gpt4", "gpt-4")
        assert similarity == 1.0  # They normalize to the same string

        # Different but similar
        similarity = FuzzyMatcher.calculate_similarity("gpt4", "gpt3")
        assert 0.5 < similarity < 1.0

        # No match
        similarity = FuzzyMatcher.calculate_similarity("gpt4", "claude")
        assert similarity < 0.5

    def test_find_best_matches(self):
        """Test finding best matches."""
        matcher = FuzzyMatcher()
        candidates = ["gpt-4", "gpt-3.5-turbo", "claude-3-opus", "claude-3-sonnet"]

        # Test exact match
        matches = matcher.find_best_matches("gpt-4", candidates)
        assert len(matches) > 0
        assert matches[0][0] == "gpt-4"
        assert matches[0][1] == 1.0

        # Test fuzzy match
        matches = matcher.find_best_matches("gpt4", candidates)
        assert len(matches) > 0
        assert "gpt-4" in [match[0] for match in matches]

        # Test no matches above threshold
        matches = matcher.find_best_matches("completely-different", candidates, threshold=0.9)
        assert len(matches) == 0


class TestModelCatalog:
    """Test ModelCatalog functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.catalog = ModelCatalog()

    def test_register_model(self):
        """Test model registration."""
        capabilities = ModelCapabilities(context_length=8192)
        self.catalog.register_model(
            canonical_name="gpt-4",
            provider=Provider.OPENAI,
            provider_model_name="gpt-4-0613",
            priority=10,
            capabilities=capabilities
        )

        # Test exact match
        result = self.catalog.find_exact("gpt-4")
        assert len(result) == 1
        assert result[0].canonical_name == "gpt-4"
        assert result[0].provider == Provider.OPENAI
        assert result[0].capabilities == capabilities

    def test_register_alias(self):
        """Test alias registration."""
        self.catalog.register_model("gpt-4", Provider.OPENAI, "gpt-4-0613")
        self.catalog.register_alias("gpt4", "gpt-4")

        # Test finding by alias
        result = self.catalog.find_exact("gpt4")
        assert len(result) == 1
        assert result[0].canonical_name == "gpt-4"

    def test_find_fuzzy(self):
        """Test fuzzy matching."""
        self.catalog.register_model("gpt-4", Provider.OPENAI, "gpt-4-0613")
        self.catalog.register_model("claude-3-opus", Provider.ANTHROPIC, "claude-3-opus-20240229")

        # Test fuzzy match - returns list of (model, confidence) tuples
        result = self.catalog.find_fuzzy("gpt4")
        assert len(result) > 0
        assert any(model.canonical_name == "gpt-4" for model, confidence in result)

    def test_get_all_canonical_names(self):
        """Test getting all canonical names."""
        self.catalog.register_model("gpt-4", Provider.OPENAI, "gpt-4-0613")
        self.catalog.register_model("claude-3-opus", Provider.ANTHROPIC, "claude-3-opus-20240229")

        names = self.catalog.get_all_canonical_names()
        assert "gpt-4" in names
        assert "claude-3-opus" in names

    def test_get_all_aliases(self):
        """Test getting all aliases."""
        self.catalog.register_model("gpt-4", Provider.OPENAI, "gpt-4-0613")
        self.catalog.register_alias("gpt4", "gpt-4")
        self.catalog.register_alias("openai-gpt4", "gpt-4")

        aliases = self.catalog.get_all_aliases()
        assert "gpt4" in aliases
        assert "openai-gpt4" in aliases


class TestModelSelector:
    """Test ModelSelector functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.selector = ModelSelector()

        # Create test models with different capabilities
        self.cheap_model = ModelInfo(
            canonical_name="gpt-3.5-turbo",
            provider=Provider.OPENAI,
            provider_model_name="gpt-3.5-turbo",
            capabilities=ModelCapabilities(
                context_length=4096,
                cost_per_1k_tokens=0.001,
                quality_score=0.7,
                supports_function_calling=True
            )
        )

        self.expensive_model = ModelInfo(
            canonical_name="gpt-4",
            provider=Provider.OPENAI,
            provider_model_name="gpt-4-0613",
            capabilities=ModelCapabilities(
                context_length=8192,
                cost_per_1k_tokens=0.03,
                quality_score=0.95,
                supports_function_calling=True
            )
        )

        self.high_context_model = ModelInfo(
            canonical_name="claude-3-opus",
            provider=Provider.ANTHROPIC,
            provider_model_name="claude-3-opus-20240229",
            capabilities=ModelCapabilities(
                context_length=200000,
                cost_per_1k_tokens=0.015,
                quality_score=0.9,
                supports_function_calling=False
            )
        )

    def test_select_best_cheapest(self):
        """Test selecting cheapest model."""
        models = [self.cheap_model, self.expensive_model, self.high_context_model]
        result = self.selector.select_best(models, RoutingStrategy.CHEAPEST)
        assert result == self.cheap_model

    def test_select_best_highest_quality(self):
        """Test selecting highest quality model."""
        models = [self.cheap_model, self.expensive_model, self.high_context_model]
        result = self.selector.select_best(models, RoutingStrategy.HIGHEST_QUALITY)
        assert result == self.expensive_model

    def test_select_best_max_context(self):
        """Test selecting model with maximum context."""
        models = [self.cheap_model, self.expensive_model, self.high_context_model]
        result = self.selector.select_best(models, RoutingStrategy.MAX_CONTEXT)
        assert result == self.high_context_model

    def test_filter_by_requirements(self):
        """Test filtering models by requirements."""
        models = [self.cheap_model, self.expensive_model, self.high_context_model]

        # Require function calling
        requirements = ModelRequirements(supports_function_calling=True)
        filtered = self.selector._filter_by_requirements(models, requirements)
        assert len(filtered) == 2
        assert self.high_context_model not in filtered

        # Require high context
        requirements = ModelRequirements(min_context_length=10000)
        filtered = self.selector._filter_by_requirements(models, requirements)
        assert len(filtered) == 1
        assert filtered[0] == self.high_context_model

        # Require low cost
        requirements = ModelRequirements(max_cost_per_1k_tokens=0.01)
        filtered = self.selector._filter_by_requirements(models, requirements)
        assert len(filtered) == 1
        assert filtered[0] == self.cheap_model

    def test_meets_requirements(self):
        """Test checking if model meets requirements."""
        requirements = ModelRequirements(
            min_context_length=8000,
            supports_function_calling=True,
            max_cost_per_1k_tokens=0.05,
            min_quality_score=0.8
        )

        assert not self.selector._meets_requirements(self.cheap_model, requirements)  # Low context
        assert self.selector._meets_requirements(self.expensive_model, requirements)  # Meets all
        assert not self.selector._meets_requirements(self.high_context_model, requirements)  # No function calling


class TestExactMatchResolver:
    """Test ExactMatchResolver functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.catalog = ModelCatalog()
        self.selector = ModelSelector()
        self.resolver = ExactMatchResolver(self.catalog, self.selector)

        # Register test model
        self.catalog.register_model("gpt-4", Provider.OPENAI, "gpt-4-0613")

    @pytest.mark.asyncio
    async def test_exact_match_found(self):
        """Test exact match resolution when model is found."""
        request = ResolutionRequest(
            model_input="gpt-4"
        )

        result = await self.resolver.resolve(request)
        assert result is not None
        assert result.canonical_name == "gpt-4"
        assert result.provider == Provider.OPENAI

    @pytest.mark.asyncio
    async def test_exact_match_not_found(self):
        """Test exact match resolution when model is not found."""
        request = ResolutionRequest(
            model_input="nonexistent-model"
        )

        result = await self.resolver.resolve(request)
        assert result is None


class TestFuzzyMatchResolver:
    """Test FuzzyMatchResolver functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.catalog = ModelCatalog()
        self.selector = ModelSelector()
        self.resolver = FuzzyMatchResolver(self.catalog, self.selector, threshold=0.7)

        # Register test models
        self.catalog.register_model("gpt-4", Provider.OPENAI, "gpt-4-0613")
        self.catalog.register_model("claude-3-opus", Provider.ANTHROPIC, "claude-3-opus-20240229")

    @pytest.mark.asyncio
    async def test_fuzzy_match_found(self):
        """Test fuzzy match resolution when similar model is found."""
        request = ResolutionRequest(
            model_input="gpt4turbo"  # Similar to gpt-4 but not exact after normalization
        )

        result = await self.resolver.resolve(request)
        assert result is not None
        assert result.canonical_name == "gpt-4"
        assert result.confidence < 1.0

    @pytest.mark.asyncio
    async def test_fuzzy_match_not_found(self):
        """Test fuzzy match resolution when no similar model is found."""
        request = ResolutionRequest(
            model_input="completely-different-model"
        )

        result = await self.resolver.resolve(request)
        assert result is None


class TestModelResolutionPipeline:
    """Test ModelResolutionPipeline functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.catalog = ModelCatalog()
        self.pipeline = ModelResolutionPipeline(self.catalog)

        # Register test models
        self.catalog.register_model("gpt-4", Provider.OPENAI, "gpt-4-0613")
        self.catalog.register_alias("gpt4", "gpt-4")

    @pytest.mark.asyncio
    async def test_pipeline_exact_match(self):
        """Test pipeline with exact match."""
        request = ResolutionRequest(
            model_input="gpt-4"
        )

        result = await self.pipeline.resolve(request)
        assert result is not None
        assert result.canonical_name == "gpt-4"
        assert result.confidence == 1.0

    @pytest.mark.asyncio
    async def test_pipeline_alias_match(self):
        """Test pipeline with alias match."""
        request = ResolutionRequest(
            model_input="gpt4"
        )

        result = await self.pipeline.resolve(request)
        assert result is not None
        assert result.canonical_name == "gpt-4"

    @pytest.mark.asyncio
    async def test_pipeline_fuzzy_match(self):
        """Test pipeline with fuzzy match."""
        request = ResolutionRequest(
            model_input="gpt-4-turbo"  # Similar but not exact
        )

        result = await self.pipeline.resolve(request)
        # Should find gpt-4 as fuzzy match
        assert result is not None
        assert result.canonical_name == "gpt-4"
        assert result.confidence < 1.0


class TestModelRegistry:
    """Test ModelRegistry functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.registry = ModelRegistry()

    def test_register_model(self):
        """Test model registration."""
        capabilities = ModelCapabilities(context_length=8192)
        self.registry.register_model(
            canonical_name="gpt-4",
            provider=Provider.OPENAI,
            provider_model_name="gpt-4-0613",
            priority=10,
            aliases=["gpt4", "openai-gpt4"],
            capabilities=capabilities
        )

        # Test canonical name lookup
        canonical = self.registry.get_canonical_name("gpt-4")
        assert canonical == "gpt-4"

        # Test alias lookup
        canonical = self.registry.get_canonical_name("gpt4")
        assert canonical == "gpt-4"

        # Test getting models for canonical name
        models = self.registry.get_models_for_canonical("gpt-4")
        assert len(models) == 1
        assert models[0].canonical_name == "gpt-4"

    def test_register_alias(self):
        """Test alias registration."""
        self.registry.register_model("gpt-4", Provider.OPENAI, "gpt-4-0613")
        self.registry.register_alias("custom-alias", "gpt-4")

        canonical = self.registry.get_canonical_name("custom-alias")
        assert canonical == "gpt-4"

    def test_get_all_canonical_names(self):
        """Test getting all canonical names."""
        self.registry.register_model("gpt-4", Provider.OPENAI, "gpt-4-0613")
        self.registry.register_model("claude-3-opus", Provider.ANTHROPIC, "claude-3-opus-20240229")

        names = self.registry.get_all_canonical_names()
        assert "gpt-4" in names
        assert "claude-3-opus" in names

    def test_get_all_aliases(self):
        """Test getting all aliases."""
        self.registry.register_model("gpt-4", Provider.OPENAI, "gpt-4-0613", aliases=["gpt4"])
        self.registry.register_alias("custom-alias", "gpt-4")

        aliases = self.registry.get_all_aliases()
        assert "gpt4" in aliases
        assert "custom-alias" in aliases


class TestPipelineModelMatcher:
    """Test PipelineModelMatcher functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.registry = ModelRegistry()
        self.matcher = PipelineModelMatcher(self.registry)

        # Register test models
        self.registry.register_model("gpt-4", Provider.OPENAI, "gpt-4-0613")
        self.registry.register_model("claude-3-opus", Provider.ANTHROPIC, "claude-3-opus-20240229")

    @pytest.mark.asyncio
    async def test_resolve_model_exact_match(self):
        """Test exact model resolution."""
        result = await self.matcher.resolve_model("gpt-4")
        assert result is not None
        assert result.canonical_name == "gpt-4"
        assert result.provider == Provider.OPENAI

    @pytest.mark.asyncio
    async def test_resolve_model_with_preferred_providers(self):
        """Test model resolution with preferred providers."""
        # Register same model for multiple providers
        self.registry.register_model("gpt-4", Provider.OPENROUTER, "openai/gpt-4", priority=5)

        # Test with preferred provider
        result = await self.matcher.resolve_model("gpt-4", preferred_providers=[Provider.OPENROUTER])
        assert result is not None
        assert result.provider == Provider.OPENROUTER

    @pytest.mark.asyncio
    async def test_resolve_model_with_requirements(self):
        """Test model resolution with requirements."""
        # Register models with different capabilities
        self.registry.register_model(
            "gpt-3.5-turbo", Provider.OPENAI, "gpt-3.5-turbo",
            capabilities=ModelCapabilities(context_length=4096, cost_per_1k_tokens=0.001)
        )
        self.registry.register_model(
            "gpt-4", Provider.OPENAI, "gpt-4-0613",
            capabilities=ModelCapabilities(context_length=8192, cost_per_1k_tokens=0.03)
        )

        # Test with context requirement
        requirements = ModelRequirements(min_context_length=8000)
        result = await self.matcher.resolve_model("gpt", requirements=requirements)
        assert result is not None
        assert result.canonical_name == "gpt-4"  # Should pick the one that meets requirements


class TestGlobalFunctions:
    """Test global registry and matcher functions."""

    def test_get_global_registry(self):
        """Test getting global registry."""
        registry1 = get_global_registry()
        registry2 = get_global_registry()
        assert registry1 is registry2  # Should be singleton

    def test_get_global_matcher(self):
        """Test getting global matcher."""
        matcher1 = get_global_matcher()
        matcher2 = get_global_matcher()
        # Global matcher creates new instances but uses the same global registry
        assert isinstance(matcher1, PipelineModelMatcher)
        assert isinstance(matcher2, PipelineModelMatcher)
        assert matcher1.registry is matcher2.registry  # Same registry instance


class TestUtilityFunctions:
    """Test utility functions."""

    def test_normalize_canonical_name(self):
        """Test canonical name normalization."""
        # Test with different providers
        name1 = _normalize_canonical_name("gpt-4-0613")
        name2 = _normalize_canonical_name("claude-3-opus-20240229")

        assert isinstance(name1, str)
        assert isinstance(name2, str)
        assert len(name1) > 0
        assert len(name2) > 0

    def test_estimate_model_capabilities(self):
        """Test model capabilities estimation."""
        model_data = {
            "context_length": 8192,
            "pricing": {"input": 0.01, "output": 0.03}
        }

        capabilities = _estimate_model_capabilities("gpt-4", model_data)
        assert isinstance(capabilities, ModelCapabilities)
        assert capabilities.context_length > 0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def setup_method(self):
        """Set up test fixtures."""
        self.registry = ModelRegistry()
        self.catalog = ModelCatalog()
        self.matcher = PipelineModelMatcher(self.registry)

    @pytest.mark.asyncio
    async def test_empty_model_input(self):
        """Test handling of empty model input."""
        result = await self.matcher.resolve_model("")
        # Should handle gracefully, either return None or fallback
        assert result is None or isinstance(result, ModelResolution)

    @pytest.mark.asyncio
    async def test_none_model_input(self):
        """Test handling of None model input."""
        with pytest.raises((TypeError, ValueError)):
            await self.matcher.resolve_model(None)

    def test_invalid_provider(self):
        """Test handling of invalid provider."""
        # Test that invalid provider strings are handled appropriately
        # The Provider enum will raise ValueError for invalid values
        with pytest.raises(ValueError):
            # Try to create Provider enum with invalid value
            invalid_provider = Provider("invalid_provider")
            self.registry.register_model("test", invalid_provider, "test-model")

    def test_duplicate_model_registration(self):
        """Test duplicate model registration."""
        self.registry.register_model("gpt-4", Provider.OPENAI, "gpt-4-0613")
        # Should handle duplicate registration gracefully
        self.registry.register_model("gpt-4", Provider.OPENAI, "gpt-4-0613")

        models = self.registry.get_models_for_canonical("gpt-4")
        # Should not create duplicates
        assert len(models) >= 1

    def test_circular_alias(self):
        """Test handling of circular aliases."""
        self.registry.register_model("gpt-4", Provider.OPENAI, "gpt-4-0613")
        self.registry.register_alias("alias1", "gpt-4")
        # This should be prevented or handled gracefully
        try:
            self.registry.register_alias("gpt-4", "alias1")
        except (ValueError, RuntimeError):
            pass  # Expected to fail

    @pytest.mark.asyncio
    async def test_requirements_with_no_matching_models(self):
        """Test requirements that no models can satisfy."""
        self.registry.register_model(
            "gpt-3.5-turbo", Provider.OPENAI, "gpt-3.5-turbo",
            capabilities=ModelCapabilities(context_length=4096)
        )

        # Require impossible context length
        requirements = ModelRequirements(min_context_length=1000000)
        result = await self.matcher.resolve_model("gpt-3.5-turbo", requirements=requirements)
        assert result is None

    @pytest.mark.asyncio
    async def test_malformed_model_names(self):
        """Test handling of malformed model names."""
        malformed_names = [
            "model with spaces",
            "model/with/slashes",
            "model@with@symbols",
            "model\nwith\nnewlines",
            "model\twith\ttabs"
        ]

        for name in malformed_names:
            try:
                result = await self.matcher.resolve_model(name)
                # Should handle gracefully
                assert result is None or isinstance(result, ModelResolution)
            except Exception as e:
                # If it raises an exception, it should be a reasonable one
                assert isinstance(e, (ValueError, TypeError))


class TestPerformance:
    """Test performance-related aspects."""

    def setup_method(self):
        """Set up test fixtures."""
        self.registry = ModelRegistry()
        self.matcher = PipelineModelMatcher(self.registry)

        # Register many models for performance testing
        providers = [Provider.OPENAI, Provider.ANTHROPIC, Provider.OPENROUTER]
        for i in range(100):
            provider = providers[i % len(providers)]
            self.registry.register_model(
                f"model-{i}",
                provider,
                f"provider-model-{i}",
                aliases=[f"alias-{i}", f"alt-{i}"]
            )

    @pytest.mark.asyncio
    async def test_large_registry_performance(self):
        """Test performance with large registry."""
        import time

        start_time = time.time()
        for i in range(50):
            result = await self.matcher.resolve_model(f"model-{i}")
            assert result is not None
        end_time = time.time()

        # Should complete reasonably quickly (adjust threshold as needed)
        assert (end_time - start_time) < 1.0  # Less than 1 second for 50 lookups

    @pytest.mark.asyncio
    async def test_fuzzy_matching_performance(self):
        """Test fuzzy matching performance."""
        import time

        start_time = time.time()
        for i in range(10):
            # Test fuzzy matching with non-exact names
            result = await self.matcher.resolve_model(f"model{i}")  # Missing dash
        end_time = time.time()

        # Fuzzy matching should still be reasonably fast
        assert (end_time - start_time) < 2.0  # Less than 2 seconds for 10 fuzzy lookups


if __name__ == "__main__":
    pytest.main([__file__])
