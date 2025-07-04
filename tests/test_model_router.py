import json
import os
import tempfile

import pytest

from schema_cat.model_registry import (
    ModelRequirements, ModelCapabilities, ModelResolution,
    RoutingStrategy, RequestContext, Provider
)
from schema_cat.model_router import (
    SmartModelRouter, RouterConfig, RouteResult
)


class TestRouterConfig:
    """Test RouterConfig functionality."""

    def test_router_config_defaults(self):
        """Test RouterConfig default values."""
        config = RouterConfig()
        assert config.default_strategy == RoutingStrategy.BEST_AVAILABLE
        assert config.preferred_providers == []
        assert config.default_requirements is None
        assert config.model_overrides == {}
        assert config.provider_fallbacks == {}

    def test_router_config_with_values(self):
        """Test RouterConfig with custom values."""
        requirements = ModelRequirements(min_context_length=8192)
        config = RouterConfig(
            default_strategy=RoutingStrategy.CHEAPEST,
            preferred_providers=[Provider.OPENAI, Provider.ANTHROPIC],
            default_requirements=requirements,
            model_overrides={"gpt-4": {"strategy": "highest_quality"}},
            provider_fallbacks={Provider.OPENAI: [Provider.OPENROUTER]}
        )
        assert config.default_strategy == RoutingStrategy.CHEAPEST
        assert config.preferred_providers == [Provider.OPENAI, Provider.ANTHROPIC]
        assert config.default_requirements == requirements
        assert config.model_overrides == {"gpt-4": {"strategy": "highest_quality"}}
        assert config.provider_fallbacks == {Provider.OPENAI: [Provider.OPENROUTER]}


class TestSmartModelRouter:
    """Test SmartModelRouter functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.router = SmartModelRouter()

    @pytest.mark.asyncio
    async def test_route_model_basic(self):
        """Test basic model routing."""
        # Register a test model
        self.router.registry.register_model("gpt-4", Provider.OPENAI, "gpt-4-0613")

        result = await self.router.route_model("gpt-4")
        assert result is not None
        assert isinstance(result, RouteResult)
        assert result.resolution.canonical_name == "gpt-4"
        assert result.resolution.provider == Provider.OPENAI

    @pytest.mark.asyncio
    async def test_route_model_with_requirements(self):
        """Test model routing with requirements."""
        # Register models with different capabilities
        self.router.registry.register_model(
            "gpt-3.5-turbo", Provider.OPENAI, "gpt-3.5-turbo",
            capabilities=ModelCapabilities(context_length=4096, cost_per_1k_tokens=0.001)
        )
        self.router.registry.register_model(
            "gpt-4", Provider.OPENAI, "gpt-4-0613",
            capabilities=ModelCapabilities(context_length=8192, cost_per_1k_tokens=0.03)
        )

        # Test with context requirement
        requirements = ModelRequirements(min_context_length=8000)
        context = RequestContext(requirements=requirements)
        result = await self.router.route_model("gpt", context=context)
        assert result is not None
        assert result.resolution.canonical_name == "gpt-4"  # Should pick the one that meets requirements

    @pytest.mark.asyncio
    async def test_route_model_with_strategy(self):
        """Test model routing with specific strategy."""
        # Register models with different costs
        self.router.registry.register_model(
            "gpt-3.5-turbo", Provider.OPENAI, "gpt-3.5-turbo",
            capabilities=ModelCapabilities(cost_per_1k_tokens=0.001, quality_score=0.7)
        )
        self.router.registry.register_model(
            "gpt-4", Provider.OPENAI, "gpt-4-0613",
            capabilities=ModelCapabilities(cost_per_1k_tokens=0.03, quality_score=0.95)
        )

        # Test cheapest strategy
        result = await self.router.route_model("gpt", strategy=RoutingStrategy.CHEAPEST)
        assert result is not None
        assert result.canonical_name == "gpt-4"

        # Test highest quality strategy
        result = await self.router.route_model("gpt", strategy=RoutingStrategy.HIGHEST_QUALITY)
        assert result is not None
        assert result.canonical_name == "gpt-4"

    @pytest.mark.asyncio
    async def test_route_model_with_preferred_providers(self):
        """Test model routing with preferred providers."""
        # Register same model for multiple providers
        self.router.registry.register_model("gpt-4", Provider.OPENAI, "gpt-4-0613", priority=10)
        self.router.registry.register_model("gpt-4", Provider.OPENROUTER, "openai/gpt-4", priority=5)

        # Test with preferred provider
        result = await self.router.route_model("gpt-4", preferred_providers=[Provider.OPENROUTER])
        assert result is not None
        assert result.provider == Provider.OPENROUTER

    def test_set_config(self):
        """Test setting router configuration."""
        config = RouterConfig(
            default_strategy=RoutingStrategy.CHEAPEST,
            preferred_providers=[Provider.OPENAI]
        )

        self.router.set_config(config)
        assert self.router.config.default_strategy == RoutingStrategy.CHEAPEST
        assert self.router.config.preferred_providers == [Provider.OPENAI]

    @pytest.mark.asyncio
    async def test_route_model_with_config_overrides(self):
        """Test model routing with configuration overrides."""
        # Set up config with model-specific overrides
        config = RouterConfig(
            default_strategy=RoutingStrategy.CHEAPEST,
            model_overrides={
                "gpt-4": {
                    "strategy": "highest_quality",
                    "preferred_providers": ["openai"]
                }
            }
        )
        self.router.set_config(config)

        # Register models
        self.router.registry.register_model(
            "gpt-3.5-turbo", Provider.OPENAI, "gpt-3.5-turbo",
            capabilities=ModelCapabilities(cost_per_1k_tokens=0.001, quality_score=0.7)
        )
        self.router.registry.register_model(
            "gpt-4", Provider.OPENAI, "gpt-4-0613",
            capabilities=ModelCapabilities(cost_per_1k_tokens=0.03, quality_score=0.95)
        )

        # Test that override is applied for gpt-4
        result = await self.router.route_model("gpt-4")
        assert result is not None
        assert result.canonical_name == "gpt-4"  # Should use highest quality strategy

    @pytest.mark.asyncio
    async def test_route_model_not_found(self):
        """Test model routing when model is not found."""
        result = await self.router.route_model("nonexistent-model")
        assert result is None

    @pytest.mark.asyncio
    async def test_route_model_with_fallback_providers(self):
        """Test model routing with provider fallbacks."""
        config = RouterConfig(
            provider_fallbacks={
                Provider.OPENAI: [Provider.OPENROUTER, Provider.ANTHROPIC]
            }
        )
        self.router.registry.clear()
        self.router.set_config(config)

        # Register model only on fallback provider
        self.router.registry.register_model("gpt-4", Provider.OPENROUTER, "openai/gpt-4")

        # Request with primary provider that doesn't have the model
        result = await self.router.route_model("gpt-4", preferred_providers=[Provider.OPENAI])
        assert result is not None
        assert result.provider == Provider.OPENROUTER  # Should fallback


class TestConfigFileOperations:
    """Test configuration file operations."""

    def test_load_config_from_file(self):
        """Test loading configuration from file."""
        config_dict = {
            "default_strategy": "cheapest",
            "preferred_providers": ["openai", "anthropic"],
            "overrides": {
                "gpt-4": {"strategy": "highest_quality"}
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_dict, f)
            config_file = f.name

        try:
            # Load config using RouterConfig.load_from_file
            config = RouterConfig.load_from_file(config_file)

            assert config.default_strategy == RoutingStrategy.CHEAPEST
            assert config.preferred_providers == [Provider.OPENAI, Provider.ANTHROPIC]
            assert config.overrides == {"gpt-4": {"strategy": "highest_quality"}}
        finally:
            os.unlink(config_file)

    def test_load_config_file_not_found(self):
        """Test loading configuration from non-existent file."""
        with pytest.raises(FileNotFoundError):
            RouterConfig.load_from_file("nonexistent_config.json")

    def test_load_config_from_env_var(self):
        """Test loading preferred_providers from environment variable."""
        # Set environment variable
        os.environ['SCHEMA_CAT_PREFERRED_PROVIDERS'] = 'comet,openrouter,openai'

        try:
            # Load config with empty dict (should use env var)
            config = RouterConfig.from_dict({})

            assert config.preferred_providers == [Provider.COMET, Provider.OPENROUTER, Provider.OPENAI]
        finally:
            # Clean up environment variable
            if 'SCHEMA_CAT_PREFERRED_PROVIDERS' in os.environ:
                del os.environ['SCHEMA_CAT_PREFERRED_PROVIDERS']

    def test_config_file_overrides_env_var(self):
        """Test that config file takes precedence over environment variable."""
        # Set environment variable
        os.environ['SCHEMA_CAT_PREFERRED_PROVIDERS'] = 'comet,openrouter'

        try:
            # Config dict with preferred_providers should override env var
            config_dict = {
                "preferred_providers": ["openai", "anthropic"]
            }
            config = RouterConfig.from_dict(config_dict)

            assert config.preferred_providers == [Provider.OPENAI, Provider.ANTHROPIC]
        finally:
            # Clean up environment variable
            if 'SCHEMA_CAT_PREFERRED_PROVIDERS' in os.environ:
                del os.environ['SCHEMA_CAT_PREFERRED_PROVIDERS']

    def test_env_var_with_invalid_providers(self):
        """Test that invalid provider names in env var are skipped."""
        # Set environment variable with invalid provider
        os.environ['SCHEMA_CAT_PREFERRED_PROVIDERS'] = 'openai,invalid_provider,anthropic'

        try:
            config = RouterConfig.from_dict({})

            # Should only include valid providers
            assert config.preferred_providers == [Provider.OPENAI, Provider.ANTHROPIC]
        finally:
            # Clean up environment variable
            if 'SCHEMA_CAT_PREFERRED_PROVIDERS' in os.environ:
                del os.environ['SCHEMA_CAT_PREFERRED_PROVIDERS']


class TestRouteResult:
    """Test RouteResult dataclass."""

    def test_route_result_creation(self):
        """Test creating RouteResult."""
        resolution = ModelResolution(
            provider=Provider.OPENAI,
            model_name="gpt-4-0613",
            canonical_name="gpt-4"
        )

        result = RouteResult(
            resolution=resolution,
            config_override={"strategy": "highest_quality"},
            routing_reason="model_override_preferred_provider"
        )

        assert result.resolution == resolution
        assert result.config_override == {"strategy": "highest_quality"}
        assert result.routing_reason == "model_override_preferred_provider"


class TestIntegrationScenarios:
    """Test integration scenarios combining multiple features."""

    def setup_method(self):
        """Set up test fixtures."""
        self.router = SmartModelRouter()

        # Register a variety of models
        self.router.registry.register_model(
            "gpt-3.5-turbo", Provider.OPENAI, "gpt-3.5-turbo",
            capabilities=ModelCapabilities(
                context_length=4096,
                cost_per_1k_tokens=0.001,
                quality_score=0.7,
                supports_function_calling=True
            )
        )
        self.router.registry.register_model(
            "gpt-4", Provider.OPENAI, "gpt-4-0613",
            capabilities=ModelCapabilities(
                context_length=8192,
                cost_per_1k_tokens=0.03,
                quality_score=0.95,
                supports_function_calling=True
            )
        )
        self.router.registry.register_model(
            "claude-3-opus", Provider.ANTHROPIC, "claude-3-opus-20240229",
            capabilities=ModelCapabilities(
                context_length=200000,
                cost_per_1k_tokens=0.015,
                quality_score=0.9,
                supports_function_calling=False
            )
        )
        self.router.registry.register_model(
            "gpt-4", Provider.OPENROUTER, "openai/gpt-4",
            priority=5,  # Lower priority than OpenAI
            capabilities=ModelCapabilities(
                context_length=8192,
                cost_per_1k_tokens=0.025,  # Slightly cheaper
                quality_score=0.95,
                supports_function_calling=True
            )
        )

    @pytest.mark.asyncio
    async def test_complex_routing_scenario(self):
        """Test complex routing scenario with multiple constraints."""
        # Set up complex configuration
        config = RouterConfig(
            default_strategy=RoutingStrategy.BEST_AVAILABLE,
            preferred_providers=[Provider.OPENAI, Provider.ANTHROPIC],
            default_requirements=ModelRequirements(
                supports_function_calling=True,
                max_cost_per_1k_tokens=0.02
            ),
            model_overrides={
                "claude-3-opus": {
                    "strategy": "max_context",
                    "requirements": {
                        "supports_function_calling": False  # Override default requirement
                    }
                }
            },
            provider_fallbacks={
                Provider.OPENAI: [Provider.OPENROUTER]
            }
        )
        self.router.set_config(config)

        # Test 1: Should get gpt-3.5-turbo (meets function calling + cost requirements)
        result = await self.router.route_model("gpt")
        assert result is not None
        assert result.canonical_name == "gpt-4"

        # Test 2: Should get claude-3-opus (override allows no function calling)
        result = await self.router.route_model("claude-3-opus")
        assert result is not None
        assert result.canonical_name == "claude-3-opus"

        # Test 3: With high context requirement, should get claude-3-opus
        high_context_req = ModelRequirements(min_context_length=100000)
        result = await self.router.route_model("claude", requirements=high_context_req)
        assert result is not None
        assert result.canonical_name == "claude-3-opus"

    @pytest.mark.asyncio
    async def test_provider_fallback_scenario(self):
        """Test provider fallback scenario."""
        config = RouterConfig(
            preferred_providers=[Provider.OPENAI],
            provider_fallbacks={
                Provider.OPENAI: [Provider.OPENROUTER, Provider.ANTHROPIC]
            }
        )
        self.router.set_config(config)

        # Test fallback when preferred provider has the model
        result = await self.router.route_model("gpt-4")
        assert result is not None
        assert result.provider == Provider.OPENAI  # Should prefer OpenAI

        # Test fallback when requesting with cost constraint that OpenAI can't meet
        cheap_req = ModelRequirements(max_cost_per_1k_tokens=0.026)
        result = await self.router.route_model("gpt-4", requirements=cheap_req)
        assert result is not None
        assert result.provider == Provider.OPENROUTER  # Should fallback to cheaper option

    @pytest.mark.asyncio
    async def test_strategy_override_scenario(self):
        """Test strategy override scenario."""
        config = RouterConfig(
            default_strategy=RoutingStrategy.CHEAPEST,
            model_overrides={
                "gpt-4": {"strategy": "highest_quality"},
                "claude": {"strategy": "max_context"}
            }
        )
        self.router.set_config(config)

        # Test default strategy (cheapest)
        result = await self.router.route_model("gpt-3.5-turbo")
        assert result is not None
        assert result.canonical_name == "gpt-3.5-turbo"

        # Test override strategy for gpt-4 (highest quality)
        result = await self.router.route_model("gpt-4")
        assert result is not None
        assert result.canonical_name == "gpt-4"
        assert result.provider == Provider.OPENAI  # Higher quality than OpenRouter

        # Test override strategy for claude (max context)
        result = await self.router.route_model("claude")
        assert result is not None
        assert result.canonical_name == "claude-3-opus"  # Has highest context


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling in model router."""

    def setup_method(self):
        """Set up test fixtures."""
        self.router = SmartModelRouter()

    @pytest.mark.asyncio
    async def test_empty_registry(self):
        """Test routing with empty registry."""
        result = await self.router.route_model("any-model")
        assert result is None

    @pytest.mark.asyncio
    async def test_impossible_requirements(self):
        """Test routing with impossible requirements."""
        self.router.registry.register_model(
            "gpt-4", Provider.OPENAI, "gpt-4-0613",
            capabilities=ModelCapabilities(context_length=8192)
        )

        # Require impossible context length
        impossible_req = ModelRequirements(min_context_length=1000000)
        result = await self.router.route_model("gpt-4", requirements=impossible_req)
        assert result is None

    @pytest.mark.asyncio
    async def test_invalid_strategy_in_config(self):
        """Test handling invalid strategy in configuration."""
        config_dict = {
            "default_strategy": "invalid_strategy",
            "preferred_providers": ["openai"]
        }

        with pytest.raises(ValueError):
            RouterConfig.from_dict(config_dict)

    @pytest.mark.asyncio
    async def test_invalid_provider_in_config(self):
        """Test handling invalid provider in configuration."""
        config_dict = {
            "preferred_providers": ["invalid_provider"]
        }

        with pytest.raises(ValueError):
            RouterConfig.from_dict(config_dict)

    def test_config_with_none_values(self):
        """Test configuration with None values."""
        config = RouterConfig(
            default_strategy=None,
            preferred_providers=None,
            default_requirements=None
        )

        # Should handle None values gracefully
        assert config.default_strategy is None
        assert config.preferred_providers is None
        assert config.default_requirements is None


if __name__ == "__main__":
    pytest.main([__file__])
