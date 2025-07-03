import pytest
from dotenv import load_dotenv
from pydantic import BaseModel

import schema_cat
from schema_cat import Provider, RoutingStrategy, ModelResolution, discover_and_register_models, get_global_registry


class Response(BaseModel):
    message: str


@pytest.mark.asyncio
async def test_schema_cat_providers_without_comet():
    get_global_registry().clear()
    await discover_and_register_models()
    await assert_model_resolved("gemini",
                                "google/gemini-2.5-pro",
                                prefered_providers=[Provider.COMET],
                                expected_provider=Provider.OPENROUTER,
                                routing_strategy=RoutingStrategy.MAX_CONTEXT)

    await assert_model_resolved("gemini",
                                "google/gemini-flash-1.5",
                                expected_provider=Provider.OPENROUTER,
                                routing_strategy=RoutingStrategy.CHEAPEST)

    await assert_model_resolved("sonar",
                                "perplexity/sonar",
                                expected_provider=Provider.OPENROUTER)


@pytest.mark.asyncio
async def test_schema_cat_providers():
    load_dotenv()
    get_global_registry().clear()
    await discover_and_register_models()
    await assert_model_resolved("gemini",
                                "gemini-2.0-pro-exp",
                                prefered_providers=[Provider.COMET],
                                expected_provider=Provider.COMET,
                                routing_strategy=RoutingStrategy.MAX_CONTEXT)

    await assert_model_resolved("gemini",
                                'google/gemini-2.5-pro',
                                routing_strategy=RoutingStrategy.MAX_CONTEXT)

    await assert_model_resolved("sonar",
                                "perplexity/sonar",
                                expected_provider=Provider.OPENROUTER)


async def assert_model_resolved(model: str,
                                expected_model: str,
                                prefered_providers: list[Provider] = None,
                                expected_provider: Provider = None,
                                routing_strategy: RoutingStrategy = None):
    model: ModelResolution = await schema_cat.resolve_model(model,
                                                            preferred_providers=prefered_providers,
                                                            routing_strategy=routing_strategy)
    assert model is not None
    if expected_provider is not None:
        assert model.provider == expected_provider
    assert model.model_name == expected_model
