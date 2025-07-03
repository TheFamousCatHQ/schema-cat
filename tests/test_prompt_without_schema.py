import pytest

from schema_cat import prompt_without_schema


@pytest.mark.asyncio
@pytest.mark.slow
async def test_prompt_without_schema_openrouter_e2e():
    """Test freeform prompting with OpenRouter provider."""
    prompt = "Write a short greeting message."
    model = "google/gemma-3-4b-it"  # Use a model you have access to
    result = await prompt_without_schema(prompt, model)
    assert isinstance(result, str)
    assert len(result.strip()) > 0
    # Should not contain XML tags since it's freeform
    assert not result.strip().startswith('<?xml')


@pytest.mark.asyncio
@pytest.mark.slow
async def test_prompt_without_schema_openai_e2e():
    """Test freeform prompting with OpenAI provider."""
    prompt = "Say hello in a creative way."
    model = "gpt-4.1-nano-2025-04-14"  # Use a model you have access to
    result = await prompt_without_schema(prompt, model)
    assert isinstance(result, str)
    assert len(result.strip()) > 0
    # Should not contain XML tags since it's freeform
    assert not result.strip().startswith('<?xml')


@pytest.mark.asyncio
@pytest.mark.slow
async def test_prompt_without_schema_anthropic_e2e():
    """Test freeform prompting with Anthropic provider."""
    prompt = "Write a brief explanation of what AI is."
    model = "claude-3-5-haiku-latest"  # Use a model you have access to
    result = await prompt_without_schema(prompt, model)
    assert isinstance(result, str)
    assert len(result.strip()) > 0
    # Should not contain XML tags since it's freeform
    assert not result.strip().startswith('<?xml')


@pytest.mark.asyncio
async def test_prompt_without_schema_with_system_prompt():
    """Test freeform prompting with a system prompt."""
    prompt = "What is 2+2?"
    sys_prompt = "You are a helpful math tutor. Always explain your reasoning."
    model = "gpt-4.1-nano-2025-04-14"
    result = await prompt_without_schema(prompt, model, sys_prompt=sys_prompt)
    assert isinstance(result, str)
    assert len(result.strip()) > 0
    # Should contain some explanation since we asked for reasoning
    assert len(result.split()) > 5  # More than just "4"


@pytest.mark.asyncio
async def test_prompt_without_schema_with_temperature():
    """Test freeform prompting with different temperature settings."""
    prompt = "Write a single word."
    model = "gpt-4.1-nano-2025-04-14"
    
    # Test with low temperature (more deterministic)
    result_low = await prompt_without_schema(prompt, model, temperature=0.0)
    assert isinstance(result_low, str)
    assert len(result_low.strip()) > 0
    
    # Test with high temperature (more creative)
    result_high = await prompt_without_schema(prompt, model, temperature=0.9)
    assert isinstance(result_high, str)
    assert len(result_high.strip()) > 0


@pytest.mark.asyncio
async def test_prompt_without_schema_max_tokens():
    """Test freeform prompting with max_tokens limit."""
    prompt = "Write a very long story about a cat."
    model = "gpt-4.1-nano-2025-04-14"
    
    # Test with very low max_tokens
    result = await prompt_without_schema(prompt, model, max_tokens=10)
    assert isinstance(result, str)
    assert len(result.strip()) > 0
    # Should be relatively short due to token limit
    assert len(result.split()) <= 15  # Allow some buffer for tokenization differences