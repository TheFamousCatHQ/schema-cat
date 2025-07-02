from unittest.mock import AsyncMock, patch, MagicMock

import httpx
import pytest

from schema_cat.retry import retry_with_exponential_backoff, with_retry


# Test the retry_with_exponential_backoff function
@pytest.mark.asyncio
async def test_retry_success_first_attempt():
    """Test that a function that succeeds on the first attempt is not retried."""
    mock_func = AsyncMock(return_value="success")

    result = await retry_with_exponential_backoff(mock_func, "arg1", kwarg1="kwarg1")

    assert result == "success"
    mock_func.assert_called_once_with("arg1", kwarg1="kwarg1")


@pytest.mark.asyncio
async def test_retry_success_after_retries():
    """Test that a function that fails initially but succeeds later is retried correctly."""
    # Mock function that fails twice then succeeds
    mock_func = AsyncMock(side_effect=[
        ConnectionError("Connection failed"),
        ConnectionError("Connection failed again"),
        "success"
    ])

    # Use a short delay for testing
    result = await retry_with_exponential_backoff(
        mock_func, "arg1", 
        initial_delay=0.01, 
        max_delay=0.05,
        backoff_factor=1.5
    )

    assert result == "success"
    assert mock_func.call_count == 3
    mock_func.assert_called_with("arg1")


@pytest.mark.asyncio
async def test_retry_max_retries_exceeded():
    """Test that a function that always fails eventually gives up after max_retries."""
    mock_func = AsyncMock(side_effect=ConnectionError("Connection failed"))

    with pytest.raises(ConnectionError, match="Connection failed"):
        await retry_with_exponential_backoff(
            mock_func, "arg1", 
            max_retries=3,
            initial_delay=0.01,
            max_delay=0.05
        )

    assert mock_func.call_count == 4  # Initial attempt + 3 retries


@pytest.mark.asyncio
async def test_retry_non_retryable_exception():
    """Test that a function that raises a non-retryable exception is not retried."""
    mock_func = AsyncMock(side_effect=ValueError("Invalid value"))

    with pytest.raises(ValueError, match="Invalid value"):
        await retry_with_exponential_backoff(
            mock_func, "arg1",
            retry_exceptions=["ConnectionError", "TimeoutError"],
            initial_delay=0.01
        )

    mock_func.assert_called_once_with("arg1")


@pytest.mark.asyncio
async def test_retry_custom_exceptions():
    """Test that custom exception types can be specified for retry."""
    class CustomError(Exception):
        pass

    mock_func = AsyncMock(side_effect=[
        CustomError("Custom error"),
        "success"
    ])

    result = await retry_with_exponential_backoff(
        mock_func, "arg1",
        retry_exceptions=[CustomError],
        initial_delay=0.01
    )

    assert result == "success"
    assert mock_func.call_count == 2


# Test the with_retry decorator
@pytest.mark.asyncio
async def test_with_retry_decorator():
    """Test that the with_retry decorator correctly applies retry logic."""
    mock_impl = AsyncMock(side_effect=[
        ConnectionError("Connection failed"),
        "success"
    ])

    @with_retry(initial_delay=0.01, max_delay=0.05)
    async def func_with_retry(arg):
        return await mock_impl(arg)

    result = await func_with_retry("test_arg")

    assert result == "success"
    assert mock_impl.call_count == 2
    mock_impl.assert_called_with("test_arg")


# Test with specific provider-like errors
@pytest.mark.asyncio
async def test_retry_with_openai_error():
    """Test retry with OpenAI-specific errors."""
    # Create a mock OpenAI error
    class RateLimitError(Exception):
        pass

    # Mock the function that raises the error
    mock_func = AsyncMock(side_effect=[
        RateLimitError("Rate limit exceeded"),
        "success"
    ])

    # Use the actual exception type for retry
    result = await retry_with_exponential_backoff(
        mock_func, 
        retry_exceptions=[RateLimitError],
        initial_delay=0.01
    )

    assert result == "success"
    assert mock_func.call_count == 2


@pytest.mark.asyncio
async def test_retry_with_httpx_error():
    """Test retry with httpx errors."""
    mock_func = AsyncMock(side_effect=[
        httpx.ConnectError("Connection error"),
        "success"
    ])

    result = await retry_with_exponential_backoff(
        mock_func,
        initial_delay=0.01
    )

    assert result == "success"
    assert mock_func.call_count == 2


# Test integration with provider functions (mocked)
@pytest.mark.asyncio
async def test_openai_provider_retry():
    """Test that the OpenAI provider retries correctly."""
    from schema_cat.provider_enum import Provider
    from xml.etree import ElementTree

    # Mock the OpenAI client
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(side_effect=[
        ConnectionError("Connection failed"),
        MagicMock(choices=[MagicMock(message=MagicMock(content="<Test>content</Test>"))])
    ])

    # Mock the xml_from_string function
    mock_xml = ElementTree.fromstring("<Test>content</Test>")

    with patch("openai.AsyncOpenAI", return_value=mock_client), \
         patch("schema_cat.xml.xml_from_string", return_value=mock_xml):

        result = await Provider.OPENAI.call(
            "gpt-4",
            "System prompt",
            "User prompt",
            "<schema>",
            max_retries=3,
            initial_delay=0.01,
            max_delay=0.05
        )

        # Compare XML content rather than ElementTree objects
        assert ElementTree.tostring(result) == ElementTree.tostring(mock_xml)
        assert mock_client.chat.completions.create.call_count == 2
