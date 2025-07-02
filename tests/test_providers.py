from unittest.mock import AsyncMock, MagicMock, patch
from xml.etree import ElementTree

import pytest

from schema_cat.providers import OpenAIProvider, OpenRouterProvider, AnthropicProvider
from schema_cat.xml import XMLParsingError


class TestOpenAIProvider:
    """Test cases for OpenAIProvider class."""

    @pytest.mark.asyncio
    async def test_openai_provider_success(self):
        """Test successful OpenAI provider call."""
        provider = OpenAIProvider()

        # Mock the OpenAI client
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="<test>success</test>"))]
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        # Mock xml_from_string to return a valid XML element
        mock_xml = ElementTree.fromstring("<test>success</test>")

        with patch("openai.AsyncOpenAI", return_value=mock_client), \
             patch("schema_cat.xml.xml_from_string", return_value=mock_xml), \
             patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):

            result = await provider.call(
                model="gpt-4",
                sys_prompt="System prompt",
                user_prompt="User prompt",
                xml_schema="<schema></schema>"
            )

            assert ElementTree.tostring(result) == ElementTree.tostring(mock_xml)
            mock_client.chat.completions.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_openai_provider_with_custom_params(self):
        """Test OpenAI provider with custom parameters."""
        provider = OpenAIProvider()

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="<test>custom</test>"))]
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        mock_xml = ElementTree.fromstring("<test>custom</test>")

        with patch("openai.AsyncOpenAI", return_value=mock_client), \
             patch("schema_cat.xml.xml_from_string", return_value=mock_xml), \
             patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):

            result = await provider.call(
                model="gpt-3.5-turbo",
                sys_prompt="Custom system",
                user_prompt="Custom user",
                xml_schema="<schema></schema>",
                max_tokens=4096,
                temperature=0.5
            )

            # Verify the call was made with correct parameters
            call_args = mock_client.chat.completions.create.call_args
            assert call_args.kwargs["model"] == "gpt-3.5-turbo"
            assert call_args.kwargs["max_tokens"] == 4096
            assert call_args.kwargs["temperature"] == 0.5


class TestOpenRouterProvider:
    """Test cases for OpenRouterProvider class."""

    @pytest.mark.asyncio
    async def test_openrouter_provider_success(self):
        """Test successful OpenRouter provider call."""
        provider = OpenRouterProvider()

        # Mock httpx client response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "<test>openrouter</test>"}}]
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        mock_xml = ElementTree.fromstring("<test>openrouter</test>")

        with patch("httpx.AsyncClient", return_value=mock_client), \
             patch("schema_cat.xml.xml_from_string", return_value=mock_xml), \
             patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):

            result = await provider.call(
                model="openai/gpt-4",
                sys_prompt="System prompt",
                user_prompt="User prompt",
                xml_schema="<schema></schema>"
            )

            assert ElementTree.tostring(result) == ElementTree.tostring(mock_xml)
            mock_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_openrouter_provider_xml_parsing_error_retry(self):
        """Test OpenRouter provider handles XML parsing errors with retry."""
        provider = OpenRouterProvider()

        # Mock httpx client response with invalid XML first, then valid XML
        mock_response1 = MagicMock()
        mock_response1.json.return_value = {
            "choices": [{"message": {"content": "invalid xml content"}}]
        }
        mock_response1.raise_for_status = MagicMock()

        mock_response2 = MagicMock()
        mock_response2.json.return_value = {
            "choices": [{"message": {"content": "<test>retry_success</test>"}}]
        }
        mock_response2.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.post = AsyncMock(side_effect=[mock_response1, mock_response2])
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        mock_xml = ElementTree.fromstring("<test>retry_success</test>")

        with patch("httpx.AsyncClient", return_value=mock_client), \
             patch("schema_cat.xml.xml_from_string", side_effect=[XMLParsingError("Invalid XML"), mock_xml]), \
             patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):

            result = await provider.call(
                model="openai/gpt-4",
                sys_prompt="System prompt",
                user_prompt="User prompt",
                xml_schema="<schema></schema>",
                max_retries=1
            )

            assert ElementTree.tostring(result) == ElementTree.tostring(mock_xml)
            assert mock_client.post.call_count == 2


class TestAnthropicProvider:
    """Test cases for AnthropicProvider class."""

    @pytest.mark.asyncio
    async def test_anthropic_provider_success(self):
        """Test successful Anthropic provider call."""
        provider = AnthropicProvider()

        # Mock the Anthropic client
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_content = MagicMock()
        mock_content.text = "<test>anthropic</test>"
        mock_response.content = [mock_content]
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        mock_xml = ElementTree.fromstring("<test>anthropic</test>")

        with patch("anthropic.AsyncAnthropic", return_value=mock_client), \
             patch("schema_cat.xml.xml_from_string", return_value=mock_xml), \
             patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):

            result = await provider.call(
                model="claude-3-sonnet-20240229",
                sys_prompt="System prompt",
                user_prompt="User prompt",
                xml_schema="<schema></schema>"
            )

            assert ElementTree.tostring(result) == ElementTree.tostring(mock_xml)
            mock_client.messages.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_anthropic_provider_with_dict_content(self):
        """Test Anthropic provider with dictionary-style content."""
        provider = AnthropicProvider()

        # Mock the Anthropic client with dict-style content
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_content = {"text": "<test>dict_content</test>"}
        mock_response.content = [mock_content]
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        mock_xml = ElementTree.fromstring("<test>dict_content</test>")

        with patch("anthropic.AsyncAnthropic", return_value=mock_client), \
             patch("schema_cat.xml.xml_from_string", return_value=mock_xml), \
             patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):

            result = await provider.call(
                model="claude-3-haiku-20240307",
                sys_prompt="System prompt",
                user_prompt="User prompt",
                xml_schema="<schema></schema>",
                max_tokens=4096,
                temperature=0.3
            )

            assert ElementTree.tostring(result) == ElementTree.tostring(mock_xml)
            # Verify the call was made with correct parameters
            call_args = mock_client.messages.create.call_args
            assert call_args.kwargs["model"] == "claude-3-haiku-20240307"
            assert call_args.kwargs["max_tokens"] == 4096
            assert call_args.kwargs["temperature"] == 0.3


class TestProviderIntegration:
    """Integration tests for provider classes."""

    @pytest.mark.asyncio
    async def test_all_providers_implement_base_interface(self):
        """Test that all provider classes implement the base interface correctly."""
        providers = [OpenAIProvider(), OpenRouterProvider(), AnthropicProvider()]

        for provider in providers:
            # Check that the call method exists and has the correct signature
            assert hasattr(provider, 'call')
            assert callable(provider.call)

            # Check that it's an async method
            import inspect
            assert inspect.iscoroutinefunction(provider.call)
