# schema-cat

A Python library for creating typed prompts for Large Language Models (LLMs). Schema-cat allows you to define the structure of LLM responses using Pydantic models, making it easy to get structured, typed data from LLM APIs.

Published by [The Famous Cat](https://www.thefamouscat.com).

## Features

- Define response structures using Pydantic models
- Automatically convert Pydantic models to XML schemas
- Parse LLM responses back into Pydantic models
- Built-in retry mechanism with exponential backoff for handling rate limits and transient errors
- Support for multiple LLM providers:
  - OpenAI
  - Anthropic
  - OpenRouter

## Installation

```bash
pip install schema-cat
```

## Usage

### Basic Usage

```python
from pydantic import BaseModel
from schema_cat import prompt_with_schema, Provider
import asyncio

# Define your response structure
class UserInfo(BaseModel):
    name: str
    age: int
    is_student: bool

# Create a prompt
prompt = "Extract information about John Doe, who is 25 years old and not a student."

# Get a structured response
async def main():
    result = await prompt_with_schema(
        prompt=prompt,
        schema=UserInfo,
        model="gpt-4-turbo",  # Use an appropriate model
        provider=Provider.OPENAI
    )

    print(f"Name: {result.name}")
    print(f"Age: {result.age}")
    print(f"Is student: {result.is_student}")

asyncio.run(main())
```

### Using Different Providers

```python
# OpenAI
result = await prompt_with_schema(prompt, UserInfo, "gpt-4-turbo", Provider.OPENAI)

# Anthropic
result = await prompt_with_schema(prompt, UserInfo, "claude-3-haiku-20240307", Provider.ANTHROPIC)

# OpenRouter
result = await prompt_with_schema(prompt, UserInfo, "anthropic/claude-3-opus-20240229", Provider.OPENROUTER)
```

### Working with Complex Schemas

```python
from pydantic import BaseModel
from typing import List
from schema_cat import prompt_with_schema, Provider

class Address(BaseModel):
    street: str
    city: str
    zip_code: str

class Person(BaseModel):
    name: str
    age: int
    addresses: List[Address]

prompt = """
Extract information about Jane Smith, who is 30 years old.
She has two addresses:
1. 123 Main St, New York, 10001
2. 456 Park Ave, Boston, 02108
"""

async def main():
    result = await prompt_with_schema(prompt, Person, "gpt-4o")
    print(f"Name: {result.name}")
    print(f"Age: {result.age}")
    print(f"Addresses:")
    for addr in result.addresses:
        print(f"  - {addr.street}, {addr.city}, {addr.zip_code}")

asyncio.run(main())
```

## API Reference

### `prompt_with_schema(prompt: str, schema: Type[T], model: str, max_tokens: int = 8192, temperature: float = 0.0, sys_prompt: str = "", max_retries: int = 5, initial_delay: float = 1.0, max_delay: float = 60.0) -> T`

Makes a request to an LLM provider with a prompt and schema, returning a structured response.

- `prompt`: The prompt to send to the LLM
- `schema`: A Pydantic model class defining the expected response structure
- `model`: The LLM model to use (e.g., "gpt-4-turbo", "claude-3-haiku")
- `max_tokens`: Maximum number of tokens to generate (default: 8192)
- `temperature`: Sampling temperature (0.0 to 1.0, default: 0.0)
- `sys_prompt`: Optional system prompt to prepend (default: "")
- `max_retries`: Maximum number of retries for API calls (default: 5)
- `initial_delay`: Initial delay between retries in seconds (default: 1.0)
- `max_delay`: Maximum delay between retries in seconds (default: 60.0)

### `schema_to_xml(schema: Type[BaseModel]) -> ElementTree.XML`

Converts a Pydantic model class to an XML representation.

### `xml_to_base_model(xml_tree: ElementTree.XML, schema: Type[T]) -> T`

Converts an XML element to a Pydantic model instance.

### `xml_to_string(xml_tree: ElementTree.XML) -> str`

Converts an XML element to a pretty-printed string.

## Environment Variables

The library uses the following environment variables:

- `OPENAI_API_KEY`: Required for OpenAI provider
- `OPENAI_BASE_URL`: Optional, defaults to "https://api.openai.com/v1"
- `ANTHROPIC_API_KEY`: Required for Anthropic provider
- `OPENROUTER_API_KEY`: Required for OpenRouter provider
- `OPENROUTER_BASE_URL`: Optional, defaults to "https://openrouter.ai/api/v1"
- `OPENROUTER_HTTP_REFERER`: Optional, defaults to "https://www.thefamouscat.com"
- `OPENROUTER_X_TITLE`: Optional, defaults to "SchemaCat"

## Development

Install dependencies with Poetry:

```bash
poetry install
```

## Retry Mechanism

Schema-cat includes a built-in retry mechanism with exponential backoff to handle rate limits and transient errors when making API calls to LLM providers. This helps improve the reliability of your application by automatically retrying failed requests.

### How It Works

When an API call fails with a retryable error (such as a network error or rate limit), schema-cat will:

1. Wait for a short delay
2. Retry the API call
3. If it fails again, wait for a longer delay (exponential backoff)
4. Continue retrying until the call succeeds or the maximum number of retries is reached

### Configuration

You can configure the retry behavior using the following parameters in `prompt_with_schema`:

```python
result = await prompt_with_schema(
    prompt="Your prompt",
    schema=YourSchema,
    model="gpt-4-turbo",
    max_retries=5,           # Maximum number of retry attempts
    initial_delay=1.0,       # Initial delay between retries in seconds
    max_delay=60.0           # Maximum delay between retries in seconds
)
```

### Retryable Errors

By default, the retry mechanism handles common errors such as:

- Network errors (ConnectionError, TimeoutError)
- HTTP client errors (httpx.ConnectError, httpx.ReadTimeout, etc.)
- Provider-specific errors (openai.RateLimitError, anthropic.RateLimitError, etc.)

### Running Tests

```bash
pytest
```

For end-to-end tests that make actual API calls:

```bash
pytest -m slow
```

## Publishing

To publish to PyPI:

```bash
poetry build
poetry publish
```
