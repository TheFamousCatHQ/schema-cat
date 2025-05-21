# schema-cat

A Python library for creating typed prompts for Large Language Models (LLMs). Schema-cat allows you to define the structure of LLM responses using Pydantic models, making it easy to get structured, typed data from LLM APIs.

Published by [The Famous Cat](https://www.thefamouscat.com).

## Features

- Define response structures using Pydantic models
- Automatically convert Pydantic models to XML schemas
- Parse LLM responses back into Pydantic models
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
    result = await prompt_with_schema(prompt, Person, "gpt-4-turbo", Provider.OPENAI)
    print(f"Name: {result.name}")
    print(f"Age: {result.age}")
    print(f"Addresses:")
    for addr in result.addresses:
        print(f"  - {addr.street}, {addr.city}, {addr.zip_code}")

asyncio.run(main())
```

## API Reference

### `prompt_with_schema(prompt: str, schema: Type[T], model: str, provider: Provider) -> T`

Makes a request to an LLM provider with a prompt and schema, returning a structured response.

- `prompt`: The prompt to send to the LLM
- `schema`: A Pydantic model class defining the expected response structure
- `model`: The LLM model to use (e.g., "gpt-4-turbo", "claude-3-haiku")
- `provider`: The LLM provider to use (Provider.OPENAI, Provider.ANTHROPIC, or Provider.OPENROUTER)

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
