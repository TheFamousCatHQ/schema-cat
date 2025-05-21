"""schema-cat: A Python library for typed prompts."""
import logging
import os
from enum import Enum
from typing import Type, TypeVar
from xml.etree import ElementTree

import httpx
from pydantic import BaseModel

from schema_cat.xml import xml_from_string

logger = logging.getLogger("schema_cat")

T = TypeVar('T', bound=BaseModel)


class Provider(str, Enum):
    OPENROUTER = "openrouter"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


def hello_world():
    """Entry function that returns 'Hello World'."""
    return "Hello World"


async def call_openrouter(model: str,
                          sys_prompt: str,
                          user_prompt: str,
                          xml_schema: str,
                          max_tokens: int = 8192,
                          temperature: float = 0.0) -> ElementTree.XML:
    api_key = os.getenv("OPENROUTER_API_KEY")
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

    # Prepare the data payload
    data = {
        "model": model,
        "messages": [
            {"role": "system",
             "content": sys_prompt + "\n\nReturn the results in XML format using the following structure:\n\n" + xml_schema},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER", "https://www.thefamouscat.com"),
        "X-Title": os.getenv("OPENROUTER_X_TITLE", "SchemaCat"),
        "Content-Type": "application/json"
    }

    logger.info(f"Calling OpenRouter API directly with model {model}")
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json=data,
            timeout=60
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"].strip()

    logger.info("Successfully received response from OpenRouter")
    logger.debug(f"Raw response content: {content}")

    # Parse the response content as XML
    # Try to extract XML from the response
    root = xml_from_string(content)
    logger.debug("Successfully parsed response as XML")
    return root


def schema_to_xml(schema: Type[BaseModel]) -> ElementTree.XML:
    """Serializes a pydantic type to an example xml representation."""

    # Create an example instance using default values or type-based dummies
    def example_value(field):
        # Use field.is_required() to check if the field is required (no default or default_factory)
        if not field.is_required():
            if field.default_factory is not None:
                return field.default_factory()
            return field.default
        # Handle nested Pydantic models
        if isinstance(field.annotation, type) and issubclass(field.annotation, BaseModel):
            nested_values = {}
            for n, f in field.annotation.model_fields.items():
                if not f.is_required():
                    if f.default_factory is not None:
                        nested_values[n] = f.default_factory()
                    else:
                        nested_values[n] = f.default
                elif isinstance(f.annotation, type) and issubclass(f.annotation, BaseModel):
                    nested_values[n] = example_value(f)
                elif f.annotation in (int, float):
                    nested_values[n] = 0
                elif f.annotation is bool:
                    nested_values[n] = False
                elif f.annotation is str:
                    nested_values[n] = "example"
                elif hasattr(f.annotation, "__origin__") and f.annotation.__origin__ is list:
                    nested_values[n] = []
                else:
                    nested_values[n] = "example"
            return field.annotation(**nested_values)
        if field.annotation in (int, float):
            return 0
        if field.annotation is bool:
            return False
        if field.annotation is str:
            return "example"
        if hasattr(field.annotation, "__origin__") and field.annotation.__origin__ is list:
            return []
        return "example"

    values = {}
    for name, field in schema.model_fields.items():
        values[name] = example_value(field)
    instance = schema(**values)
    data = instance.model_dump()

    def dict_to_xml(tag, d):
        elem = ElementTree.Element(tag)
        for key, val in d.items():
            if isinstance(val, dict):
                elem.append(dict_to_xml(key, val))
            elif isinstance(val, list):
                if not val:
                    # Add an empty element for empty lists
                    elem.append(ElementTree.Element(key))
                else:
                    for item in val:
                        if isinstance(item, dict):
                            elem.append(dict_to_xml(key, item))
                        else:
                            child = ElementTree.Element(key)
                            child.text = str(item)
                            elem.append(child)
            else:
                child = ElementTree.Element(key)
                child.text = str(val)
                elem.append(child)
        return elem

    root = dict_to_xml(schema.__name__, data)
    return root


def xml_to_string(xml_tree: ElementTree.XML) -> str:
    """Converts an ElementTree XML element to a pretty-printed XML string."""
    import xml.dom.minidom
    rough_string = ElementTree.tostring(xml_tree, encoding="utf-8")
    reparsed = xml.dom.minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


def xml_to_base_model(xml_tree: ElementTree.XML, schema: Type[T]) -> T:
    """Converts an ElementTree XML element to a Pydantic BaseModel instance."""

    def parse_element(elem, schema):
        values = {}
        for name, field in schema.model_fields.items():
            child = elem.find(name)
            if child is None:
                values[name] = None
                continue
            # Handle nested models
            if isinstance(field.annotation, type) and issubclass(field.annotation, BaseModel):
                values[name] = parse_element(child, field.annotation)
            # Handle lists
            elif hasattr(field.annotation, "__origin__") and field.annotation.__origin__ is list:
                item_type = field.annotation.__args__[0]
                values[name] = []
                for item_elem in elem.findall(name):
                    if isinstance(item_type, type) and issubclass(item_type, BaseModel):
                        values[name].append(parse_element(item_elem, item_type))
                    else:
                        values[name].append(item_elem.text)
            else:
                # Convert to the correct type
                if field.annotation is int:
                    values[name] = int(child.text)
                elif field.annotation is float:
                    values[name] = float(child.text)
                elif field.annotation is bool:
                    values[name] = child.text.lower() == "true"
                else:
                    values[name] = child.text
        return schema(**values)

    return parse_element(xml_tree, schema)


async def call_openai(model: str,
                      sys_prompt: str,
                      user_prompt: str,
                      xml_schema: str,
                      max_tokens: int = 8192,
                      temperature: float = 0.0) -> ElementTree.XML:
    import openai
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
    messages = [
        {"role": "system",
         "content": sys_prompt + "\n\nReturn the results in XML format using the following structure:\n\n" + xml_schema},
        {"role": "user", "content": user_prompt}
    ]
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    content = response.choices[0].message.content.strip()
    logger.info("Successfully received response from OpenAI")
    logger.debug(f"Raw response content: {content}")
    root = xml_from_string(content)
    logger.debug("Successfully parsed response as XML")
    return root


async def call_anthropic(model: str,
                         sys_prompt: str,
                         user_prompt: str,
                         xml_schema: str,
                         max_tokens: int = 8192,
                         temperature: float = 0.0) -> ElementTree.XML:
    import anthropic
    api_key = os.getenv("ANTHROPIC_API_KEY")
    client = anthropic.AsyncAnthropic(api_key=api_key)
    system_prompt = sys_prompt + "\n\nReturn the results in XML format using the following structure:\n\n" + xml_schema
    response = await client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}]
    )
    content = response.content[0].text.strip() if hasattr(response.content[0], 'text') else response.content[0][
        'text'].strip()
    logger.info("Successfully received response from Anthropic")
    logger.debug(f"Raw response content: {content}")
    root = xml_from_string(content)
    logger.debug("Successfully parsed response as XML")
    return root


async def prompt_with_schema(prompt: str, schema: Type[T], model: str, provider: Provider) -> T:
    xml: str = xml_to_string(schema_to_xml(schema))
    if provider == Provider.OPENROUTER:
        xml_elem = await call_openrouter(model, "", prompt, xml_schema=xml)
        return xml_to_base_model(xml_elem, schema)
    elif provider == Provider.OPENAI:
        xml_elem = await call_openai(model, "", prompt, xml_schema=xml)
        return xml_to_base_model(xml_elem, schema)
    elif provider == Provider.ANTHROPIC:
        xml_elem = await call_anthropic(model, "", prompt, xml_schema=xml)
        return xml_to_base_model(xml_elem, schema)
    else:
        raise Exception(f"Provider {provider} not supported")
