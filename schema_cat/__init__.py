"""schema-cat: A Python library for typed prompts."""
import logging
from typing import Type
from xml.etree import ElementTree

from pydantic import BaseModel

logger = logging.getLogger("schema_cat")


def hello_world():
    """Entry function that returns 'Hello World'."""
    return "Hello World"


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


def prompt_with_schema(prompt: str, schema: Type[BaseModel]) -> BaseModel:
    xml: str = schema_to_xml(schema)
    pass
