import xml.etree.ElementTree as ET

import pytest
from pydantic import BaseModel, Field

from schema_cat import schema_to_xml, xml_to_string, xml_to_base_model, prompt_with_schema, Provider


def strip_cdata(text):
    if text and text.startswith('<![CDATA[') and text.endswith(']]>'):
        return text[9:-3]
    return text


def xml_to_dict(elem):
    d = {elem.tag: {}}
    for child in elem:
        if len(child):
            d[elem.tag][child.tag] = xml_to_dict(child)[child.tag]
        else:
            d[elem.tag][child.tag] = strip_cdata(child.text)
    return d


class SimpleModel(BaseModel):
    foo: str = Field(..., description="A string field for foo.")
    bar: int = Field(..., description="An integer field for bar.")
    baz: bool = Field(True, description="A boolean field for baz, default True.")


class NestedModel(BaseModel):
    name: str = Field(..., description="The name of the nested model.")
    child: SimpleModel = Field(..., description="A SimpleModel instance as a child.")


class ListModel(BaseModel):
    items: list[int] = Field(..., description="A list of integers.")


def test_simple_model():
    xml = schema_to_xml(SimpleModel)
    d = xml_to_dict(xml)
    assert 'SimpleModel' in d
    assert set(d['SimpleModel'].keys()) == {'foo', 'bar', 'baz'}
    assert d['SimpleModel']['foo'] == 'example'
    assert d['SimpleModel']['bar'] == '0'
    assert d['SimpleModel']['baz'] == 'True' or d['SimpleModel']['baz'] == 'False'


def test_nested_model():
    xml = schema_to_xml(NestedModel)
    d = xml_to_dict(xml)
    assert 'NestedModel' in d
    assert 'name' in d['NestedModel']
    assert 'child' in d['NestedModel']
    assert 'foo' in d['NestedModel']['child']
    assert 'bar' in d['NestedModel']['child']
    assert 'baz' in d['NestedModel']['child']


def test_list_model():
    xml = schema_to_xml(ListModel)
    d = xml_to_dict(xml)
    assert 'ListModel' in d
    assert 'items' in d['ListModel']
    # Should be empty list, so items is not present as subelements
    assert d['ListModel']['items'] is None or d['ListModel']['items'] == ''


def test_xml_to_string_simple():
    xml_elem = schema_to_xml(SimpleModel)
    xml_str = xml_to_string(xml_elem)
    assert isinstance(xml_str, str)
    assert xml_str.strip().startswith('<?xml')
    assert '<SimpleModel>' in xml_str
    assert '<foo>' in xml_str and '<bar>' in xml_str and '<baz>' in xml_str
    # Check pretty-print (indented)
    assert '  <foo>' in xml_str or '\n  <foo>' in xml_str


def test_xml_to_string_nested():
    xml_elem = schema_to_xml(NestedModel)
    xml_str = xml_to_string(xml_elem)
    assert isinstance(xml_str, str)
    assert '<NestedModel>' in xml_str
    assert '<child>' in xml_str
    assert '<foo>' in xml_str and '<bar>' in xml_str and '<baz>' in xml_str
    print(xml_str)


def test_xml_to_base_model_simple():
    xml_elem = schema_to_xml(SimpleModel)
    xml_str = xml_to_string(xml_elem)
    parsed_elem = ET.fromstring(xml_str)
    model = xml_to_base_model(parsed_elem, SimpleModel)
    assert isinstance(model, SimpleModel)
    assert model.foo == "example"
    assert model.bar == 0
    assert model.baz in (True, False)


def test_xml_to_base_model_nested():
    xml_elem = schema_to_xml(NestedModel)
    xml_str = xml_to_string(xml_elem)
    parsed_elem = ET.fromstring(xml_str)
    model = xml_to_base_model(parsed_elem, NestedModel)
    assert isinstance(model, NestedModel)
    assert model.name == "example"
    assert isinstance(model.child, SimpleModel)
    assert model.child.foo == "example"
    assert model.child.bar == 0
    assert model.child.baz in (True, False)


def test_xml_to_base_model_list():
    # Create a ListModel with some items
    class ListModelWithData(BaseModel):
        items: list[int] = Field(..., description="A list of integers.")

    xml = ET.Element("ListModelWithData")
    for i in [1, 2, 3]:
        item_elem = ET.Element("items")
        item_elem.text = str(i)
        xml.append(item_elem)
    model = xml_to_base_model(xml, ListModelWithData)
    assert isinstance(model, ListModelWithData)
    assert model.items == [1, 2, 3]


class E2ESimpleModel(BaseModel):
    foo: str = Field(..., description="A string field for foo.")
    bar: int = Field(..., description="An integer field for bar.")


@pytest.mark.asyncio
@pytest.mark.slow
async def test_prompt_with_schema_openrouter_e2e():
    prompt = "Return foo as 'hello' and bar as 42."
    model = "google/gemma-3-4b-it"  # Use a model you have access to
    result = await prompt_with_schema(prompt, E2ESimpleModel, model, Provider.OPENROUTER)
    assert isinstance(result, E2ESimpleModel)
    assert result.foo.lower() == "hello"
    assert result.bar == 42


@pytest.mark.asyncio
@pytest.mark.slow
async def test_prompt_with_schema_openai_e2e():
    prompt = "Return foo as 'world' and bar as 99."
    model = "gpt-4.1-nano-2025-04-14"  # Use a model you have access to
    result = await prompt_with_schema(prompt, E2ESimpleModel, model, Provider.OPENAI)
    assert isinstance(result, E2ESimpleModel)
    assert result.foo.lower() == "world"
    assert result.bar == 99


@pytest.mark.asyncio
@pytest.mark.slow
async def test_prompt_with_schema_anthropic_e2e():
    prompt = "Return foo as 'anthropic' and bar as 123."
    model = "claude-3-5-haiku-20241022"  # Use a model you have access to
    result = await prompt_with_schema(prompt, E2ESimpleModel, model, Provider.ANTHROPIC)
    assert isinstance(result, E2ESimpleModel)
    assert result.foo.lower() == "anthropic"
    assert result.bar == 123
