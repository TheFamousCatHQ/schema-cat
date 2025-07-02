import pytest
from pydantic import BaseModel, Field

from schema_cat import schema_to_xml, xml_to_string, prompt_with_schema


def strip_cdata(text):
    if text and text.startswith('<![CDATA[') and text.endswith(']]>'):
        return text[9:-3]
    return text


def xml_to_dict(elem):
    d = {elem.tag: {}}
    children = list(elem)
    if not children:
        d[elem.tag] = strip_cdata(elem.text)
        return d
    child_dict = {}
    for child in children:
        child_data = xml_to_dict(child)[child.tag]
        if child.tag in child_dict:
            if not isinstance(child_dict[child.tag], list):
                child_dict[child.tag] = [child_dict[child.tag]]
            child_dict[child.tag].append(child_data)
        else:
            child_dict[child.tag] = child_data
    d[elem.tag] = child_dict
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
    assert d['SimpleModel']['foo'] == 'A_STRING_FIELD_FOR_FOO'
    assert d['SimpleModel']['bar'] == 'AN_INTEGER_FIELD_FOR_BAR'
    assert d['SimpleModel']['baz'] == 'A_BOOLEAN_FIELD_FOR_BAZ_DEFAULT_TRUE'


def test_nested_model():
    xml = schema_to_xml(NestedModel)
    d = xml_to_dict(xml)
    assert 'NestedModel' in d
    assert 'name' in d['NestedModel']
    assert 'child' in d['NestedModel']
    assert d['NestedModel']['name'] == 'THE_NAME_OF_THE_NESTED_MODEL'
    assert d['NestedModel']['child']['foo'] == 'A_STRING_FIELD_FOR_FOO'
    assert d['NestedModel']['child']['bar'] == 'AN_INTEGER_FIELD_FOR_BAR'
    assert d['NestedModel']['child']['baz'] == 'A_BOOLEAN_FIELD_FOR_BAZ_DEFAULT_TRUE'


def test_list_model():
    xml = schema_to_xml(ListModel)
    d = xml_to_dict(xml)
    assert 'ListModel' in d
    assert 'items' in d['ListModel']
    # Should be a list of two elements with the TO_THIS_STYLE description
    items = d['ListModel']['items']
    # Handle different possible structures
    if isinstance(items, dict):
        # Case 1: Nested structure with same name {'items': [...]}
        if 'items' in items:
            items = items['items']
        # Case 2: Nested structure with singular name {'item': [...]}
        elif 'item' in items:
            items = items['item']
    assert items == ['A_LIST_OF_INTEGERS', 'A_LIST_OF_INTEGERS']


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


class E2ESimpleModel(BaseModel):
    foo: str = Field(..., description="A string field for foo.")
    bar: int = Field(..., description="An integer field for bar.")


@pytest.mark.asyncio
@pytest.mark.slow
async def test_prompt_with_schema_openrouter_e2e():
    prompt = "Return foo as 'hello' and bar as 42."
    model = "google/gemma-3-4b-it"  # Use a model you have access to
    result = await prompt_with_schema(prompt, E2ESimpleModel, model)
    assert isinstance(result, E2ESimpleModel)
    assert isinstance(result.foo, str)
    assert isinstance(result.bar, int)


@pytest.mark.asyncio
@pytest.mark.slow
async def test_prompt_with_schema_openai_e2e():
    prompt = "Return foo as 'world' and bar as 99."
    model = "gpt-4.1-nano-2025-04-14"  # Use a model you have access to
    result = await prompt_with_schema(prompt, E2ESimpleModel, model)
    assert isinstance(result, E2ESimpleModel)
    assert isinstance(result.foo, str)
    assert isinstance(result.bar, int)


@pytest.mark.asyncio
@pytest.mark.slow
async def test_prompt_with_schema_anthropic_e2e():
    prompt = "Return foo as 'anthropic' and bar as 123."
    model = "claude-3-5-haiku-latest"  # Use a model you have access to
    result = await prompt_with_schema(prompt, E2ESimpleModel, model)
    assert isinstance(result, E2ESimpleModel)
    assert isinstance(result.foo, str)
    assert isinstance(result.bar, int)
