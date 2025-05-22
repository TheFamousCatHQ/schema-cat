import pytest
from typing import Union, Optional, Dict, Literal
from pydantic import BaseModel, Field

from schema_cat import schema_to_xml, xml_to_string, xml_to_base_model
from tests.test_schema_to_xml import xml_to_dict, strip_cdata


class UnionModel(BaseModel):
    """Model with a Union field."""
    value: Union[int, str] = Field(..., description="A field that can be either an integer or a string.")


class OptionalModel(BaseModel):
    """Model with an Optional field."""
    value: Optional[str] = Field(None, description="An optional string field.")


class DictModel(BaseModel):
    """Model with a Dict field."""
    mapping: Dict[str, int] = Field(..., description="A mapping from strings to integers.")


class LiteralModel(BaseModel):
    """Model with a Literal field."""
    status: Literal["pending", "completed", "failed"] = Field(..., description="Status with limited options.")


class ComplexNestedModel(BaseModel):
    """Model with nested complex types."""
    union_field: Union[int, str] = Field(..., description="A field that can be either an integer or a string.")
    optional_field: Optional[int] = Field(None, description="An optional integer field.")
    dict_field: Dict[str, Union[int, str]] = Field(..., description="A mapping from strings to either integers or strings.")
    literal_field: Literal[1, 2, 3] = Field(..., description="A field with literal values 1, 2, or 3.")
    list_of_unions: list[Union[int, str]] = Field(..., description="A list of items that can be either integers or strings.")
    optional_dict: Optional[Dict[str, int]] = Field(None, description="An optional mapping from strings to integers.")


def test_union_model():
    """Test schema_to_xml and xml_to_base_model with Union types."""
    xml = schema_to_xml(UnionModel)
    d = xml_to_dict(xml)

    # Check that the XML was generated correctly
    assert 'UnionModel' in d
    assert 'value' in d['UnionModel']

    # Convert back to a model
    xml_str = xml_to_string(xml)
    xml_elem = xml_to_base_model(xml, UnionModel)

    # Check that the model was parsed correctly
    assert isinstance(xml_elem, UnionModel)
    assert isinstance(xml_elem.value, (int, str))


def test_optional_model():
    """Test schema_to_xml and xml_to_base_model with Optional types."""
    xml = schema_to_xml(OptionalModel)
    d = xml_to_dict(xml)

    # Check that the XML was generated correctly
    assert 'OptionalModel' in d
    assert 'value' in d['OptionalModel']

    # Convert back to a model
    xml_elem = xml_to_base_model(xml, OptionalModel)

    # Check that the model was parsed correctly
    assert isinstance(xml_elem, OptionalModel)
    # The value could be None or a string
    assert xml_elem.value is None or isinstance(xml_elem.value, str)


def test_dict_model():
    """Test schema_to_xml and xml_to_base_model with Dict types."""
    xml = schema_to_xml(DictModel)
    d = xml_to_dict(xml)

    # Check that the XML was generated correctly
    assert 'DictModel' in d
    assert 'mapping' in d['DictModel']

    # Convert back to a model
    xml_elem = xml_to_base_model(xml, DictModel)

    # Check that the model was parsed correctly
    assert isinstance(xml_elem, DictModel)
    assert isinstance(xml_elem.mapping, dict)

    # Check that the keys and values have the correct types
    for key, value in xml_elem.mapping.items():
        assert isinstance(key, str)
        assert isinstance(value, int)


def test_literal_model():
    """Test schema_to_xml and xml_to_base_model with Literal types."""
    xml = schema_to_xml(LiteralModel)
    d = xml_to_dict(xml)

    # Check that the XML was generated correctly
    assert 'LiteralModel' in d
    assert 'status' in d['LiteralModel']

    # Convert back to a model
    xml_elem = xml_to_base_model(xml, LiteralModel)

    # Check that the model was parsed correctly
    assert isinstance(xml_elem, LiteralModel)
    assert xml_elem.status in ["pending", "completed", "failed"]


def test_complex_nested_model():
    """Test schema_to_xml and xml_to_base_model with nested complex types."""
    xml = schema_to_xml(ComplexNestedModel)
    d = xml_to_dict(xml)

    # Check that the XML was generated correctly
    assert 'ComplexNestedModel' in d
    assert 'union_field' in d['ComplexNestedModel']
    assert 'optional_field' in d['ComplexNestedModel']
    assert 'dict_field' in d['ComplexNestedModel']
    assert 'literal_field' in d['ComplexNestedModel']
    assert 'list_of_unions' in d['ComplexNestedModel']
    assert 'optional_dict' in d['ComplexNestedModel']

    # Convert back to a model
    xml_elem = xml_to_base_model(xml, ComplexNestedModel)

    # Check that the model was parsed correctly
    assert isinstance(xml_elem, ComplexNestedModel)
    assert isinstance(xml_elem.union_field, (int, str))
    assert xml_elem.optional_field is None or isinstance(xml_elem.optional_field, int)
    assert isinstance(xml_elem.dict_field, dict)
    assert xml_elem.literal_field in [1, 2, 3]
    assert isinstance(xml_elem.list_of_unions, list)
    assert all(isinstance(item, (int, str)) for item in xml_elem.list_of_unions)
    assert xml_elem.optional_dict is None or isinstance(xml_elem.optional_dict, dict)


def test_custom_xml_for_union():
    """Test parsing custom XML for a Union type."""
    # Create a custom XML for UnionModel with a string value
    xml_str = """
    <UnionModel>
        <value><![CDATA[test string]]></value>
    </UnionModel>
    """
    from xml.etree import ElementTree
    xml = ElementTree.fromstring(xml_str)
    model = xml_to_base_model(xml, UnionModel)
    assert isinstance(model, UnionModel)
    assert model.value == "test string"

    # Create a custom XML for UnionModel with an integer value
    xml_str = """
    <UnionModel>
        <value>42</value>
    </UnionModel>
    """
    xml = ElementTree.fromstring(xml_str)
    model = xml_to_base_model(xml, UnionModel)
    assert isinstance(model, UnionModel)
    assert model.value == 42


def test_custom_xml_for_dict():
    """Test parsing custom XML for a Dict type."""
    # Create a custom XML for DictModel
    xml_str = """
    <DictModel>
        <mapping>
            <item>
                <key><![CDATA[key1]]></key>
                <value>1</value>
            </item>
            <item>
                <key><![CDATA[key2]]></key>
                <value>2</value>
            </item>
        </mapping>
    </DictModel>
    """
    from xml.etree import ElementTree
    xml = ElementTree.fromstring(xml_str)
    model = xml_to_base_model(xml, DictModel)
    assert isinstance(model, DictModel)
    assert model.mapping == {"key1": 1, "key2": 2}


def test_custom_xml_for_literal():
    """Test parsing custom XML for a Literal type."""
    # Create a custom XML for LiteralModel
    xml_str = """
    <LiteralModel>
        <status>completed</status>
    </LiteralModel>
    """
    from xml.etree import ElementTree
    xml = ElementTree.fromstring(xml_str)
    model = xml_to_base_model(xml, LiteralModel)
    assert isinstance(model, LiteralModel)
    assert model.status == "completed"
