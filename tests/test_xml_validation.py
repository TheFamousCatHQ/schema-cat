from typing import List, Optional

import pytest
from pydantic import BaseModel

from schema_cat.schema import xml_to_base_model
from schema_cat.xml import xml_from_string


class SimpleModel(BaseModel):
    """A simple model for testing XML validation."""
    name: str
    age: int
    is_active: bool


class NestedModel(BaseModel):
    """A model with nested fields for testing XML validation."""
    title: str
    details: SimpleModel
    tags: List[str]
    optional_field: Optional[str] = None


# Test cases for xml_from_string function
def test_valid_xml():
    """Test parsing a valid XML string."""
    xml_str = "<SimpleModel><name>John</name><age>30</age><is_active>true</is_active></SimpleModel>"
    xml_elem = xml_from_string(xml_str)
    assert xml_elem.tag == "SimpleModel"
    assert xml_elem.find("name").text == "John"
    assert xml_elem.find("age").text == "30"
    assert xml_elem.find("is_active").text == "true"


def test_xml_with_extra_text():
    """Test parsing XML with extra text before and after."""
    xml_str = "Some text before <SimpleModel><name>John</name><age>30</age><is_active>true</is_active></SimpleModel> Some text after"
    xml_elem = xml_from_string(xml_str)
    assert xml_elem.tag == "SimpleModel"
    assert xml_elem.find("name").text == "John"


# Test cases for xml_to_base_model function
def test_valid_xml_to_model():
    """Test converting valid XML to a model."""
    xml_str = "<SimpleModel><name>John</name><age>30</age><is_active>true</is_active></SimpleModel>"
    xml_elem = xml_from_string(xml_str)
    model = xml_to_base_model(xml_elem, SimpleModel)
    assert model.name == "John"
    assert model.age == 30
    assert model.is_active is True


def test_xml_missing_required_field():
    """Test converting XML with a missing required field."""
    xml_str = "<SimpleModel><name>John</name><is_active>true</is_active></SimpleModel>"
    xml_elem = xml_from_string(xml_str)
    with pytest.raises(Exception) as excinfo:
        xml_to_base_model(xml_elem, SimpleModel)
    error_msg = str(excinfo.value).lower()
    assert "age" in error_msg
    # Just check that we get some kind of validation error
    assert "validation" in error_msg or "input" in error_msg


def test_xml_invalid_field_type():
    """Test converting XML with an invalid field type."""
    xml_str = "<SimpleModel><name>John</name><age>thirty</age><is_active>true</is_active></SimpleModel>"
    xml_elem = xml_from_string(xml_str)
    with pytest.raises(Exception) as excinfo:
        xml_to_base_model(xml_elem, SimpleModel)
    error_msg = str(excinfo.value).lower()
    assert "age" in error_msg
    # Just check that we get some kind of validation error
    assert "validation" in error_msg or "input" in error_msg


def test_xml_wrong_root_tag():
    """Test converting XML with the wrong root tag."""
    xml_str = "<WrongModel><name>John</name><age>30</age><is_active>true</is_active></WrongModel>"
    xml_elem = xml_from_string(xml_str)
    with pytest.raises(Exception) as excinfo:
        xml_to_base_model(xml_elem, SimpleModel)
    error_msg = str(excinfo.value).lower()
    assert "xml validation error" in error_msg
    assert "root tag" in error_msg
    assert "simplemodel" in error_msg


def test_nested_model_valid():
    """Test converting valid XML to a nested model."""
    xml_str = """
    <NestedModel>
        <title>Test</title>
        <details>
            <name>John</name>
            <age>30</age>
            <is_active>true</is_active>
        </details>
        <tags>
            <tag>one</tag>
            <tag>two</tag>
        </tags>
    </NestedModel>
    """
    xml_elem = xml_from_string(xml_str)
    model = xml_to_base_model(xml_elem, NestedModel)
    assert model.title == "Test"
    assert model.details.name == "John"
    assert model.details.age == 30
    assert model.details.is_active is True
    assert model.tags == ["one", "two"]


def test_nested_model_missing_nested_field():
    """Test converting XML with a missing nested field."""
    xml_str = """
    <NestedModel>
        <title>Test</title>
        <details>
            <name>John</name>
            <is_active>true</is_active>
        </details>
        <tags>
            <tag>one</tag>
            <tag>two</tag>
        </tags>
    </NestedModel>
    """
    xml_elem = xml_from_string(xml_str)
    with pytest.raises(Exception) as excinfo:
        xml_to_base_model(xml_elem, NestedModel)
    error_msg = str(excinfo.value).lower()
    # The error might mention details.age or just age depending on the Pydantic version
    assert any(field in error_msg for field in ["details", "age"])


def test_recovery_from_extra_fields():
    """Test recovery from extra fields in XML."""
    xml_str = """
    <SimpleModel>
        <name>John</name>
        <age>30</age>
        <is_active>true</is_active>
        <extra_field>extra value</extra_field>
    </SimpleModel>
    """
    xml_elem = xml_from_string(xml_str)
    model = xml_to_base_model(xml_elem, SimpleModel)
    assert model.name == "John"
    assert model.age == 30
    assert model.is_active is True
    # Extra field should be ignored
