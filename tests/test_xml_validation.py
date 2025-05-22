import pytest
from pydantic import BaseModel, Field
from xml.etree import ElementTree
import re
from typing import List, Optional

from schema_cat.xml import xml_from_string
from schema_cat.schema import xml_to_base_model


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


def test_malformed_xml_missing_closing_tag():
    """Test parsing XML with a missing closing tag."""
    xml_str = "<SimpleModel><name>John</name><age>30</age><is_active>true</is_active>"
    with pytest.raises(Exception) as excinfo:
        xml_from_string(xml_str)
    assert "XML parsing error" in str(excinfo.value)
    assert "missing closing tag" in str(excinfo.value).lower()


def test_malformed_xml_invalid_tag():
    """Test parsing XML with an invalid tag."""
    xml_str = "<SimpleModel><name>John<name><age>30</age><is_active>true</is_active></SimpleModel>"
    with pytest.raises(Exception) as excinfo:
        xml_from_string(xml_str)
    assert "XML parsing error" in str(excinfo.value)
    assert "mismatched tag" in str(excinfo.value).lower() or "no matching tag" in str(excinfo.value).lower()


def test_malformed_xml_invalid_attribute():
    """Test parsing XML with an invalid attribute."""
    xml_str = "<SimpleModel invalid><name>John</name><age>30</age><is_active>true</is_active></SimpleModel>"
    with pytest.raises(Exception) as excinfo:
        xml_from_string(xml_str)
    assert "XML parsing error" in str(excinfo.value)
    assert "invalid attribute" in str(excinfo.value).lower() or "syntax error" in str(excinfo.value).lower()


def test_malformed_xml_unclosed_cdata():
    """Test parsing XML with an unclosed CDATA section."""
    xml_str = "<SimpleModel><name><![CDATA[John</name><age>30</age><is_active>true</is_active></SimpleModel>"
    # This should be fixed by fix_cdata_sections
    xml_elem = xml_from_string(xml_str)
    assert xml_elem.tag == "SimpleModel"
    assert xml_elem.find("name").text == "John"


def test_empty_input():
    """Test parsing an empty string."""
    xml_str = ""
    with pytest.raises(Exception) as excinfo:
        xml_from_string(xml_str)
    assert "XML parsing error" in str(excinfo.value)
    assert "empty input" in str(excinfo.value).lower()


def test_non_xml_input():
    """Test parsing a string that doesn't contain XML."""
    xml_str = "This is not XML"
    with pytest.raises(Exception) as excinfo:
        xml_from_string(xml_str)
    assert "XML parsing error" in str(excinfo.value)
    assert "no xml found" in str(excinfo.value).lower()


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


def test_recovery_from_fixable_issues():
    """Test recovery from fixable issues in XML."""
    # Missing closing tag for CDATA, should be fixed by fix_cdata_sections
    xml_str = """
    <SimpleModel>
        <name><![CDATA[John</name>
        <age>30</age>
        <is_active>true</is_active>
    </SimpleModel>
    """
    xml_elem = xml_from_string(xml_str)
    model = xml_to_base_model(xml_elem, SimpleModel)
    assert model.name == "John"
    assert model.age == 30
    assert model.is_active is True


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
