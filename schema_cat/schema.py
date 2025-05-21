import logging
from typing import Type, TypeVar
from xml.etree import ElementTree
from xml.etree.ElementTree import Element, tostring
import re

from pydantic import BaseModel, ValidationError

T = TypeVar('T', bound=BaseModel)

logger = logging.getLogger("schema_cat")


def _wrap_cdata(text: str) -> str:
    return f"<![CDATA[{text}]]>"


def to_this_style(s):
    # Convert to uppercase with underscores, preserving non-alphabetic chars
    return re.sub(r'[^A-Za-z0-9]+', '_', s).strip('_').upper()


def schema_to_xml(schema: Type[BaseModel]) -> ElementTree.XML:
    """Serializes a pydantic type to an example xml representation, always using field description if available (converted to TO_THIS_STYLE). Lists output two elements with the description as content. Does not instantiate the model."""

    def field_to_xml(key, field):
        # List handling
        if hasattr(field.annotation, "__origin__") and field.annotation.__origin__ is list:
            item_type = field.annotation.__args__[0]
            elem = ElementTree.Element(key)
            for _ in range(2):
                # If item_type is a BaseModel, use its name for the child element
                if isinstance(item_type, type) and issubclass(item_type, BaseModel):
                    child = ElementTree.Element(item_type.__name__)
                    for n, f in item_type.model_fields.items():
                        grandchild = field_to_xml(n, f)
                        child.append(grandchild)
                    elem.append(child)
                else:
                    value = getattr(field, 'description', None) or 'example'
                    value = to_this_style(value)
                    child = ElementTree.Element(key[:-1] if key.endswith('s') else key)
                    if item_type is str:
                        child.text = _wrap_cdata(str(value))
                    else:
                        child.text = str(value)
                    elem.append(child)
            return elem
        # Nested model
        if isinstance(field.annotation, type) and issubclass(field.annotation, BaseModel):
            elem = ElementTree.Element(key)
            for n, f in field.annotation.model_fields.items():
                child_elem = field_to_xml(n, f)
                elem.append(child_elem)
            return elem
        # Leaf field
        desc_val = getattr(field, 'description', None)
        value = desc_val if desc_val is not None else 'example'
        value = to_this_style(value)
        elem = ElementTree.Element(key)
        if field.annotation is str:
            elem.text = _wrap_cdata(str(value))
        else:
            elem.text = str(value)
        return elem

    root = ElementTree.Element(schema.__name__)
    for name, field in schema.model_fields.items():
        child_elem = field_to_xml(name, field)
        root.append(child_elem)
    return root


def xml_to_string(xml_tree: ElementTree.XML) -> str:
    """Converts an ElementTree XML element to a pretty-printed XML string, ensuring CDATA sections for str fields."""
    import xml.dom.minidom
    from xml.dom.minidom import parseString, CDATASection

    rough_string = ElementTree.tostring(xml_tree, encoding="utf-8")
    dom = parseString(rough_string)

    def replace_cdata_nodes(node):
        for child in list(node.childNodes):
            if child.nodeType == child.ELEMENT_NODE:
                replace_cdata_nodes(child)
            elif child.nodeType == child.TEXT_NODE:
                if child.data.startswith('<![CDATA[') and child.data.endswith(']]>'):
                    cdata_content = child.data[len('<![CDATA['):-len(']]>')]
                    cdata_node = dom.createCDATASection(cdata_content)
                    node.replaceChild(cdata_node, child)
    replace_cdata_nodes(dom)
    return dom.toprettyxml(indent="  ")


def xml_to_base_model(xml_tree: ElementTree.XML, schema: Type[T]) -> T:
    """Converts an ElementTree XML element to a Pydantic BaseModel instance."""

    def parse_element(elem, schema):
        values = {}
        for name, field in schema.model_fields.items():
            child = elem.find(name)
            if child is None:
                values[name] = None
                continue
            if isinstance(field.annotation, type) and issubclass(field.annotation, BaseModel):
                values[name] = parse_element(child, field.annotation)
            elif hasattr(field.annotation, "__origin__") and field.annotation.__origin__ is list:
                item_type = field.annotation.__args__[0]
                values[name] = []
                # For BaseModel items, look for elements with the item type name
                if isinstance(item_type, type) and issubclass(item_type, BaseModel):
                    # First, check if the child element exists and has children
                    if child is not None and len(list(child)) > 0:
                        # If the child element has children, look for list items there
                        for item_elem in child.findall(item_type.__name__):
                            values[name].append(parse_element(item_elem, item_type))
                    else:
                        # Otherwise, look for list items directly under the parent element
                        for item_elem in elem.findall(item_type.__name__):
                            values[name].append(parse_element(item_elem, item_type))
                else:
                    # For primitive types, look for elements with the field name or singular form
                    # First, check if the child element exists and has children
                    if child is not None and len(list(child)) > 0:
                        # If the child element has children, look for list items there
                        singular_name = name[:-1] if name.endswith('s') else name
                        for item_elem in child.findall(singular_name):
                            values[name].append(item_elem.text)
                    else:
                        # Otherwise, look for list items directly under the parent element
                        # Try the exact field name first
                        items = elem.findall(name)
                        # If not found, try the singular form (remove 's' at the end)
                        if not items and name.endswith('s'):
                            items = elem.findall(name[:-1])
                        for item_elem in items:
                            values[name].append(item_elem.text)
            else:
                try:
                    if field.annotation is int:
                        values[name] = int(child.text)
                    elif field.annotation is float:
                        values[name] = float(child.text)
                    elif field.annotation is bool:
                        values[name] = child.text.lower() == "true"
                    else:
                        values[name] = child.text
                except Exception:
                    # Fallback for invalid type (e.g., 'example' for int)
                    values[name] = None
        try:
            return schema(**values)
        except ValidationError as e:
            logger.error(f"Schema validation failed for {ElementTree.tostring(xml_tree, encoding='utf-8')}")
            raise e

    return parse_element(xml_tree, schema)
