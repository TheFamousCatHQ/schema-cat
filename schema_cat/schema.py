import logging
from typing import Type, TypeVar
from xml.etree import ElementTree
from xml.etree.ElementTree import Element, tostring

from pydantic import BaseModel, ValidationError

T = TypeVar('T', bound=BaseModel)

logger = logging.getLogger("schema_cat")


def _wrap_cdata(text: str) -> str:
    return f"<![CDATA[{text}]]>"


def schema_to_xml(schema: Type[BaseModel]) -> ElementTree.XML:
    """Serializes a pydantic type to an example xml representation, wrapping str fields in CDATA."""

    def example_value(field):
        if not field.is_required():
            if field.default_factory is not None:
                return field.default_factory()
            return field.default
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

    def dict_to_xml(tag, d, model=None):
        elem = ElementTree.Element(tag)
        model_fields = getattr(model, 'model_fields', None) if model else None
        for key, val in d.items():
            field_type = None
            if model_fields and key in model_fields:
                field_type = model_fields[key].annotation
            if isinstance(val, dict):
                # Nested model
                child_model = field_type if (isinstance(field_type, type) and issubclass(field_type, BaseModel)) else None
                elem.append(dict_to_xml(key, val, child_model))
            elif isinstance(val, list):
                if not val:
                    elem.append(ElementTree.Element(key))
                else:
                    for item in val:
                        if isinstance(item, dict):
                            child_model = None
                            if field_type and hasattr(field_type, "__origin__") and field_type.__origin__ is list:
                                item_type = field_type.__args__[0]
                                if isinstance(item_type, type) and issubclass(item_type, BaseModel):
                                    child_model = item_type
                            elem.append(dict_to_xml(key, item, child_model))
                        else:
                            child = ElementTree.Element(key)
                            # For lists, try to infer type from field_type
                            item_type = None
                            if field_type and hasattr(field_type, "__origin__") and field_type.__origin__ is list:
                                item_type = field_type.__args__[0]
                            if item_type is str:
                                child.text = _wrap_cdata(str(item))
                            else:
                                child.text = str(item)
                            elem.append(child)
            else:
                child = ElementTree.Element(key)
                if field_type is str:
                    child.text = _wrap_cdata(str(val))
                else:
                    child.text = str(val)
                elem.append(child)
        return elem

    root = dict_to_xml(schema.__name__, data, schema)
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
                for item_elem in elem.findall(name):
                    if isinstance(item_type, type) and issubclass(item_type, BaseModel):
                        values[name].append(parse_element(item_elem, item_type))
                    else:
                        values[name].append(item_elem.text)
            else:
                if field.annotation is int:
                    values[name] = int(child.text)
                elif field.annotation is float:
                    values[name] = float(child.text)
                elif field.annotation is bool:
                    values[name] = child.text.lower() == "true"
                else:
                    values[name] = child.text
        try:
            return schema(**values)
        except ValidationError as e:
            logger.error(f"Schema validation failed for {ElementTree.tostring(xml_tree, encoding='utf-8')}")
            raise e

    return parse_element(xml_tree, schema)
