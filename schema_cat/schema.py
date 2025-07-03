import logging
from typing import Type, TypeVar, Union, Dict, Literal, Optional, get_args, get_origin, Any
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

    # Validate that schema is actually a BaseModel class
    if not isinstance(schema, type):
        raise TypeError(f"schema_to_xml expects a Pydantic BaseModel class, but got {type(schema).__name__}: {schema}")

    if not issubclass(schema, BaseModel):
        raise TypeError(f"schema_to_xml expects a Pydantic BaseModel class, but got {schema.__name__} which is not a BaseModel subclass")

    def field_to_xml(key, field):
        origin = get_origin(field.annotation)
        args = get_args(field.annotation)

        # List handling
        if origin is list:
            item_type = args[0]
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

        # Dict handling
        if origin is dict:
            key_type, value_type = args
            elem = ElementTree.Element(key)
            # Add example key-value pairs
            for i in range(2):
                key_elem = ElementTree.Element('key')
                key_desc = f"example_key_{i+1}"
                if key_type is str:
                    key_elem.text = _wrap_cdata(key_desc)
                else:
                    key_elem.text = key_desc

                value_elem = ElementTree.Element('value')
                # Use appropriate example values based on the value type
                if value_type is int:
                    value_elem.text = str(i + 1)  # Simple integer values
                elif value_type is float:
                    value_elem.text = str(float(i + 1.5))  # Simple float values
                elif value_type is bool:
                    value_elem.text = str(i % 2 == 0).lower()  # Alternating boolean values
                elif value_type is str:
                    value_desc = getattr(field, 'description', None) or f"example_value_{i+1}"
                    value_desc = to_this_style(value_desc)
                    value_elem.text = _wrap_cdata(value_desc)
                elif get_origin(value_type) is Union:
                    # For Union types, use the first non-None type
                    union_args = get_args(value_type)
                    for arg in union_args:
                        if arg is not type(None):
                            if arg is int:
                                value_elem.text = str(i + 1)
                            elif arg is float:
                                value_elem.text = str(float(i + 1.5))
                            elif arg is bool:
                                value_elem.text = str(i % 2 == 0).lower()
                            else:  # Default to string
                                value_desc = f"example_union_value_{i+1}"
                                value_elem.text = _wrap_cdata(value_desc)
                            break
                else:
                    # For other types, use a simple string representation
                    value_desc = f"example_value_{i+1}"
                    value_elem.text = value_desc

                item_elem = ElementTree.Element('item')
                item_elem.append(key_elem)
                item_elem.append(value_elem)
                elem.append(item_elem)
            return elem

        # Union handling (including Optional)
        if origin is Union:
            # For Optional (Union[T, None]), use the non-None type
            if type(None) in args:
                # Find the non-None type
                for arg in args:
                    if arg is not type(None):
                        # Use the non-None type directly
                        if isinstance(arg, type) and issubclass(arg, BaseModel):
                            # Handle BaseModel type
                            elem = ElementTree.Element(key)
                            for n, f in arg.model_fields.items():
                                child_elem = field_to_xml(n, f)
                                elem.append(child_elem)
                            return elem
                        elif get_origin(arg) is list:
                            # Handle list type
                            item_type = get_args(arg)[0]
                            elem = ElementTree.Element(key)
                            for _ in range(2):
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
                        elif get_origin(arg) is dict:
                            # Handle dict type
                            key_type, value_type = get_args(arg)
                            elem = ElementTree.Element(key)
                            for i in range(2):
                                key_elem = ElementTree.Element('key')
                                key_desc = f"example_key_{i+1}"
                                if key_type is str:
                                    key_elem.text = _wrap_cdata(key_desc)
                                else:
                                    key_elem.text = key_desc

                                value_elem = ElementTree.Element('value')
                                if value_type is int:
                                    value_elem.text = str(i + 1)
                                elif value_type is float:
                                    value_elem.text = str(float(i + 1.5))
                                elif value_type is bool:
                                    value_elem.text = str(i % 2 == 0).lower()
                                elif value_type is str:
                                    value_desc = getattr(field, 'description', None) or f"example_value_{i+1}"
                                    value_desc = to_this_style(value_desc)
                                    value_elem.text = _wrap_cdata(value_desc)
                                else:
                                    value_desc = f"example_value_{i+1}"
                                    value_elem.text = value_desc

                                item_elem = ElementTree.Element('item')
                                item_elem.append(key_elem)
                                item_elem.append(value_elem)
                                elem.append(item_elem)
                            return elem
                        else:
                            # Handle primitive type
                            elem = ElementTree.Element(key)
                            desc_val = getattr(field, 'description', None)
                            value = desc_val if desc_val is not None else 'example'
                            value = to_this_style(value)
                            if arg is str:
                                elem.text = _wrap_cdata(str(value))
                            else:
                                elem.text = str(value)
                            return elem
            # For regular Union, use the first type as an example
            else:
                # Use the first type directly
                arg = args[0]
                if isinstance(arg, type) and issubclass(arg, BaseModel):
                    # Handle BaseModel type
                    elem = ElementTree.Element(key)
                    for n, f in arg.model_fields.items():
                        child_elem = field_to_xml(n, f)
                        elem.append(child_elem)
                    return elem
                elif get_origin(arg) is list:
                    # Handle list type
                    item_type = get_args(arg)[0]
                    elem = ElementTree.Element(key)
                    for _ in range(2):
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
                elif get_origin(arg) is dict:
                    # Handle dict type
                    key_type, value_type = get_args(arg)
                    elem = ElementTree.Element(key)
                    for i in range(2):
                        key_elem = ElementTree.Element('key')
                        key_desc = f"example_key_{i+1}"
                        if key_type is str:
                            key_elem.text = _wrap_cdata(key_desc)
                        else:
                            key_elem.text = key_desc

                        value_elem = ElementTree.Element('value')
                        if value_type is int:
                            value_elem.text = str(i + 1)
                        elif value_type is float:
                            value_elem.text = str(float(i + 1.5))
                        elif value_type is bool:
                            value_elem.text = str(i % 2 == 0).lower()
                        elif value_type is str:
                            value_desc = getattr(field, 'description', None) or f"example_value_{i+1}"
                            value_desc = to_this_style(value_desc)
                            value_elem.text = _wrap_cdata(value_desc)
                        else:
                            value_desc = f"example_value_{i+1}"
                            value_elem.text = value_desc

                        item_elem = ElementTree.Element('item')
                        item_elem.append(key_elem)
                        item_elem.append(value_elem)
                        elem.append(item_elem)
                    return elem
                else:
                    # Handle primitive type
                    elem = ElementTree.Element(key)
                    desc_val = getattr(field, 'description', None)
                    value = desc_val if desc_val is not None else 'example'
                    value = to_this_style(value)
                    if arg is str:
                        elem.text = _wrap_cdata(str(value))
                    else:
                        elem.text = str(value)
                    return elem

        # Literal handling
        if origin is Literal:
            elem = ElementTree.Element(key)
            # Use the first literal value as an example
            value = str(args[0])
            if isinstance(args[0], str):
                elem.text = _wrap_cdata(value)
            else:
                elem.text = value
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


class XMLValidationError(Exception):
    """Exception raised for errors during XML validation against a schema."""
    pass


def xml_to_base_model(xml_tree: ElementTree.XML, schema: Type[T]) -> T:
    """
    Converts an ElementTree XML element to a Pydantic BaseModel instance with enhanced validation.

    Args:
        xml_tree: The XML element to convert
        schema: The Pydantic model class to convert to

    Returns:
        An instance of the Pydantic model

    Raises:
        XMLValidationError: If the XML doesn't match the expected schema
        ValidationError: If the data doesn't validate against the Pydantic model
    """
    # Validate root tag
    if xml_tree.tag != schema.__name__:
        error_msg = f"XML validation error: Root tag mismatch. Got '{xml_tree.tag}', expected '{schema.__name__}'"
        logger.error(error_msg)
        raise XMLValidationError(f"XML validation error: Root tag mismatch. Expected '{schema.__name__}'.")

    def parse_element(elem, schema):
        values = {}
        missing_required_fields = []
        type_conversion_errors = {}

        for name, field in schema.model_fields.items():
            child = elem.find(name)
            is_required = field.is_required()

            if child is None:
                if is_required:
                    missing_required_fields.append(name)
                values[name] = None
                continue

            origin = get_origin(field.annotation)
            args = get_args(field.annotation)

            # Handle BaseModel
            if isinstance(field.annotation, type) and issubclass(field.annotation, BaseModel):
                try:
                    values[name] = parse_element(child, field.annotation)
                except Exception as e:
                    logger.warning(f"Error parsing nested model '{name}': {str(e)}")
                    if is_required:
                        type_conversion_errors[name] = f"Failed to parse nested model: {str(e)}"
                    values[name] = None

            # Handle List
            elif origin is list:
                item_type = args[0]
                values[name] = []
                parse_errors = []

                # For BaseModel items, look for elements with the item type name
                if isinstance(item_type, type) and issubclass(item_type, BaseModel):
                    # First, check if the child element exists and has children
                    if child is not None and len(list(child)) > 0:
                        # If the child element has children, look for list items there
                        for item_elem in child.findall(item_type.__name__):
                            try:
                                values[name].append(parse_element(item_elem, item_type))
                            except Exception as e:
                                parse_errors.append(str(e))
                    else:
                        # Otherwise, look for list items directly under the parent element
                        for item_elem in elem.findall(item_type.__name__):
                            try:
                                values[name].append(parse_element(item_elem, item_type))
                            except Exception as e:
                                parse_errors.append(str(e))
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

                if parse_errors and is_required and not values[name]:
                    error_msg = "; ".join(parse_errors)
                    type_conversion_errors[name] = f"Failed to parse list items: {error_msg}"

            # Handle Dict
            elif origin is dict:
                values[name] = {}
                key_type, value_type = args
                parse_errors = []

                # Look for item elements containing key-value pairs
                for item_elem in child.findall('item'):
                    key_elem = item_elem.find('key')
                    value_elem = item_elem.find('value')

                    if key_elem is not None and value_elem is not None:
                        key_text = key_elem.text
                        value_text = value_elem.text

                        try:
                            # Convert key to appropriate type
                            if key_type is int:
                                key_val = int(key_text)
                            elif key_type is float:
                                key_val = float(key_text)
                            elif key_type is bool:
                                key_val = key_text.lower() == "true"
                            else:
                                key_val = key_text

                            # Convert value to appropriate type
                            if value_type is int:
                                value_val = int(value_text)
                            elif value_type is float:
                                value_val = float(value_text)
                            elif value_type is bool:
                                value_val = value_text.lower() == "true"
                            else:
                                value_val = value_text

                            values[name][key_val] = value_val
                        except (ValueError, TypeError) as e:
                            parse_errors.append(f"Key-value pair conversion error: {str(e)}")

                if parse_errors and is_required and not values[name]:
                    error_msg = "; ".join(parse_errors)
                    type_conversion_errors[name] = f"Failed to parse dictionary: {error_msg}"

            # Handle Union (including Optional)
            elif origin is Union:
                # Check if this is an Optional type (Union with None)
                is_optional = type(None) in args

                # If it's an Optional field with a default value of None, and the XML element is empty or has no content
                if is_optional and (child is None or (child.text is None and len(list(child)) == 0)):
                    values[name] = None
                    continue

                # Try each type in the Union until one works
                union_errors = []
                success = False

                for arg in args:
                    if arg is type(None):
                        continue
                    try:
                        temp_values = {}

                        if isinstance(arg, type) and issubclass(arg, BaseModel):
                            temp_values[name] = parse_element(child, arg)
                        elif get_origin(arg) is list:
                            # Handle list type within Union
                            # Create a temporary schema for parsing
                            class TempModel(BaseModel):
                                temp_field: arg
                            # Parse using the temporary schema
                            temp_elem = ElementTree.Element('TempModel')
                            temp_elem.append(child)
                            temp_model = parse_element(temp_elem, TempModel)
                            temp_values[name] = temp_model.temp_field
                        elif get_origin(arg) is dict:
                            # Handle dict type within Union
                            key_type, value_type = get_args(arg)
                            temp_dict = {}

                            # Look for item elements containing key-value pairs
                            for item_elem in child.findall('item'):
                                key_elem = item_elem.find('key')
                                value_elem = item_elem.find('value')

                                if key_elem is not None and value_elem is not None:
                                    key_text = key_elem.text
                                    value_text = value_elem.text

                                    # Convert key to appropriate type
                                    if key_type is int:
                                        key_val = int(key_text)
                                    elif key_type is float:
                                        key_val = float(key_text)
                                    elif key_type is bool:
                                        key_val = key_text.lower() == "true"
                                    else:
                                        key_val = key_text

                                    # Convert value to appropriate type
                                    if value_type is int:
                                        value_val = int(value_text)
                                    elif value_type is float:
                                        value_val = float(value_text)
                                    elif value_type is bool:
                                        value_val = value_text.lower() == "true"
                                    else:
                                        value_val = value_text

                                    temp_dict[key_val] = value_val

                            temp_values[name] = temp_dict
                        else:
                            # Handle primitive types
                            if arg is int:
                                temp_values[name] = int(child.text)
                            elif arg is float:
                                temp_values[name] = float(child.text)
                            elif arg is bool:
                                temp_values[name] = child.text.lower() == "true"
                            else:
                                temp_values[name] = child.text

                        values[name] = temp_values[name]
                        success = True
                        break
                    except (ValueError, TypeError, ValidationError) as e:
                        union_errors.append(f"{arg.__name__}: {str(e)}")
                        continue

                # If we couldn't parse any type and it's Optional, set to None
                if not success:
                    if is_optional:
                        values[name] = None
                    elif is_required:
                        error_msg = "; ".join(union_errors)
                        type_conversion_errors[name] = f"Failed to parse union type: {error_msg}"

            # Handle Literal
            elif origin is Literal:
                # Try to match the text with one of the literal values
                text = child.text
                literal_match_found = False

                for arg in args:
                    try:
                        if isinstance(arg, str) and text == arg:
                            values[name] = arg
                            literal_match_found = True
                            break
                        elif isinstance(arg, (int, float, bool)):
                            # Try to convert and compare
                            if isinstance(arg, int) and int(text) == arg:
                                values[name] = arg
                                literal_match_found = True
                                break
                            elif isinstance(arg, float) and float(text) == arg:
                                values[name] = arg
                                literal_match_found = True
                                break
                            elif isinstance(arg, bool) and (text.lower() == "true") == arg:
                                values[name] = arg
                                literal_match_found = True
                                break
                    except (ValueError, TypeError):
                        continue

                # If no match found, use the first literal value as default
                if not literal_match_found:
                    if is_required:
                        allowed_values = ", ".join([str(arg) for arg in args])
                        type_conversion_errors[name] = f"Invalid literal value: '{text}'. Allowed values: {allowed_values}"
                    values[name] = args[0]  # Default to first value

            # Handle primitive types
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
                except Exception as e:
                    # Fallback for invalid type (e.g., 'example' for int)
                    if is_required:
                        type_conversion_errors[name] = f"Type conversion error: {str(e)}"
                    values[name] = None

        # Check for missing required fields and type conversion errors
        if missing_required_fields or type_conversion_errors:
            error_details = []

            if missing_required_fields:
                error_details.append(f"Missing required fields: {', '.join(missing_required_fields)}")

            if type_conversion_errors:
                for field_name, error in type_conversion_errors.items():
                    error_details.append(f"Field '{field_name}': {error}")

            error_msg = f"XML validation error in {schema.__name__}: {'; '.join(error_details)}"
            logger.warning(error_msg)

            # Continue with best effort if we have some values

        try:
            return schema(**values)
        except ValidationError as e:
            error_msg = f"Schema validation failed for {schema.__name__}: {str(e)}"
            logger.error(error_msg)

            # Try to provide more context about the validation error
            error_details = []
            for error in e.errors():
                field_path = ".".join(str(loc) for loc in error["loc"])
                error_details.append(f"Field '{field_path}': {error['msg']}")

            detailed_error = f"{error_msg}\nDetails: {'; '.join(error_details)}"
            # Re-raise the original ValidationError to maintain compatibility
            raise

    return parse_element(xml_tree, schema)
