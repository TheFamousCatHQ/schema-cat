import logging
import re
import warnings
from typing import Tuple
from xml.etree import ElementTree
from xml.parsers.expat import ExpatError

from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning

logger = logging.getLogger("schema_cat")


class XMLParsingError(Exception):
    """Exception raised for errors during XML parsing."""
    pass


def fix_cdata_sections(xml_string: str) -> str:
    """
    Fix CDATA sections in XML strings by ensuring they have proper closing tags.

    Args:
        xml_string: The XML string to fix

    Returns:
        The fixed XML string
    """
    # First, handle the case where CDATA is not closed at all
    if "<![CDATA[" in xml_string and "]]>" not in xml_string:
        # Find where the CDATA section starts
        cdata_start = xml_string.find("<![CDATA[")
        # Find the next tag after the CDATA section
        next_tag = xml_string.find("<", cdata_start + 9)
        if next_tag > cdata_start:
            # Extract the content of the CDATA section
            cdata_content = xml_string[cdata_start + 9:next_tag]
            # Replace the unclosed CDATA section with a properly closed one
            xml_string = xml_string[:cdata_start] + f"<![CDATA[{cdata_content}]]>" + xml_string[next_tag:]

    # Then handle other CDATA issues
    def replace_cdata(match):
        content = match.group(1)
        if content.endswith(']]') or content.endswith(']>'):
            content = content[:-2]
        elif content.endswith(']'):
            content = content[:-1]
        return f'<![CDATA[{content}]]>'

    pattern = r'<!\[CDATA\[(.*?)(?:\]\]*>)'
    fixed_xml = re.sub(pattern, replace_cdata, xml_string, flags=re.DOTALL)
    return fixed_xml


def extract_xml_content(param: str) -> str:
    """
    Extract XML content from a string, handling various edge cases.

    Args:
        param: The string that may contain XML

    Returns:
        A tuple of (xml_string, error_message)

    Raises:
        XMLParsingError: If no XML content can be found
    """
    if not param:
        raise XMLParsingError(f"No XML found in the input '{param}'")

    try:
        xml_start = param.index("<")
        xml_end = param.rfind(">")

        if xml_end <= xml_start:
            raise XMLParsingError(f"Invalid XML: No closing tag found in the input '{param}'")

        xml_string = param[xml_start:xml_end + 1]
        return xml_string
    except ValueError:
        raise XMLParsingError(f"No XML found in the input '{param}'")


def attempt_xml_repair(xml_string: str) -> Tuple[str, bool, str]:
    """
    Attempt to repair common XML issues.

    Args:
        xml_string: The potentially malformed XML string

    Returns:
        A tuple of (repaired_xml, was_repaired, repair_message)
    """
    original = xml_string
    repair_message = []

    # Fix unclosed CDATA sections
    xml_string = fix_cdata_sections(xml_string)
    if xml_string != original:
        repair_message.append("Fixed unclosed CDATA sections")
        original = xml_string

    # Remove invalid XML characters
    xml_string = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', xml_string)
    if xml_string != original:
        repair_message.append("Removed invalid XML characters")
        original = xml_string

    # Try to fix missing closing tags by using BeautifulSoup
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)
            soup = BeautifulSoup(xml_string, "xml")
        soup_xml = str(soup)
        if soup_xml != xml_string and soup_xml.count('<') == soup_xml.count('>'):
            xml_string = soup_xml
            repair_message.append("Fixed missing or mismatched tags")
    except Exception:
        # If BeautifulSoup fails, continue with the original string
        pass

    was_repaired = bool(repair_message)
    repair_message_str = "; ".join(repair_message) if repair_message else "No repairs needed"

    return xml_string, was_repaired, repair_message_str


def xml_from_string(param: str) -> ElementTree.XML:
    """
    Parse a string into an XML ElementTree, with enhanced error handling and recovery.

    Args:
        param: The string to parse, which may contain XML

    Returns:
        An ElementTree.XML object

    Raises:
        XMLParsingError: If the string cannot be parsed as valid XML
    """
    # Extract XML content
    xml_string = extract_xml_content(param)

    # Try to parse the XML
    try:
        # First try with ElementTree directly
        try:
            return ElementTree.fromstring(xml_string)
        except (ElementTree.ParseError, ExpatError) as e:
            logger.warning(f"ElementTree parsing failed: {str(e)}. Trying with BeautifulSoup...")

            # If that fails, try with BeautifulSoup
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)
                soup = BeautifulSoup(xml_string, "xml")

            try:
                return ElementTree.fromstring(str(soup))
            except (ElementTree.ParseError, ExpatError) as e2:
                error_msg = f"XML parsing error: Failed to parse XML after BeautifulSoup cleanup. Original error: {str(e)}. BeautifulSoup error: {str(e2)}"
                logger.error(error_msg)
                raise XMLParsingError(error_msg)
    except Exception as e:
        error_msg = f"XML parsing error: Unexpected error while parsing XML: {str(e)}"
        logger.error(error_msg)
        raise XMLParsingError(error_msg)
