import re
from xml.etree import ElementTree

from bs4 import BeautifulSoup


def fix_cdata_sections(xml_string: str) -> str:
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


def xml_from_string(param: str) -> ElementTree.XML:
    xml_start = param.index("<")
    xml_end = param.rfind(">")
    xml_string = param[xml_start:xml_end + 1]
    xml_string = fix_cdata_sections(xml_string)
    soup = BeautifulSoup(xml_string, "xml")
    xml = ElementTree.fromstring(str(soup))
    return xml
