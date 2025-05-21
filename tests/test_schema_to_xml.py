from pydantic import BaseModel

from schema_cat import schema_to_xml


def xml_to_dict(elem):
    d = {elem.tag: {}}
    for child in elem:
        if len(child):
            d[elem.tag][child.tag] = xml_to_dict(child)[child.tag]
        else:
            d[elem.tag][child.tag] = child.text
    return d


class SimpleModel(BaseModel):
    foo: str
    bar: int
    baz: bool = True


class NestedModel(BaseModel):
    name: str
    child: SimpleModel


class ListModel(BaseModel):
    items: list[int]


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
