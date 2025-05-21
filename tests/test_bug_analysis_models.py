from typing import List

from pydantic import Field, BaseModel

from schema_cat import schema_to_xml, xml_to_string, xml_to_base_model


class BugAnalysis(BaseModel):
    """Model for bug analysis results."""
    file_path: str = Field(...)
    line_number: str = Field(...)
    description: str = Field(...)
    severity: str = Field(...)
    confidence: str = Field(...)
    suggested_fix: str = Field(...)
    code_snippet: str = Field(...)


class BugAnalysisReport(BaseModel):
    """Model for the complete bug analysis report."""
    commit_id: str = Field(...)
    timestamp: str = Field(...)
    affected_files: List[str] = Field(...)
    bugs: List[BugAnalysis] = Field(default_factory=list)
    summary: str = Field(...)


def test_to_xml():
    xml_elem = schema_to_xml(BugAnalysisReport)
    xml = xml_to_string(xml_elem)
    assert xml is not None
    model = xml_to_base_model(xml_elem, BugAnalysisReport)
    assert model is not None
