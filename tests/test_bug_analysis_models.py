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


def test_roundtrip_with_lists():
    """Test that a BugAnalysisReport with lists can be serialized to XML and parsed back."""
    # Create a BugAnalysisReport with lists
    bug1 = BugAnalysis(
        file_path="file1.py",
        line_number="42",
        description="Bug 1",
        severity="high",
        confidence="high",
        suggested_fix="Fix bug 1",
        code_snippet="print('bug 1')"
    )
    bug2 = BugAnalysis(
        file_path="file2.py",
        line_number="24",
        description="Bug 2",
        severity="medium",
        confidence="medium",
        suggested_fix="Fix bug 2",
        code_snippet="print('bug 2')"
    )
    report = BugAnalysisReport(
        commit_id="abc123",
        timestamp="2024-01-01T00:00:00Z",
        affected_files=["file1.py", "file2.py"],
        bugs=[bug1, bug2],
        summary="Test report"
    )

    # Manually create XML for the report
    from xml.etree.ElementTree import Element, SubElement

    def bug_to_xml(bug: BugAnalysis) -> Element:
        bug_elem = Element("BugAnalysis")
        for field, value in bug.model_dump().items():
            child = SubElement(bug_elem, field)
            child.text = value
        return bug_elem

    def report_to_xml(report: BugAnalysisReport) -> Element:
        root = Element("BugAnalysisReport")
        SubElement(root, "commit_id").text = report.commit_id
        SubElement(root, "timestamp").text = report.timestamp
        affected_files_elem = SubElement(root, "affected_files")
        for f in report.affected_files:
            af = SubElement(affected_files_elem, "affected_file")
            af.text = f
        bugs_elem = SubElement(root, "bugs")
        for b in report.bugs:
            bugs_elem.append(bug_to_xml(b))
        SubElement(root, "summary").text = report.summary
        return root

    # Serialize to XML
    xml_elem = report_to_xml(report)
    xml = xml_to_string(xml_elem)

    # Parse back to model
    result = xml_to_base_model(xml_elem, BugAnalysisReport)

    # Verify the result
    assert isinstance(result, BugAnalysisReport)
    assert result.commit_id == report.commit_id
    assert result.timestamp == report.timestamp
    assert len(result.affected_files) == len(report.affected_files)
    for i, file in enumerate(report.affected_files):
        assert result.affected_files[i] == file
    assert len(result.bugs) == len(report.bugs)
    for i, bug in enumerate(report.bugs):
        assert result.bugs[i].file_path == bug.file_path
        assert result.bugs[i].line_number == bug.line_number
        assert result.bugs[i].description == bug.description
        assert result.bugs[i].severity == bug.severity
        assert result.bugs[i].confidence == bug.confidence
        assert result.bugs[i].suggested_fix == bug.suggested_fix
        assert result.bugs[i].code_snippet == bug.code_snippet
    assert result.summary == report.summary
