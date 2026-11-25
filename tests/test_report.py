from datetime import date

from researchreadernerd.report import render_report
from researchreadernerd.schema import PaperAnalysis

from .conftest import SAMPLE_ANALYSIS


def render(data=SAMPLE_ANALYSIS):
    return render_report(
        PaperAnalysis.model_validate(data), source="paper.pdf", model="claude-opus-5-5", today=date(2026, 10, 22)
    )


def test_sections_present():
    report = render()
    assert report.startswith("# A Sample Paper")
    for heading in ("## Summary", "## R2: Benchmarks", "## Caveats"):
        assert heading in report
    assert "2026-10-22" in report


def test_proposed_method_bold_and_pipes_escaped():
    report = render()
    assert "| **SampleNet** | accuracy | 91.2 ± 0.3 | Table 2 |" in report
    assert "| Baseline \\| v2 |" in report


def test_empty_analysis():
    data = dict(SAMPLE_ANALYSIS, benchmarks=[], caveats=[])
    report = render(data)
    assert "No benchmarks were identified." in report
    assert "## Caveats" not in report
