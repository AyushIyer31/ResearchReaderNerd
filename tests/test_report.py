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
    for heading in ("## Summary", "## R1: Assumptions", "## R2: Benchmarks", "## Caveats"):
        assert heading in report
    assert "2026-10-22" in report


def test_assumptions_keep_rank_order_with_kind_marked():
    report = render()
    assert "Top 2, most important first." in report
    # The sample lists the inferred assumption first; rank order wins over stated/inferred.
    first = report.index("1. **Training and test data come from the same distribution.** (inferred, not stated in the paper;")
    second = report.index("2. **Noise is Gaussian.** (stated;")
    assert first < second


def test_proposed_method_bold_and_pipes_escaped():
    report = render()
    assert "| **SampleNet** | accuracy | 91.2 ± 0.3 | Table 2 |" in report
    assert "| Baseline \\| v2 |" in report


def test_empty_analysis():
    data = dict(SAMPLE_ANALYSIS, assumptions=[], benchmarks=[], caveats=[])
    report = render(data)
    assert "No assumptions were identified." in report
    assert "No benchmarks were identified." in report
    assert "## Caveats" not in report
