from datetime import date

from researchreadernerd.report_html import render_html
from researchreadernerd.schema import PaperAnalysis

from .conftest import SAMPLE_ANALYSIS


def render(data=SAMPLE_ANALYSIS, **kwargs):
    return render_html(
        PaperAnalysis.model_validate(data), source="https://arxiv.org/pdf/2401.01234", model="claude-opus-5-5",
        today=date(2026, 10, 22), **kwargs,
    )


def test_standalone_and_fragment():
    assert render().startswith("<!doctype html>")
    fragment = render(standalone=False)
    assert fragment.startswith("<title>A Sample Paper</title>")
    assert "<html" not in fragment


def test_text_is_escaped():
    data = dict(SAMPLE_ANALYSIS, title="<script>alert(1)</script>")
    html = render(data)
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_results_pivoted_one_row_per_method():
    html = render()
    assert "<th>accuracy</th>" in html
    assert html.count('<tr class="proposed"><td>SampleNet</td>') == 1
    assert "Baseline | v2" in html


def test_same_method_from_two_locations_gets_two_rows():
    bench = dict(SAMPLE_ANALYSIS["benchmarks"][0])
    bench["results"] = bench["results"] + [
        {"method": "SampleNet", "metric": "accuracy", "value": "90.0", "is_proposed_method": True, "location": "Section 5 text"}
    ]
    html = render(dict(SAMPLE_ANALYSIS, benchmarks=[bench]))
    assert html.count('<tr class="proposed"><td>SampleNet</td>') == 2


def test_inferred_assumptions_marked():
    html = render()
    assert 'class="tag inferred">Inferred' in html
    assert "1 · the results depend on it" in html


def test_sections_collapsible_with_benchmarks_closed():
    html = render()
    assert html.count('<details class="sec" open>') == 4  # summary, R1, R2, caveats
    assert '<details class="sub" open><summary><h3>Stated in the paper' in html
    assert '<details class="sub"><summary><h3>ToyBench</h3><span class="sum-meta">classification · 2 rows' in html
    assert 'data-open="0">Collapse all' in html
