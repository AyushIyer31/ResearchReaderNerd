from researchreadernerd.analyze import Prompts
from researchreadernerd.consider import consider_benchmarks
from researchreadernerd.pricing import Spend
from researchreadernerd.report import render_report
from researchreadernerd.report_html import render_html
from researchreadernerd.schema import PaperAnalysis

from .conftest import SAMPLE_ANALYSIS


def run(fake_api, budget=1.0, spend=None):
    analysis = PaperAnalysis.model_validate(SAMPLE_ANALYSIS)
    spend = spend or Spend()
    c = consider_benchmarks(analysis, client=fake_api.client, model="claude-haiku-5-5", effort="low",
                            prompts=Prompts(), spend=spend, budget_usd=budget, arxiv_id="2401.01234")
    return analysis, c, spend


def test_suggestions_come_only_from_candidates_with_evidence(fake_api):
    _, c, spend = run(fake_api)
    # Candidate 0 is CIFAR-10 (found in the reference and the co-using paper); candidate 99 does not exist.
    assert [s.name for s in c.suggestions] == ["CIFAR-10"]
    s = c.suggestions[0]
    assert set(s.found_by) == {"references", "co-usage"}
    assert "Used by 1 of the paper's references" in s.evidence
    assert any(l.url.startswith("https://doi.org/") for l in s.links)
    assert any("Dropped 1" in n for n in c.notes)
    assert [x["step"] for x in spend.calls] == ["plan", "extract", "rank"]


def test_own_benchmarks_are_never_suggested(fake_api):
    _, c, _ = run(fake_api)
    # The extraction names ToyBench, which the paper already uses: it must not become a candidate.
    assert c.stats["candidates"] == 2  # CIFAR-10 from the literature + TPC-H from the catalog
    assert all(s.name != "ToyBench" for s in c.suggestions)


def test_used_by_lists_papers_citing_each_benchmark(fake_api):
    _, c, _ = run(fake_api)
    assert c.used_by[0].benchmark == "ToyBench"
    assert c.used_by[0].source.title == "ToyBench: a toy benchmark"
    assert c.used_by[0].papers[0].title == "A paper using ToyBench (2020)"


def test_budget_stops_paid_steps(fake_api):
    spend = Spend()
    spend.add("analyze", "claude-haiku-5-5", 10_000_000, 0)  # pretend R1/R2 already cost $1
    _, c, spend = run(fake_api, spend=spend)
    assert [x["step"] for x in spend.calls] == ["analyze", "plan"]
    assert c.suggestions == []
    assert any("budget" in n for n in c.notes)


def test_reports_render_the_section(fake_api):
    analysis, c, _ = run(fake_api)
    md = render_report(analysis, source="x", model="m", consideration=c)
    assert "## R3: Benchmarks to consider" in md and "1. **CIFAR-10** (references, co-usage)" in md
    html = render_html(analysis, source="x", model="m", consideration=c)
    assert "Benchmarks to consider" in html and "Who else uses this paper" in html
