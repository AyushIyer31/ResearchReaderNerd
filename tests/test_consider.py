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
    # The bibliography labelling call runs first so the budget check between
    # extraction waves sees its cost.
    assert [x["step"] for x in spend.calls] == ["plan", "bibliographies", "extract", "rank"]


def test_own_benchmarks_are_never_suggested(fake_api):
    _, c, _ = run(fake_api)
    # The extraction names ToyBench, which the paper already uses: it must not become a candidate.
    assert c.stats["candidates"] == 3  # CIFAR-10 (abstracts), CelebA (bibliographies), TPC-H (catalog)
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


def test_bibliographies_surface_dataset_papers(fake_api):
    _, c, _ = run(fake_api)
    # W9 (CelebA's paper) is cited by both related papers; W8 by only one, so it is not checked.
    # W2 is ToyBench's own source paper, so it is left out before Claude sees the list.
    label_call = next(r for r in fake_api.requests
                      if "dataset_papers" in r["output_config"]["format"]["schema"]["properties"])
    listing = label_call["messages"][0]["content"][0]["text"]
    assert "CelebA" in listing and "ToyBench" not in listing
    assert c.stats["cited_works_checked"] == 1 and c.stats["dataset_papers"] == 1
    # Item 7 does not exist: ignored. CelebA becomes a candidate with its evidence and link.
    ranked = fake_api.requests[-1]["messages"][0]["content"][0]["text"]
    assert "CelebA | Its source paper, \u201cCelebA: a large-scale face attributes dataset\u201d, is cited by 2 of the 2 related papers" in ranked
    assert "Not listed" not in ranked


def test_bibliography_failure_is_a_note_not_a_crash(fake_api, monkeypatch):
    from researchreadernerd import scholar

    real = scholar.get_works

    def flaky(ids):
        if "W9" in ids:
            raise scholar.ScholarError("OpenAlex request failed (HTTP 503).")
        return real(ids)

    monkeypatch.setattr(scholar, "get_works", flaky)
    _, c, spend = run(fake_api)
    assert any("bibliographies could not be checked" in n for n in c.notes)
    assert "bibliographies" not in [x["step"] for x in spend.calls]


def test_budget_stops_between_extraction_waves(fake_api, monkeypatch):
    """A run that crosses its budget mid-extraction must not fire the remaining paid calls."""
    import researchreadernerd.pricing as pricing
    from researchreadernerd import scholar
    from researchreadernerd.analyze import Prompts
    from researchreadernerd.schema import PaperAnalysis
    import tests.conftest as conftest

    # 60 references + 5 benchmarks x 15 citing papers = 135 related papers = 5 extraction
    # batches of 30; a wave is 4, so batch 5 is the one the budget gate must stop.
    def work(i, title):
        return scholar.Work(id=f"W{i}", title=title, abstract=f"About {title}.", year=2021,
                            url=f"https://doi.org/10.1/{i}", cited_by=5, references=())

    monkeypatch.setattr(scholar, "find_paper",
                        lambda title, arxiv_id=None: (work(0, title), [f"W{i}" for i in range(1, 61)]))
    monkeypatch.setattr(scholar, "get_works", lambda ids: [work(i, f"Paper {i}") for i in ids])
    monkeypatch.setattr(scholar, "find_by_title", lambda title: work(2, title))
    per_bench = {}

    def citing(work_id, limit=20):
        n = per_bench.get(work_id, 0) + 1
        per_bench[work_id] = n
        return [work(1000 + 100 * n + i, f"Citing {n}-{i}") for i in range(15)]

    monkeypatch.setattr(scholar, "citing_works", citing)
    # Every call costs $0.50; the budget allows the plan plus one wave (4 calls), not batch 5.
    monkeypatch.setattr(pricing, "estimate_cost", lambda model, i, o: 0.50)
    monkeypatch.setitem(conftest.SAMPLE_PLAN, "source_titles",
                        [{"benchmark": f"Bench{b}", "title": f"Bench{b} source"} for b in range(1, 6)])

    analysis = PaperAnalysis.model_validate(
        {**conftest.SAMPLE_ANALYSIS,
         "benchmarks": [{"name": f"Bench{b}", "task": "t", "splits": [], "metrics": [],
                         "cited_as": None, "results": [], "notes": None} for b in range(1, 6)]})
    spend = Spend()
    c = consider_benchmarks(analysis, client=fake_api.client, model="claude-haiku-5-5", effort="low",
                            prompts=Prompts(), spend=spend, budget_usd=2.20, arxiv_id="2401.01234")

    extraction_calls = [r for r in fake_api.requests
                        if "papers" in r["output_config"]["format"]["schema"]["properties"]]
    assert len(extraction_calls) == 4  # one wave; batch 5 never ran
    assert [x["step"] for x in spend.calls].count("extract") == 4
    assert any("budget" in n and "1 of 5 batches were not read" in n for n in c.notes)
    assert c.stats["papers_read"] == 8  # 2 named papers per batch x 4 batches
