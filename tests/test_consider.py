import json

from researchreadernerd import consider
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
    # The confirmed proposal MNIST takes the reserved tier ahead of it.
    assert c.ranker_picks == ["CIFAR-10"] and [s.name for s in c.suggestions] == ["MNIST", "CIFAR-10"]
    s = c.suggestions[1]
    assert set(s.found_by) == {"references", "co-usage"}
    assert "Used by 1 of the paper's references" in s.evidence
    assert any(l.url.startswith("https://doi.org/") for l in s.links)
    assert any("Dropped 1" in n for n in c.notes)
    assert [x["step"] for x in spend.calls] == ["plan", "extract", "bibliographies", "propose", "rank"]


def test_own_benchmarks_are_never_suggested(fake_api):
    _, c, _ = run(fake_api)
    # The extraction names ToyBench, which the paper already uses: it must not become a candidate.
    assert c.stats["candidates"] == 4  # CIFAR-10 (abstracts), CelebA (bibliographies), MNIST (proposal), TPC-H (catalog)
    assert all(s.name != "ToyBench" for s in c.suggestions)


def test_proposals_enter_only_when_openalex_confirms_them(fake_api):
    _, c, _ = run(fake_api)
    # Three proposals: MNIST has a source paper in OpenAlex; ToyBench is the paper's own; Madeup has nothing.
    assert c.stats["proposals"] == 3 and c.stats["proposals_confirmed"] == 1
    ranked = fake_api.requests[-1]["messages"][0]["content"][0]["text"]
    assert "MNIST | Named from the paper's topic; its source paper, \u201cGradient-based learning applied to document recognition\u201d (2020), has 10 citations in OpenAlex" in ranked
    assert "Madeup" not in ranked and "ToyBench |" not in ranked
    # Candidates are listed by evidence: CIFAR-10 (a reference and a co-using paper) before MNIST (one source paper).
    assert ranked.index("CIFAR-10 |") < ranked.index("MNIST |")


def test_proposals_can_be_switched_off(fake_api):
    analysis = PaperAnalysis.model_validate(SAMPLE_ANALYSIS)
    c = consider_benchmarks(analysis, client=fake_api.client, model="claude-haiku-5-5", effort="low", prompts=Prompts(),
                            spend=Spend(), budget_usd=1.0, arxiv_id="2401.01234", propose=False)
    assert c.stats["proposals"] == 0 and c.stats["candidates"] == 3
    assert not any("proposals" in r["output_config"]["format"]["schema"]["properties"] for r in fake_api.requests)


def test_proposals_newer_than_the_paper_are_dropped(fake_api):
    # Replaying an older paper: MNIST's source paper is dated 2020 in the fake, so it did not exist in 2019.
    analysis = PaperAnalysis.model_validate(SAMPLE_ANALYSIS)
    c = consider_benchmarks(analysis, client=fake_api.client, model="claude-haiku-5-5", effort="low", prompts=Prompts(),
                            spend=Spend(), budget_usd=1.0, arxiv_id="2401.01234", up_to_year=2019)
    assert c.stats["proposals"] == 3 and c.stats["proposals_confirmed"] == 0


def test_suggestions_are_ordered_by_evidence(fake_api, monkeypatch):
    from .conftest import fake_message
    # The model lists MNIST (one source paper) before CIFAR-10 (two related papers): the ranking swaps them.
    # MNIST is also a confirmed proposal, so the final list puts it first through the reserved tier.
    picks = {"picks": [{"candidate": 2, "why": "Classic digits."}, {"candidate": 0, "why": "Small images."}]}
    original = fake_api._reply_for

    def reply_for(body):
        props = body.get("output_config", {}).get("format", {}).get("schema", {}).get("properties", {})
        return fake_message(json.dumps(picks)) if "picks" in props else original(body)

    monkeypatch.setattr(fake_api, "_reply_for", reply_for)
    _, c, _ = run(fake_api)
    assert c.ranker_picks == ["CIFAR-10", "MNIST"]
    assert [s.name for s in c.suggestions] == ["MNIST", "CIFAR-10"] and c.tier == ["MNIST"]
    assert c.suggestions[0].found_by == ["model knowledge"] and c.suggestions[0].why == "Classic digits."


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
    assert "## R3: Benchmarks to consider" in md and "1. **MNIST** (model knowledge)" in md and "2. **CIFAR-10** (references, co-usage)" in md
    assert "**Setting:** Unpublished manuscript" in md
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


def test_unpublished_paper_reads_no_citing_papers(fake_api):
    _, c, _ = run(fake_api)
    assert c.mode == "manuscript" and c.stats["citing_papers"] == 0
    assert any("No papers citing this paper" in n for n in c.notes)
    assert all("citing papers" not in s.found_by for s in c.suggestions)


def test_published_mode_reads_citing_papers_and_keeps_their_evidence_apart(fake_api):
    # The fake's citing paper (W3) names CIFAR-10, which the reference also uses: two separate evidence lines.
    analysis = PaperAnalysis.model_validate(SAMPLE_ANALYSIS)
    c = consider_benchmarks(analysis, client=fake_api.client, model="claude-haiku-5-5", effort="low", prompts=Prompts(),
                            spend=Spend(), budget_usd=1.0, arxiv_id="2401.01234", use_citers=True)
    assert c.mode == "published" and c.stats["citing_papers"] == 1
    s = next(x for x in c.suggestions if x.name == "CIFAR-10")
    assert "citing papers" in s.found_by and "references" in s.found_by
    assert "Used by 1 of the paper's references" in s.evidence
    assert "Used by 1 of the most-cited papers citing this paper" in s.evidence
    assert any("existed by 2023" in n for n in c.notes)  # six months before 2401 is 2023
    ranked = fake_api.requests[-1]["messages"][0]["content"][0]["text"]
    assert "citing this paper" in ranked


def test_published_mode_drops_dataset_papers_newer_than_the_cutoff(fake_api):
    # Replaying a January 2019 paper: the cutoff is 2018, and CelebA's dataset paper is dated 2020 in the fake.
    analysis = PaperAnalysis.model_validate(SAMPLE_ANALYSIS)
    c = consider_benchmarks(analysis, client=fake_api.client, model="claude-haiku-5-5", effort="low", prompts=Prompts(),
                            spend=Spend(), budget_usd=1.0, arxiv_id="1901.01234", use_citers=True, up_to_year=2019)
    assert c.stats["dataset_papers"] == 0 and "CelebA" not in c.candidates


def test_benchmark_known_only_from_citing_papers_is_dated(fake_api, monkeypatch):
    from researchreadernerd import scholar
    from .conftest import fake_message
    # The citing paper alone names "NewBench"; OpenAlex finds nothing older than the cutoff naming it, and the
    # one paper that does is from 2024, so it did not exist yet and is left out with a note.
    extraction = {"papers": [{"paper": 0, "benchmarks": ["CIFAR-10"]}, {"paper": 1, "benchmarks": ["NewBench"]}]}
    original = fake_api._reply_for

    def reply_for(body):
        props = body.get("output_config", {}).get("format", {}).get("schema", {}).get("properties", {})
        return fake_message(json.dumps(extraction)) if "papers" in props else original(body)

    def topic_works(query, limit=10, up_to_year=None):
        if up_to_year:
            return []
        return [scholar.Work(id="W77", title="NewBench: a 2024 benchmark", abstract="", year=2024, url="https://doi.org/10.0/77")]

    monkeypatch.setattr(fake_api, "_reply_for", reply_for)
    monkeypatch.setattr(scholar, "topic_works", topic_works)
    analysis = PaperAnalysis.model_validate(SAMPLE_ANALYSIS)
    c = consider_benchmarks(analysis, client=fake_api.client, model="claude-haiku-5-5", effort="low", prompts=Prompts(),
                            spend=Spend(), budget_usd=1.0, arxiv_id="2401.01234", use_citers=True)
    assert c.stats["benchmarks_too_new"] == 1 and "NewBench" not in c.candidates
    assert any("did not exist by 2023, known only from papers that may postdate this one: NewBench" in n for n in c.notes)


def test_replay_dates_benchmarks_known_only_from_co_usage_papers(fake_api, monkeypatch):
    from researchreadernerd import scholar
    from .conftest import fake_message
    # Replaying a 2023 paper with its citing papers withheld: the co-usage paper (W3, fetched with no date limit)
    # alone names "NewBench", which OpenAlex first sees in 2024, so it is left out. Live on an unpublished
    # manuscript (no up_to_year) nothing is dated, since everything the tool sees is older than the manuscript.
    extraction = {"papers": [{"paper": 0, "benchmarks": ["CIFAR-10"]}, {"paper": 1, "benchmarks": ["NewBench"]}]}
    original = fake_api._reply_for

    def reply_for(body):
        props = body.get("output_config", {}).get("format", {}).get("schema", {}).get("properties", {})
        return fake_message(json.dumps(extraction)) if "papers" in props else original(body)

    def topic_works(query, limit=10, up_to_year=None):
        if up_to_year:
            return []
        return [scholar.Work(id="W77", title="NewBench: a 2024 benchmark", abstract="", year=2024, url="https://doi.org/10.0/77")]

    monkeypatch.setattr(fake_api, "_reply_for", reply_for)
    monkeypatch.setattr(scholar, "topic_works", topic_works)
    analysis = PaperAnalysis.model_validate(SAMPLE_ANALYSIS)
    kw = dict(client=fake_api.client, model="claude-haiku-5-5", effort="low", prompts=Prompts(), spend=Spend(), budget_usd=1.0)
    replay = consider_benchmarks(analysis, arxiv_id="2401.01234", use_citers=False, up_to_year=2024, **kw)
    assert replay.mode == "manuscript" and replay.stats["benchmarks_too_new"] == 1 and "NewBench" not in replay.candidates
    live = consider_benchmarks(analysis, arxiv_id="2401.01234", **kw)
    assert live.stats["benchmarks_too_new"] == 0 and "NewBench" in live.candidates


def test_published_mode_can_be_withheld_for_a_replay(fake_api, monkeypatch):
    from researchreadernerd import scholar
    import dataclasses
    real = scholar.find_paper
    monkeypatch.setattr(scholar, "find_paper", lambda title, arxiv_id=None: (dataclasses.replace(real(title)[0], cited_by=500), ["W1"]))
    analysis = PaperAnalysis.model_validate(SAMPLE_ANALYSIS)
    auto = consider_benchmarks(analysis, client=fake_api.client, model="claude-haiku-5-5", effort="low", prompts=Prompts(),
                               spend=Spend(), budget_usd=1.0, arxiv_id="2401.01234")
    held = consider_benchmarks(analysis, client=fake_api.client, model="claude-haiku-5-5", effort="low", prompts=Prompts(),
                               spend=Spend(), budget_usd=1.0, arxiv_id="2401.01234", use_citers=False)
    assert auto.mode == "published" and held.mode == "manuscript"
    assert any("treated as an unpublished manuscript" in n for n in held.notes)


def test_topic_search_stands_in_when_the_paper_has_no_references(fake_api, monkeypatch):
    from researchreadernerd import scholar
    # A paper not yet written: OpenAlex has no record of it. The most-cited papers on its topics are read instead.
    monkeypatch.setattr(scholar, "find_paper", lambda title, arxiv_id=None: None)
    monkeypatch.setattr(scholar, "topic_works", lambda query, limit=10, up_to_year=None: [
        scholar.Work(id="W50", title="A topical paper", abstract="We evaluate on CIFAR-10.", year=2020, url="https://doi.org/10.0/50")])
    _, c, _ = run(fake_api)
    assert c.stats["topic_papers"] == 1 and c.stats["references"] == 0 and c.mode == "draft"
    assert any("stand in for it" in n for n in c.notes)
    s = next(x for x in c.suggestions if x.name == "CIFAR-10")
    assert c.ranker_picks[0] == "CIFAR-10" and "topic search" in s.found_by


def test_confirmed_proposals_take_the_reserved_tier_ahead_of_the_ranking(fake_api):
    _, c, _ = run(fake_api)
    # MNIST is the one confirmed proposal; the fake ranking picks CIFAR-10. Final list: tier first, then the ranking.
    assert c.proposals_confirmed == ["MNIST"] and c.tier == ["MNIST"]
    assert c.ranker_picks == ["CIFAR-10"]
    assert [s.name for s in c.suggestions] == ["MNIST", "CIFAR-10"]
    assert "confirmed in OpenAlex" in c.suggestions[0].why and c.suggestions[1].why == "The standard small image benchmark."


def test_reserve_zero_restores_the_pure_ranking(fake_api):
    analysis = PaperAnalysis.model_validate(SAMPLE_ANALYSIS)
    c = consider_benchmarks(analysis, client=fake_api.client, model="claude-haiku-5-5", effort="low", prompts=Prompts(),
                            spend=Spend(), budget_usd=1.0, arxiv_id="2401.01234", reserve=0)
    assert c.tier == [] and [s.name for s in c.suggestions] == c.ranker_picks == ["CIFAR-10"]


def test_recent_benchmarks_get_the_recent_tier_after_the_proposals(fake_api, monkeypatch):
    from researchreadernerd import scholar
    from .conftest import fake_message
    # The co-usage paper W3 names "FreshBench"; OpenAlex dates it 2024 (a dated mention). Replaying a 2024 paper,
    # 2024 is within RECENT_YEARS, so FreshBench takes a recent slot after the proposal tier even though the
    # fake ranking never picked it. Replaying a 2026 paper, 2024 is old and the list is unchanged.
    extraction = {"papers": [{"paper": 0, "benchmarks": ["CIFAR-10"]}, {"paper": 1, "benchmarks": ["FreshBench"]}]}
    original = fake_api._reply_for

    def reply_for(body):
        props = body.get("output_config", {}).get("format", {}).get("schema", {}).get("properties", {})
        return fake_message(json.dumps(extraction)) if "papers" in props else original(body)

    def topic_works(query, limit=10, up_to_year=None):
        return [scholar.Work(id="W78", title="FreshBench: a 2024 benchmark", abstract="FreshBench results", year=2024, url="https://doi.org/10.0/78")]

    monkeypatch.setattr(fake_api, "_reply_for", reply_for)
    monkeypatch.setattr(scholar, "topic_works", topic_works)
    analysis = PaperAnalysis.model_validate(SAMPLE_ANALYSIS)
    kw = dict(client=fake_api.client, model="claude-haiku-5-5", effort="low", prompts=Prompts(), spend=Spend(), budget_usd=1.0,
              arxiv_id="2401.01234", use_citers=False)
    from researchreadernerd import consider
    recorded = consider_benchmarks(analysis, up_to_year=2024, **kw)  # default: recorded, not placed
    assert recorded.first_seen.get("FreshBench") == 2024 and recorded.recent_candidates == ["FreshBench"]
    assert recorded.recent_tier == [] and "FreshBench" not in [s.name for s in recorded.suggestions]
    monkeypatch.setattr(consider, "RECENT_SLOTS", 1)
    c = consider_benchmarks(analysis, up_to_year=2024, **kw)
    assert c.recent_tier == ["FreshBench"] and [s.name for s in c.suggestions[:2]] == ["MNIST", "FreshBench"]
    assert "FreshBench" not in c.ranker_picks and "Introduced recently (first seen 2024)" in c.suggestions[1].why
    old = consider_benchmarks(analysis, up_to_year=2026, **kw)
    assert old.recent_candidates == [] and "FreshBench" not in [s.name for s in old.suggestions]


def test_second_proposal_call_widens_the_pool_without_changing_the_tier(fake_api, monkeypatch):
    """The subfield call's names join the proposals after the first call's, deduplicated by key (11 Nov 2026)."""
    from .conftest import fake_message
    systems = []
    original = fake_api._reply_for

    def reply_for(body):
        props = body.get("output_config", {}).get("format", {}).get("schema", {}).get("properties", {})
        if "proposals" in props:
            systems.append(body["system"])
            if "specific to this paper's" in body["system"]:
                return fake_message(json.dumps({"proposals": [
                    {"benchmark": "MNIST", "source_title": "Gradient-based learning applied to document recognition"},
                    {"benchmark": "DomainSet", "source_title": "DomainSet: a dataset"}]}))
        return original(body)

    monkeypatch.setattr(fake_api, "_reply_for", reply_for)
    from researchreadernerd.analyze import PROPOSE_DOMAIN_PROMPT
    analysis = PaperAnalysis.model_validate(SAMPLE_ANALYSIS)
    c = consider_benchmarks(analysis, client=fake_api.client, model="claude-haiku-5-5", effort="low", prompts=Prompts(propose_domain=PROPOSE_DOMAIN_PROMPT),
                            spend=Spend(), budget_usd=1.0, arxiv_id="2401.01234")
    assert len(systems) == 2 and systems[0] != systems[1], "one general and one subfield proposal call when the prompt is set"
    assert c.stats["proposals_domain"] == 1, "MNIST was already proposed; only DomainSet is new"
    assert c.stats["proposals"] == 4
    assert c.tier[:1] == ["MNIST"], "the reserved tier still starts from the first call's order"
    assert "DomainSet" in c.candidates


def test_arxiv_bibliography_references_are_noted(fake_api, monkeypatch):
    import dataclasses
    from researchreadernerd import scholar
    work, _ = scholar.find_paper("t")  # the conftest fake lists W1 from OpenAlex
    work = dataclasses.replace(work, references=("W1",))
    monkeypatch.setattr(scholar, "find_paper", lambda title, arxiv_id=None, **kw: (work, ["W1", "W9"]))
    _, c, _ = run(fake_api)
    assert c.stats["references_from_arxiv"] == 1
    assert any("arXiv bibliography" in n for n in c.notes)


def test_candidates_are_dated_and_the_ranker_sees_the_year(fake_api, monkeypatch):
    """When the literature gives no year, one dating call names a source paper per candidate and OpenAlex's record
    gives the year; the ranker sees it (11 Nov 2026). The fixture's related papers carry no year here."""
    from researchreadernerd import scholar

    def work(i, title, refs=()):
        return scholar.Work(id=f"W{i}", title=title, abstract=f"We evaluate on things ({title}).", year=None,
                            url=f"https://doi.org/10.0/{i}", cited_by=10, references=tuple(refs))

    library = {"W1": work(1, "A referenced paper", ["W9", "W2", "W8"]), "W9": work(9, "CelebA: a large-scale face attributes dataset")}
    monkeypatch.setattr(scholar, "get_works", lambda ids: [library[i] for i in ids if i in library])
    monkeypatch.setattr(scholar, "citing_works", lambda work_id, limit=20: [work(3, "A paper using ToyBench", ["W9", "W2"])])
    _, c, spend = run(fake_api)
    assert c.stats["candidates_dated"] >= 1 and c.stats["date_lookups"] >= 1
    assert "first seen: 2020" in c.rank_input, "the conftest OpenAlex fake dates every source paper 2020"
    assert "openalex" in c.first_seen_source.values()
    assert [x["step"] for x in spend.calls] == ["plan", "extract", "bibliographies", "propose", "date", "rank"]


def test_the_ranking_call_uses_the_ranking_model(fake_api, monkeypatch):
    seen = []
    original = fake_api._reply_for

    def reply_for(body):
        props = body.get("output_config", {}).get("format", {}).get("schema", {}).get("properties", {})
        if "picks" in props:
            seen.append(body["model"])
        return original(body)

    monkeypatch.setattr(fake_api, "_reply_for", reply_for)
    _, c, _ = run(fake_api)
    assert seen == [consider.RANK_MODEL] and c.rank_model == consider.RANK_MODEL
    analysis = PaperAnalysis.model_validate(SAMPLE_ANALYSIS)
    c2 = consider_benchmarks(analysis, client=fake_api.client, model="claude-haiku-5-5", effort="low", prompts=Prompts(),
                             spend=Spend(), budget_usd=1.0, rank_model="claude-haiku-5-5", date_candidates=False)
    assert seen[-1] == "claude-haiku-5-5" and c2.rank_model == "claude-haiku-5-5"
    assert c2.stats["candidates_dated"] == 0 and "first seen" not in c2.rank_input, "dating off: the as-run listing, no years"


def test_recency_rule_goes_with_dating(fake_api, monkeypatch):
    """The rule sentence and the years are sent only when candidates are dated (13 Nov 2026), so 'version 2 without the
    recency rule' ranks with exactly the as-run prompt and listing."""
    from researchreadernerd.analyze import RANK_PROMPT, RECENCY_RULE
    systems = []
    original = fake_api._reply_for

    def reply_for(body):
        props = body.get("output_config", {}).get("format", {}).get("schema", {}).get("properties", {})
        if "picks" in props:
            systems.append(body["system"] if isinstance(body["system"], str) else body["system"][0]["text"])
        return original(body)

    monkeypatch.setattr(fake_api, "_reply_for", reply_for)
    analysis = PaperAnalysis.model_validate(SAMPLE_ANALYSIS)
    kw = dict(client=fake_api.client, model="claude-haiku-5-5", effort="low", prompts=Prompts(), budget_usd=1.0)
    on = consider_benchmarks(analysis, spend=Spend(), date_candidates=True, **kw)
    off = consider_benchmarks(analysis, spend=Spend(), date_candidates=False, **kw)
    assert RECENCY_RULE in systems[0] and "first seen" in on.rank_input
    assert systems[1] == RANK_PROMPT.removesuffix(" " + RECENCY_RULE) and RECENCY_RULE not in systems[1]
    assert "first seen" not in off.rank_input


def test_second_proposal_call_is_off_by_default(fake_api):
    _, c, spend = run(fake_api)
    assert "propose_domain" not in [x["step"] for x in spend.calls] and c.stats["proposals_domain"] == 0


def test_launch_ledger_and_no_fallback(fake_api, monkeypatch, tmp_path):
    """Launch config (13 Nov 2026): every Claude call is written to the spend ledger, and with RRN_SERVER_FALLBACK=0 the
    Sonnet ranking call is sent without the server-side fallback, so no other model can answer it."""
    bodies = []
    original = fake_api._reply_for

    def reply_for(body):
        bodies.append(body)
        return original(body)

    monkeypatch.setattr(fake_api, "_reply_for", reply_for)
    ledger = tmp_path / "spend.jsonl"
    monkeypatch.setenv("RRN_SPEND_LEDGER", str(ledger))
    monkeypatch.setenv("RRN_SERVER_FALLBACK", "0")
    _, c, spend = run(fake_api)
    lines = [json.loads(x) for x in ledger.read_text().splitlines()]
    assert len(lines) == len(spend.calls) and all(x["usd"] is not None for x in lines)
    rank = [b for b in bodies if b["model"] == consider.RANK_MODEL]
    assert rank and all("fallbacks" not in b for b in rank)


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
