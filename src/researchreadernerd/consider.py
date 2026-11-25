"""Benchmarks a paper might consider, following Shasha's procedure:

1. Benchmarks used by the paper's citations (its references).
2. Benchmarks in public catalogs that match the paper's themes.
3. Benchmarks used by papers that also use the paper's benchmarks (co-usage).
4. Benchmarks the model names from the paper's topic, kept only when OpenAlex confirms
   them: their source paper exists (and may be cited by the related papers), or the
   most-cited papers mentioning the name really name it.
5. For a published paper only: benchmarks used by the most-cited papers that cite it, kept
   only when they existed at least six months before the paper (so its authors could have
   used them). An unpublished manuscript has no citing papers, so this step is skipped.
   Evidence from cited and from citing papers is always reported separately.

Two signals for steps 1 and 3: the model reads the related papers' titles and abstracts, and
we count the papers most often cited in their bibliographies; a benchmark's source paper
("ImageNet: A large-scale ...") shows up there even when no abstract names it. Every
suggestion keeps the evidence our code gathered for it, so none is taken on trust, and the
final list is ordered by that evidence.
"""

from __future__ import annotations

from datetime import datetime

import json
import os
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from pydantic import BaseModel, ConfigDict, Field

from . import scholar
from .analyze import RECENCY_RULE
from .catalogs import CatalogHit, search_all
from .llm import ask, is_local_model
from .pricing import Spend
from .schema import PaperAnalysis

MAX_REFERENCES = 60
MAX_CITERS = 60  # most-cited papers citing a published paper (step 5)
MONTHS_BEFORE = 6  # a benchmark from a citing paper counts only if it existed this long before the paper
MAX_BENCHMARKS_FOR_COUSAGE = 5
CITING_PER_BENCHMARK = 15
PAPERS_PER_EXTRACTION_CALL = 30
MAX_LITERATURE_CANDIDATES = 40
MIN_CITING_RELATED = 2  # a cited work counts only if at least this many related papers cite it
MAX_CITED_TO_LABEL = 80
MAX_PROPOSALS = 15  # names the model may propose from the paper's topic; each is checked in OpenAlex
MENTIONING_PAPERS = 5  # most-cited papers mentioning a proposal, when its source paper is not found
WAVE_SIZE = 4  # related-paper extraction calls per wave; the run's budget is checked between waves
SUGGESTIONS = 10
PROPOSAL_WHY = "Named by the model from the paper's topic and confirmed in OpenAlex; placed ahead of the evidence ranking."
RESERVED_PROPOSALS = 5  # of the 10: the model's own proposals (confirmed in OpenAlex) placed ahead of the evidence ranking
RECENT_TRACKED = 2  # recently introduced candidates with the strongest evidence, recorded per run (recent_candidates)
RECENT_SLOTS = 0  # how many of those enter the 10, after the proposals. 2 was tried on 10 Nov 2026 (60 post-cutoff
# papers): 44 slots, 5 hits, a net loss of 3 points, as first_seen is known for a quarter of candidates and the
# two-users fallback misdates old benchmarks. Kept at 0; the candidates are still recorded for offline comparison.
RECENT_YEARS = 1
RANK_MODEL = os.environ.get("RRN_RANK_MODEL", "claude-sonnet-5-5")  # the one ranking call (11 Nov 2026: Sonnet re-ranked the saved
# pools 5-7 points above Haiku's ranking); every other call stays on the main model. A local main model ranks with itself.
DATE_CANDIDATES = os.environ.get("RRN_DATE_CANDIDATES", "1") != "0"  # ask when each undated candidate was introduced, confirmed in OpenAlex
MAX_DATE_LOOKUPS = 60  # OpenAlex title lookups for dating, per run  # "recent" = first seen in the literature within this many years before the paper's date (or today)
RECENT_WHY = ("Introduced recently (first seen {year}); placed ahead of the ranking because few papers have had time "
              "to use a new benchmark, so its evidence is thin by construction.")
TOPIC_PAPERS_PER_QUERY = 10


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------- What the model returns at each step ----------

class SourceTitle(_Strict):
    benchmark: str
    title: str


class Plan(_Strict):
    field: str = Field(description="The paper's computer-science subfield.")
    themes: list[str]
    catalog_queries: list[str] = Field(description="3 to 5 short keyword queries for dataset catalogs.")
    source_titles: list[SourceTitle] = Field(description="Title of the paper each benchmark is cited to.")


class PaperBenchmarks(_Strict):
    paper: int = Field(description="The paper's number in the list.")
    benchmarks: list[str]


class Extraction(_Strict):
    papers: list[PaperBenchmarks]


class DatasetPaper(_Strict):
    item: int = Field(description="The paper's number in the list.")
    benchmark: str = Field(description="The standard short name of the dataset or benchmark it introduces.")


class DatasetPapers(_Strict):
    dataset_papers: list[DatasetPaper]


class Proposal(_Strict):
    benchmark: str
    source_title: str = Field(description="Title of the paper that introduced it; empty if unknown.")


class Proposals(_Strict):
    proposals: list[Proposal]


class DatedName(_Strict):
    item: int
    source_title: str
    year: int


class DatedNames(_Strict):
    items: list[DatedName]


class Pick(_Strict):
    candidate: int = Field(description="The candidate's number.")
    why: str = Field(description="One sentence on why it fits this paper.")


class Ranking(_Strict):
    picks: list[Pick]


# ---------- What we save ----------

class Link(_Strict):
    title: str
    url: str


class Suggestion(_Strict):
    name: str
    why: str
    found_by: list[str]  # "references", "citing papers", "co-usage", "topic search", "bibliographies", "model knowledge", or a catalog name
    evidence: list[str]  # plain-language facts our code gathered
    links: list[Link]


class BenchmarkUsers(_Strict):
    benchmark: str
    source: Link | None  # the paper the benchmark is cited to, as found in OpenAlex
    papers: list[Link]  # most-cited papers citing that source


class Consideration(_Strict):
    suggestions: list[Suggestion]
    used_by: list[BenchmarkUsers]  # R3: who else works with this paper's benchmarks
    field: str
    themes: list[str]
    catalog_queries: list[str]
    stats: dict[str, int]
    notes: list[str]
    candidates: list[str] = []  # every candidate's name, in the order the model saw them
    proposals_confirmed: list[str] = []  # the model's own proposals that OpenAlex confirmed, in the order it named them
    ranker_picks: list[str] = []  # the 10 the evidence ranking chose on its own, before the reserved tier
    tier: list[str] = []  # the proposals placed ahead of the ranking (the first `reserve` of proposals_confirmed)
    recent_candidates: list[str] = []  # the RECENT_TRACKED strongest candidates first seen within RECENT_YEARS, for comparison
    recent_tier: list[str] = []  # those of them placed in the list after the proposals (the first RECENT_SLOTS)
    first_seen: dict[str, int] = {}
    first_seen_source: dict[str, str] = {}  # per candidate: "literature", "openalex" (dated by its source paper) or "model"
    rank_model: str = ""  # candidate -> year its source paper appeared, or the earliest paper seen using it
    mode: str = ""  # which reader this report serves: "draft", "manuscript" or "published" (see SETTINGS)
    rank_input: str = ""  # the exact text the ranking model saw, so a ranking can be replayed with another model


SETTINGS = {  # the three users (Shasha, 8 Nov 2026) and what the tool read for each
    "draft": "Paper not yet written: no reference list in OpenAlex, so the most-cited papers on its topics were read "
             "instead, plus the model's own proposals checked in OpenAlex.",
    "manuscript": "Unpublished manuscript (or a reviewer's copy): the papers it cites were read, plus the model's own "
                  "proposals checked in OpenAlex. Papers citing it: none.",
    "published": "Published paper: the papers it cites and the most-cited papers citing it were both read, so the list "
                 "shows what the field runs that this paper did not; a benchmark known only from citing papers counts "
                 "only if it existed six months before the paper. Cited and citing evidence are shown separately.",
}


# ---------- The pipeline ----------

def _key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _already_used(name: str, own: list[str]) -> bool:
    k = _key(name)
    return any(k and o and (k in o or o in k) for o in (_key(x) for x in own))


def cutoff_year(arxiv_id: str | None, year: int | None) -> int | None:
    """The last year in which a benchmark must have existed to count as available MONTHS_BEFORE months
    before the paper: from the arXiv id's month when there is one, else the year before publication."""
    if arxiv_id and re.match(r"^\d{4}\.", arxiv_id):
        y, m = 2000 + int(arxiv_id[:2]), int(arxiv_id[2:4]) - MONTHS_BEFORE
        return y - 1 if m <= 0 else y
    return year - 1 if year else None


def consider_benchmarks(
    analysis: PaperAnalysis,
    *,
    client,  # from llm.make_client(model)
    model: str,
    effort: str,
    prompts,  # analyze.Prompts (plan, extract, rank)
    spend: Spend,
    budget_usd: float,
    arxiv_id: str | None = None,
    progress: Callable[[str], None] = lambda stage: None,
    use_references: bool = True,
    topic_search: bool | None = None,
    propose: bool = True,
    use_citers: bool | None = None,
    up_to_year: int | None = None,
    reserve: int = RESERVED_PROPOSALS,
    rank_model: str | None = None,
    date_candidates: bool | None = None,
) -> Consideration:
    """use_references: read the paper's bibliography (step 1). topic_search: also gather the
    most-cited papers matching the plan's catalog queries, published up to `up_to_year`; this
    stands in for references when a paper has none. None decides from the paper: on when OpenAlex
    has no record of it or lists no references (a paper not yet written, known by its title and
    abstract), off otherwise. Experiments pass True or False.
    propose: let the model name benchmarks from the paper's topic, each checked in OpenAlex (step 4).
    use_citers: read the most-cited papers citing this paper (step 5). None decides from the paper:
    read them when OpenAlex lists any (a published paper), skip them otherwise. Experiments pass
    False to replay a published paper as the unpublished manuscript it once was.
    reserve: how many of the 10 suggestions are kept for the model's confirmed proposals, in the order it
    named them, ahead of the evidence ranking (9 Nov 2026: the ranking alone left standard benchmarks the
    model knew in the pool). 0 restores the pure ranking; the ranker's own picks are saved either way."""
    notes: list[str] = []
    own = [b.name for b in analysis.benchmarks]
    if rank_model is None:
        rank_model = model if is_local_model(model) else RANK_MODEL
    if date_candidates is None:
        date_candidates = DATE_CANDIDATES

    def over_budget(step: str) -> bool:
        if spend.cost >= budget_usd:
            notes.append(f"Skipped {step}: this run reached its ${budget_usd:.2f} budget.")
            return True
        return False

    # Plan: themes, catalog queries, and the titles of the benchmarks' source papers.
    progress("plan")
    bench_lines = "\n".join(f"- {b.name}" + (f" (cited as: {b.cited_as})" if b.cited_as else "") for b in analysis.benchmarks)
    plan_input = (f"Title: {analysis.title}\nSummary: {analysis.summary}\n"
                  f"Proposed method: {analysis.proposed_method}\nBenchmarks used:\n{bench_lines or '- none found'}")
    answer = ask(client, model=model, effort=effort, system=prompts.plan,
                 content=[{"type": "text", "text": plan_input}], schema=Plan, max_tokens=8000)
    spend.add("plan", answer.model, answer.input_tokens, answer.output_tokens)
    plan = answer.value

    # Step 1 and 3: the paper's references, and papers that use its benchmarks.
    progress("related")
    references: list[scholar.Work] = []
    citers: list[scholar.Work] = []  # step 5: the most-cited papers citing this one, if it is published
    found = None
    from_arxiv: list[str] = []
    try:
        found = scholar.find_paper(analysis.title, arxiv_id)
        if found is None:
            notes.append("OpenAlex has no record of this paper, so its references could not be checked.")
        else:
            # OpenAlex's reference list first, then the entries of the paper's own arXiv bibliography that OpenAlex
            # had not linked (find_paper); those are kept inside the cap, since they are the newest papers cited.
            from_arxiv = [r for r in found[1] if r not in found[0].references]
            ref_ids = (from_arxiv + [r for r in found[1] if r not in from_arxiv])[:MAX_REFERENCES] if use_references else []
            references = scholar.get_works(ref_ids) if ref_ids else []
            if use_references and from_arxiv:
                notes.append(f"{len(from_arxiv)} of the {len(ref_ids)} references read came from the paper's own arXiv "
                             "bibliography; OpenAlex had not linked them.")
            if use_references and not ref_ids:
                notes.append("OpenAlex lists no references for this paper.")
            if use_citers or (use_citers is None and found[0].cited_by > 0):
                citers = [w for w in scholar.citing_works(found[0].id, MAX_CITERS) if w.id != found[0].id]
    except scholar.ScholarError as e:
        notes.append(f"References could not be fetched: {e}")
    # The year a benchmark must have existed by. Set for a published paper (citing papers postdate it) and for a
    # replay (`up_to_year`: co-usage papers are fetched without a date limit, so they can postdate the paper too).
    cutoff = (cutoff_year(arxiv_id, found[0].year if found else None) if citers
              else cutoff_year(arxiv_id, up_to_year + 1) if up_to_year else None)
    mode = "published" if citers else "manuscript" if references else "draft"
    if citers:
        notes.append(f"Published paper: OpenAlex lists {found[0].cited_by:,} papers citing it, so the {len(citers)} "
                     f"most-cited were read too. A benchmark found only through them counts if it existed by "
                     f"{cutoff}, {MONTHS_BEFORE} months before the paper; evidence from cited and from citing papers is kept apart."
                     if cutoff else
                     f"Published paper: the {len(citers)} most-cited papers citing it were read too; the paper's date is "
                     f"unknown, so benchmarks found only through them could not be checked for existing before it.")
    elif found and found[0].cited_by > 0:
        notes.append("Papers citing this paper were not read: it is treated as an unpublished manuscript.")
    else:
        notes.append("No papers citing this paper are listed in OpenAlex, so only the papers it cites were read "
                     "(an unpublished manuscript).")

    used_by: list[BenchmarkUsers] = []
    citing: dict[str, list[scholar.Work]] = {}
    own_sources: set[str] = set()
    titles = {s.benchmark: s.title for s in plan.source_titles}
    for bench in analysis.benchmarks[:MAX_BENCHMARKS_FOR_COUSAGE]:
        title = titles.get(bench.name) or next((t for b, t in titles.items() if _key(b) == _key(bench.name)), None)
        if not title:
            used_by.append(BenchmarkUsers(benchmark=bench.name, source=None, papers=[]))
            continue
        try:
            source = scholar.find_by_title(title)
            papers = scholar.citing_works(source.id, CITING_PER_BENCHMARK) if source else []
        except scholar.ScholarError as e:
            notes.append(f"Papers using {bench.name} could not be fetched: {e}")
            source, papers = None, []
        if source is None and not any(bench.name in n for n in notes):
            notes.append(f"No OpenAlex paper matches the source title given for {bench.name}, so its co-usage was not checked.")
        citing[bench.name] = papers
        if source:
            own_sources.add(source.id)
        used_by.append(BenchmarkUsers(
            benchmark=bench.name,
            source=Link(title=source.title, url=source.url) if source else None,
            papers=[Link(title=f"{p.title} ({p.year})" if p.year else p.title, url=p.url) for p in papers],
        ))

    topical: list[scholar.Work] = []
    if topic_search is None:
        topic_search = not references
        if topic_search:
            notes.append("With no reference list to read, the most-cited papers on the paper's topics stand in for it.")
    if topic_search:
        for query in plan.catalog_queries[:5]:
            try:
                topical += scholar.topic_works(query, TOPIC_PAPERS_PER_QUERY, up_to_year=up_to_year)
            except scholar.ScholarError as e:
                notes.append(f"Topic search for '{query}' failed: {e}")
        if found:
            topical = [w for w in topical if w.id != found[0].id]

    # Read titles and abstracts; list the benchmarks each related paper names.
    papers: dict[str, scholar.Work] = {w.id: w for w in references}
    papers.update({w.id: w for w in citers})
    papers.update({w.id: w for w in topical})
    for works in citing.values():
        papers.update({w.id: w for w in works})
    paper_list = [w for w in papers.values() if w.title]

    # The bibliographies: which works do the related papers cite most often?
    this_paper = found[0].id if found else None
    cited_count: Counter = Counter()
    for w in paper_list:
        cited_count.update(set(w.references))
    for skip in own_sources | {this_paper} | set(papers):
        cited_count.pop(skip, None)  # the paper's own benchmarks, the paper, and the related papers themselves
    top_cited = [i for i, n in cited_count.most_common(MAX_CITED_TO_LABEL) if n >= MIN_CITING_RELATED]
    cited_works: list[scholar.Work] = []
    if top_cited:
        try:
            cited_works = [w for w in scholar.get_works(top_cited) if w.title]
            cited_works.sort(key=lambda w: -cited_count[w.id])
        except scholar.ScholarError as e:
            notes.append(f"The related papers' bibliographies could not be checked: {e}")

    mentions: dict[str, list[str]] = {}
    dataset_papers: list[tuple[str, scholar.Work]] = []  # (benchmark name, the paper introducing it)
    proposals: list[Proposal] = []  # step 4: what the model names from the paper's topic, unchecked
    chunks = [paper_list[i:i + PAPERS_PER_EXTRACTION_CALL] for i in range(0, len(paper_list), PAPERS_PER_EXTRACTION_CALL)]

    def extract(chunk: list[scholar.Work]):
        text = "\n\n".join(f"[{i}] {w.title}\n{w.abstract[:1200]}" for i, w in enumerate(chunk))
        return chunk, ask(client, model=model, effort="low", system=prompts.extract,
                          content=[{"type": "text", "text": text}], schema=Extraction, max_tokens=8000)

    def label():
        text = "\n".join(f"[{i}] {w.title}" + (f" ({w.year})" if w.year else "") for i, w in enumerate(cited_works))
        return ask(client, model=model, effort="low", system=prompts.label,
                   content=[{"type": "text", "text": text}], schema=DatasetPapers, max_tokens=8000)

    def propose_names(system: str):
        # Title, summary, method, field and themes, up to 15 names. A baseline-style prompt (title and summary,
        # 10 names) was tried on 10 Nov 2026 and reverted: it gave fewer confirmed proposals (7 against 10-11 a
        # run) that overlapped the evidence ranking's picks, and the reserved tier became a net loss.
        # A second call with prompts.propose_domain asks for the subfield's own datasets (11 Nov 2026: a third of
        # the benchmarks the search never found were ones the model could name); it only widens the pool, since
        # the reserved tier takes the first call's proposals first.
        text = f"{plan_input}\nField: {plan.field}\nThemes: {', '.join(plan.themes)}"
        return ask(client, model=model, effort=effort, system=system,
                   content=[{"type": "text", "text": text}], schema=Proposals, max_tokens=4000)

    domain_proposals = 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        can_propose = propose and prompts.propose.strip() and not over_budget("proposing benchmarks")
        proposed = pool.submit(propose_names, prompts.propose) if can_propose else None
        proposed_domain = pool.submit(propose_names, prompts.propose_domain) if can_propose and getattr(prompts, "propose_domain", "").strip() else None
        if (paper_list or cited_works) and not over_budget("reading related papers"):
            labelled = pool.submit(label) if cited_works and prompts.label.strip() else None
            # Extraction runs in waves with the budget re-checked between them (Harsh Kashyap, PR #1, 21 Nov 2026), so a
            # run that crosses its budget mid-extraction stops instead of firing every remaining paid call. One wave can
            # still be in flight when the crossing is seen, and the labelling and proposal calls running alongside are
            # counted when they finish.
            for start in range(0, len(chunks), WAVE_SIZE):
                if spend.cost >= budget_usd:
                    notes.append(f"Stopped reading related papers at the ${budget_usd:.2f} budget; "
                                 f"{len(chunks) - start} of {len(chunks)} batches were not read.")
                    break
                for chunk, answer in pool.map(extract, chunks[start : start + WAVE_SIZE]):
                    spend.add("extract", answer.model, answer.input_tokens, answer.output_tokens)
                    for item in answer.value.papers:
                        if 0 <= item.paper < len(chunk):
                            mentions[chunk[item.paper].id] = item.benchmarks
            if labelled:
                answer = labelled.result()
                spend.add("bibliographies", answer.model, answer.input_tokens, answer.output_tokens)
                for d in answer.value.dataset_papers:
                    if 0 <= d.item < len(cited_works) and d.benchmark.strip():
                        dataset_papers.append((d.benchmark.strip(), cited_works[d.item]))
        if proposed:
            answer = proposed.result()
            spend.add("propose", answer.model, answer.input_tokens, answer.output_tokens)
            proposals = answer.value.proposals[:MAX_PROPOSALS]
        if proposed_domain:
            answer = proposed_domain.result()
            spend.add("propose_domain", answer.model, answer.input_tokens, answer.output_tokens)
            seen_keys = {_key(x.benchmark) for x in proposals}
            for x in answer.value.proposals[:MAX_PROPOSALS]:
                if _key(x.benchmark) and _key(x.benchmark) not in seen_keys:
                    seen_keys.add(_key(x.benchmark)); proposals.append(x); domain_proposals += 1

    # Count who uses what.
    display: dict[str, Counter] = defaultdict(Counter)
    in_refs: dict[str, set[str]] = defaultdict(set)
    in_citers: dict[str, set[str]] = defaultdict(set)
    in_topic: dict[str, set[str]] = defaultdict(set)
    co_use: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    ref_ids = {w.id for w in references}
    citer_ids = {w.id for w in citers}
    topic_ids = {w.id for w in topical}
    for paper_id, names in mentions.items():
        for name in names:
            k = _key(name)
            if not k or _already_used(name, own):
                continue
            display[k][name.strip()] += 1
            if paper_id in ref_ids:
                in_refs[k].add(paper_id)
            if paper_id in citer_ids:
                in_citers[k].add(paper_id)
            if paper_id in topic_ids:
                in_topic[k].add(paper_id)
            for bench, works in citing.items():
                if any(w.id == paper_id for w in works):
                    co_use[k][bench].add(paper_id)

    # A dataset paper cited by many related papers: keep the most-cited one per benchmark.
    in_bibs: dict[str, tuple[int, scholar.Work]] = {}
    for name, w in dataset_papers:
        k = _key(name)
        if not k or _already_used(name, own):
            continue
        if cutoff and w.year and w.year > cutoff:
            continue  # a dataset paper newer than the cutoff: only the citing papers can have cited it
        display[k][name] += 1
        if cited_count[w.id] > in_bibs.get(k, (0, None))[0]:
            in_bibs[k] = (cited_count[w.id], w)

    # Step 4: the model's proposals, each checked in OpenAlex before it can be a candidate.
    confirmed: dict[str, dict] = {}  # key -> evidence, links and weight of a confirmed proposal
    progress("verify")
    for prop in proposals:
        name, k = prop.benchmark.strip(), _key(prop.benchmark)
        if not k or k in confirmed or _already_used(name, own):
            continue
        try:
            source = scholar.find_by_title(prop.source_title) if prop.source_title.strip() else None
            mentioning = [] if source else scholar.topic_works(name, MENTIONING_PAPERS, up_to_year=up_to_year)
        except scholar.ScholarError as e:
            notes.append(f"The proposal {name} could not be checked in OpenAlex: {e}")
            continue
        if source and source.id in own_sources:
            continue  # one of the paper's own benchmarks under another name
        if source and up_to_year and source.year and source.year > up_to_year:
            continue  # did not exist yet (experiments replay older papers; the tool passes no year)
        mentioning = [w for w in mentioning if name.lower() in f"{w.title} {w.abstract}".lower()]
        if source is None and not mentioning:
            continue  # nothing in OpenAlex backs the name: never shown
        evidence, links, weight = [], [], 1
        if source:
            n = cited_count.get(source.id, 0)
            evidence.append(f"Named from the paper's topic; its source paper, \u201c{source.title}\u201d"
                            + (f" ({source.year})" if source.year else "") + f", has {source.cited_by:,} citations in OpenAlex"
                            + (f" and is cited by {n} of the {len(paper_list)} related papers" if n else ""))
            links.append(Link(title=source.title, url=source.url))
            weight += n
        else:
            evidence.append(f"Named from the paper's topic; {len(mentioning)} of the most-cited papers OpenAlex "
                            f"finds for the name {'uses' if len(mentioning) == 1 else 'use'} it in the title or abstract")
            links += [Link(title=w.title, url=w.url) for w in mentioning[:3]]
        display[k][name] += 1
        years = [w.year for w in ([source] if source else mentioning) if w.year]
        confirmed[k] = {"evidence": evidence, "links": links, "weight": weight, "year": min(years) if years else None}

    def strength(k: str) -> int:
        return (len(in_refs[k]) + len(in_citers[k]) + len(in_topic[k]) + sum(len(v) for v in co_use[k].values())
                + in_bibs.get(k, (0, None))[0] + confirmed.get(k, {}).get("weight", 0))

    lit_keys = sorted(display, key=strength, reverse=True)[:MAX_LITERATURE_CANDIDATES]
    lit_keys += [k for k in confirmed if k not in lit_keys]  # a confirmed proposal is always a candidate

    # A benchmark backed only by papers that may postdate this one (its citing papers, or papers using the same
    # benchmarks) must have existed by the cutoff. OpenAlex is asked for the most-cited paper up to that year
    # naming it; none found means it stays, flagged. References and topic papers predate the paper; dataset
    # papers and confirmed proposals were checked by their own year above.
    dated: dict[str, scholar.Work | None] = {}
    too_new: list[str] = []
    if cutoff:
        for k in list(lit_keys):
            if in_refs[k] or in_topic[k] or k in in_bibs or k in confirmed:
                continue
            name = display[k].most_common(1)[0][0]
            try:
                older = [w for w in scholar.topic_works(name, MENTIONING_PAPERS, up_to_year=cutoff)
                         if name.lower() in f"{w.title} {w.abstract}".lower()]
                newer = older or [w for w in scholar.topic_works(name, 1) if name.lower() in f"{w.title} {w.abstract}".lower()]
            except scholar.ScholarError as e:
                notes.append(f"Whether {name} existed before the paper could not be checked: {e}")
                dated[k] = None
                continue
            if older:
                dated[k] = older[0]
            elif newer and newer[0].year and newer[0].year > cutoff:
                too_new.append(name)  # every paper naming it is newer than the cutoff: it did not exist yet
                lit_keys.remove(k)
            else:
                dated[k] = None

    # Step 2: public catalogs.
    progress("catalogs")
    hits, catalog_notes = search_all(plan.catalog_queries[:5])
    notes += catalog_notes
    hits = [h for h in hits if not _already_used(h.name, own)]

    # Number the candidates, each with the evidence that surfaced it.
    candidates: list[dict] = []
    for k in lit_keys:
        evidence, links = [], []
        if k in in_bibs:
            n, w = in_bibs[k]
            evidence.append(f"Its source paper, \u201c{w.title}\u201d, is cited by {n} of the {len(paper_list)} related papers")
            links.append(Link(title=w.title, url=w.url))
        if in_refs[k]:
            evidence.append(f"Used by {len(in_refs[k])} of the paper's references")
        if in_citers[k]:
            evidence.append(f"Used by {len(in_citers[k])} of the most-cited papers citing this paper")
        if k in dated:
            w = dated[k]
            evidence.append(f"Existed before the paper: named in \u201c{w.title}\u201d ({w.year})" if w else
                            f"No paper naming it up to {cutoff} was found in OpenAlex, so whether it existed before this paper is unverified")
        if in_topic[k]:
            evidence.append(f"Used by {len(in_topic[k])} of the most-cited papers on the same topics")
        for bench, ids in co_use[k].items():
            evidence.append(f"Used by {len(ids)} of the most-cited papers that also use {bench}")
        for pid in list(in_refs[k]) + list(in_citers[k]) + list(in_topic[k]) + [i for ids in co_use[k].values() for i in ids]:
            w = papers.get(pid)
            if w and len(links) < 3 and all(l.url != w.url for l in links):
                links.append(Link(title=w.title, url=w.url))
        if k in confirmed:
            evidence += confirmed[k]["evidence"]
            links += [l for l in confirmed[k]["links"] if all(x.url != l.url for x in links)][: max(0, 3 - len(links))]
        found_by = ((["references"] if in_refs[k] else []) + (["citing papers"] if in_citers[k] else [])
                    + (["topic search"] if in_topic[k] else [])
                    + (["co-usage"] if co_use[k] else [])
                    + (["bibliographies"] if k in in_bibs else [])
                    + (["model knowledge"] if k in confirmed else []))
        # When the benchmark first appeared: its source paper's year when one is known (bibliographies, a confirmed
        # proposal, the dated mention); else the earliest of the papers seen using it, taken only when there are two
        # or more of them, since one paper's year says little about the benchmark's age.
        source_years = [y for y in ((in_bibs[k][1].year if k in in_bibs else None),
                                    confirmed.get(k, {}).get("year"),
                                    (dated[k].year if dated.get(k) else None)) if y]
        using = [papers[pid].year for pid in set(in_refs[k]) | set(in_citers[k]) | set(in_topic[k])
                 | {i for ids in co_use[k].values() for i in ids} if pid in papers and papers[pid].year]
        first = min(source_years) if source_years else (min(using) if len(using) >= 2 else None)
        candidates.append({"name": display[k].most_common(1)[0][0], "found_by": found_by,
                           "evidence": evidence, "links": links, "strength": strength(k), "first_seen": first})
    for hit in hits:
        candidates.append({"name": hit.name, "found_by": [hit.catalog],
                           "evidence": [f"Found in {hit.catalog} by keyword search" + (f": {hit.description}" if hit.description else "")],
                           "links": [Link(title=f"{hit.name} ({hit.catalog})", url=hit.url)], "strength": 0})

    names = [c["name"] for c in candidates]
    if too_new:
        notes.append(f"Left out {len(too_new)} benchmark{'s' if len(too_new) > 1 else ''} that did not exist by {cutoff}, "
                     f"known only from papers that may postdate this one: {', '.join(too_new[:5])}{'...' if len(too_new) > 5 else ''}.")
    # Dating: when was each still-undated literature candidate introduced? One model call names a source paper per
    # candidate; OpenAlex confirms it and gives the year. A year the model gives without a confirming record is kept
    # and marked "model". Nothing is dropped here: existence before the paper was checked above. (11 Nov 2026.)
    first_source = {c["name"]: "literature" for c in candidates if c.get("first_seen")}
    dated_count = date_lookups = 0
    undated = [c for c in candidates if not c.get("first_seen") and c["strength"] > 0]
    if date_candidates and undated and prompts.date.strip() and not over_budget("dating candidates"):
        progress("date")
        listing = "\n".join(f"[{i}] {c['name']}" for i, c in enumerate(undated))
        answer = ask(client, model=model, effort="low", system=prompts.date,
                     content=[{"type": "text", "text": listing}], schema=DatedNames, max_tokens=8000)
        spend.add("date", answer.model, answer.input_tokens, answer.output_tokens)
        for d in answer.value.items:
            if not 0 <= d.item < len(undated):
                continue
            c = undated[d.item]
            year, source = None, None
            if d.source_title.strip() and date_lookups < MAX_DATE_LOOKUPS:
                date_lookups += 1
                try:
                    w = scholar.find_by_title(d.source_title)
                except scholar.ScholarError as e:
                    notes.append(f"Dating stopped: {e}")
                    break
                if w and w.year and w.id not in own_sources:
                    year, source = w.year, "openalex"
            if year is None and d.year > 0:
                year, source = d.year, "model"
            if year:
                c["first_seen"] = year; first_source[c["name"]] = source; dated_count += 1
    stats = {"references": len(references), "citing_papers": len(citers), "topic_papers": len(topical), "related_papers": len(paper_list),
             "papers_read": len(mentions), "cited_works_checked": len(cited_works),
             "dataset_papers": len(in_bibs), "proposals": len(proposals), "proposals_confirmed": len(confirmed),
             "proposals_domain": domain_proposals, "references_from_arxiv": len(from_arxiv),
             "candidates_dated": dated_count, "date_lookups": date_lookups,
             "benchmarks_too_new": len(too_new),
             "candidates": len(candidates), "catalog_hits": len(hits)}
    if not candidates:
        notes.append("No candidate benchmarks were found.")
        return Consideration(suggestions=[], used_by=used_by, field=plan.field, themes=plan.themes,
                             catalog_queries=plan.catalog_queries, stats=stats, notes=notes, candidates=names, mode=mode)

    # Rank.
    progress("rank")
    if over_budget("ranking"):
        return Consideration(suggestions=[], used_by=used_by, field=plan.field, themes=plan.themes,
                             catalog_queries=plan.catalog_queries, stats=stats, notes=notes, candidates=names, mode=mode)
    # Year-aware ranking (the recency rule) goes with dating: the years are shown and the rule sentence kept only when
    # candidates are dated; with dating off the listing and the prompt are the as-run ones (13 Nov 2026).
    if date_candidates:
        listing = "\n".join(f"[{i}] {c['name']} | {'; '.join(c['evidence'])}; first seen: {c.get('first_seen') or 'unknown'}"
                            for i, c in enumerate(candidates))
        rank_system = prompts.rank
    else:
        listing = "\n".join(f"[{i}] {c['name']} | {'; '.join(c['evidence'])}" for i, c in enumerate(candidates))
        rank_system = prompts.rank.removesuffix(" " + RECENCY_RULE)
    rank_input = (f"{plan_input}\nField: {plan.field}\nThemes: {', '.join(plan.themes)}\n\n"
                  f"Candidates:\n{listing}")
    answer = ask(client, model=rank_model, effort=effort, system=rank_system,
                 content=[{"type": "text", "text": rank_input}], schema=Ranking, max_tokens=8000)
    spend.add("rank", answer.model, answer.input_tokens, answer.output_tokens)

    picked: list[tuple[int, Pick]] = []
    seen: set[int] = set()
    dropped = 0
    for pick in answer.value.picks:
        if not 0 <= pick.candidate < len(candidates) or pick.candidate in seen:
            dropped += 1  # a number that is not on the list: never shown
            continue
        seen.add(pick.candidate)
        picked.append((pick.candidate, pick))
        if len(picked) == SUGGESTIONS:
            break
    picked.sort(key=lambda x: -candidates[x[0]]["strength"])  # by evidence; the model's order among equals
    ranker_picks = [candidates[i]["name"] for i, _ in picked]
    why = {i: pick.why for i, pick in picked}
    # Reserved tier: the model's confirmed proposals, in the order it named them, take the first `reserve` places;
    # the ranking fills the rest. Nothing enters that is not a candidate with evidence.
    proposal_names = [display[k].most_common(1)[0][0] for k in confirmed]
    index = {c["name"]: i for i, c in enumerate(candidates)}
    tier = [index[n] for n in proposal_names if n in index][: max(reserve, 0)]
    # Recent tier: benchmarks first seen within RECENT_YEARS of the paper's date (or today, for a live run), the
    # strongest evidence first. A new benchmark has few users yet, so the evidence ranking passes it over for
    # established names; this is also the ground the model cannot know from training, so it is where the literature
    # adds the most. Fixed before the November 2026 runs; the ranking's own picks stay in ranker_picks for comparison.
    ref_year = up_to_year or datetime.now().year
    recent_all = sorted((i for i, c in enumerate(candidates)
                         if i not in tier and c["strength"] > 0 and c.get("first_seen") and c["first_seen"] >= ref_year - RECENT_YEARS),
                        key=lambda i: -candidates[i]["strength"])[: max(RECENT_TRACKED, 0)]
    recent = recent_all[: max(RECENT_SLOTS, 0)]
    final = (tier + recent + [i for i, _ in picked if i not in tier and i not in recent])[:SUGGESTIONS]
    def reason(i: int) -> str:
        if i in why:
            return why[i]
        return RECENT_WHY.format(year=candidates[i]["first_seen"]) if i in recent else PROPOSAL_WHY
    suggestions = [Suggestion(name=candidates[i]["name"], why=reason(i), found_by=candidates[i]["found_by"],
                              evidence=candidates[i]["evidence"], links=candidates[i]["links"]) for i in final]
    if dropped:
        notes.append(f"Dropped {dropped} ranking choices that did not match a candidate.")
    return Consideration(suggestions=suggestions, used_by=used_by, field=plan.field, themes=plan.themes,
                         catalog_queries=plan.catalog_queries, stats=stats, notes=notes, candidates=names,
                         rank_input=rank_input, mode=mode, proposals_confirmed=proposal_names, ranker_picks=ranker_picks,
                         tier=[candidates[i]["name"] for i in tier], recent_tier=[candidates[i]["name"] for i in recent],
                         recent_candidates=[candidates[i]["name"] for i in recent_all],
                         first_seen={c["name"]: c["first_seen"] for c in candidates if c.get("first_seen")},
                         first_seen_source={n: src for n, src in first_source.items() if src}, rank_model=rank_model)


def to_json(c: Consideration) -> str:
    return json.dumps(c.model_dump(), indent=2)
