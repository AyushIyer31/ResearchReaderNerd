"""Benchmarks a paper might consider, following Shasha's procedure:

1. Benchmarks used by the paper's citations (its references).
2. Benchmarks in public catalogs that match the paper's themes.
3. Benchmarks used by papers that also use the paper's benchmarks (co-usage).

Two signals for steps 1 and 3: Claude reads the related papers' titles and abstracts, and
we count the papers most often cited in their bibliographies; a benchmark's source paper
("ImageNet: A large-scale ...") shows up there even when no abstract names it. Every
suggestion keeps the evidence our code gathered for it, so none is taken on trust.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import anthropic
from pydantic import BaseModel, ConfigDict, Field

from . import scholar
from .catalogs import CatalogHit, search_all
from .llm import ask
from .pricing import Spend
from .schema import PaperAnalysis

MAX_REFERENCES = 60
MAX_BENCHMARKS_FOR_COUSAGE = 5
CITING_PER_BENCHMARK = 15
PAPERS_PER_EXTRACTION_CALL = 30
MAX_LITERATURE_CANDIDATES = 40
MIN_CITING_RELATED = 2  # a cited work counts only if at least this many related papers cite it
MAX_CITED_TO_LABEL = 80
SUGGESTIONS = 10


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------- What Claude returns at each step ----------

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
    found_by: list[str]  # "references", "co-usage", "bibliographies", or a catalog name
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


# ---------- The pipeline ----------

def _key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _already_used(name: str, own: list[str]) -> bool:
    k = _key(name)
    return any(k and o and (k in o or o in k) for o in (_key(x) for x in own))


def consider_benchmarks(
    analysis: PaperAnalysis,
    *,
    client: anthropic.Anthropic,
    model: str,
    effort: str,
    prompts,  # analyze.Prompts (plan, extract, rank)
    spend: Spend,
    budget_usd: float,
    arxiv_id: str | None = None,
    progress: Callable[[str], None] = lambda stage: None,
) -> Consideration:
    notes: list[str] = []
    own = [b.name for b in analysis.benchmarks]

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
    found = None
    try:
        found = scholar.find_paper(analysis.title, arxiv_id)
        if found is None:
            notes.append("OpenAlex has no record of this paper, so its references could not be checked.")
        else:
            ref_ids = found[1][:MAX_REFERENCES]
            references = scholar.get_works(ref_ids)
            if not ref_ids:
                notes.append("OpenAlex lists no references for this paper.")
    except scholar.ScholarError as e:
        notes.append(f"References could not be fetched: {e}")

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

    # Read titles and abstracts; list the benchmarks each related paper names.
    papers: dict[str, scholar.Work] = {w.id: w for w in references}
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
    if (paper_list or cited_works) and not over_budget("reading related papers"):
        chunks = [paper_list[i:i + PAPERS_PER_EXTRACTION_CALL] for i in range(0, len(paper_list), PAPERS_PER_EXTRACTION_CALL)]

        def extract(chunk: list[scholar.Work]):
            text = "\n\n".join(f"[{i}] {w.title}\n{w.abstract[:1200]}" for i, w in enumerate(chunk))
            return chunk, ask(client, model=model, effort="low", system=prompts.extract,
                              content=[{"type": "text", "text": text}], schema=Extraction, max_tokens=8000)

        def label():
            text = "\n".join(f"[{i}] {w.title}" + (f" ({w.year})" if w.year else "") for i, w in enumerate(cited_works))
            return ask(client, model=model, effort="low", system=prompts.label,
                       content=[{"type": "text", "text": text}], schema=DatasetPapers, max_tokens=8000)

        with ThreadPoolExecutor(max_workers=4) as pool:
            labelled = pool.submit(label) if cited_works and prompts.label.strip() else None
            for chunk, answer in pool.map(extract, chunks):
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

    # Count who uses what.
    display: dict[str, Counter] = defaultdict(Counter)
    in_refs: dict[str, set[str]] = defaultdict(set)
    co_use: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    ref_ids = {w.id for w in references}
    for paper_id, names in mentions.items():
        for name in names:
            k = _key(name)
            if not k or _already_used(name, own):
                continue
            display[k][name.strip()] += 1
            if paper_id in ref_ids:
                in_refs[k].add(paper_id)
            for bench, works in citing.items():
                if any(w.id == paper_id for w in works):
                    co_use[k][bench].add(paper_id)

    # A dataset paper cited by many related papers: keep the most-cited one per benchmark.
    in_bibs: dict[str, tuple[int, scholar.Work]] = {}
    for name, w in dataset_papers:
        k = _key(name)
        if not k or _already_used(name, own):
            continue
        display[k][name] += 1
        if cited_count[w.id] > in_bibs.get(k, (0, None))[0]:
            in_bibs[k] = (cited_count[w.id], w)

    def strength(k: str) -> int:
        return len(in_refs[k]) + sum(len(v) for v in co_use[k].values()) + in_bibs.get(k, (0, None))[0]

    lit_keys = sorted(display, key=strength, reverse=True)[:MAX_LITERATURE_CANDIDATES]

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
        for bench, ids in co_use[k].items():
            evidence.append(f"Used by {len(ids)} of the most-cited papers that also use {bench}")
        for pid in list(in_refs[k]) + [i for ids in co_use[k].values() for i in ids]:
            w = papers.get(pid)
            if w and len(links) < 3 and all(l.url != w.url for l in links):
                links.append(Link(title=w.title, url=w.url))
        found_by = ((["references"] if in_refs[k] else []) + (["co-usage"] if co_use[k] else [])
                    + (["bibliographies"] if k in in_bibs else []))
        candidates.append({"name": display[k].most_common(1)[0][0], "found_by": found_by,
                           "evidence": evidence, "links": links})
    for hit in hits:
        candidates.append({"name": hit.name, "found_by": [hit.catalog],
                           "evidence": [f"Found in {hit.catalog} by keyword search" + (f": {hit.description}" if hit.description else "")],
                           "links": [Link(title=f"{hit.name} ({hit.catalog})", url=hit.url)]})

    stats = {"references": len(references), "related_papers": len(paper_list),
             "papers_read": len(mentions), "cited_works_checked": len(cited_works),
             "dataset_papers": len(in_bibs), "candidates": len(candidates), "catalog_hits": len(hits)}
    if not candidates:
        notes.append("No candidate benchmarks were found.")
        return Consideration(suggestions=[], used_by=used_by, field=plan.field, themes=plan.themes,
                             catalog_queries=plan.catalog_queries, stats=stats, notes=notes)

    # Rank.
    progress("rank")
    if over_budget("ranking"):
        return Consideration(suggestions=[], used_by=used_by, field=plan.field, themes=plan.themes,
                             catalog_queries=plan.catalog_queries, stats=stats, notes=notes)
    listing = "\n".join(f"[{i}] {c['name']} | {'; '.join(c['evidence'])}" for i, c in enumerate(candidates))
    rank_input = (f"{plan_input}\nField: {plan.field}\nThemes: {', '.join(plan.themes)}\n\n"
                  f"Candidates:\n{listing}")
    answer = ask(client, model=model, effort=effort, system=prompts.rank,
                 content=[{"type": "text", "text": rank_input}], schema=Ranking, max_tokens=8000)
    spend.add("rank", answer.model, answer.input_tokens, answer.output_tokens)

    suggestions: list[Suggestion] = []
    seen: set[int] = set()
    dropped = 0
    for pick in answer.value.picks:
        if not 0 <= pick.candidate < len(candidates) or pick.candidate in seen:
            dropped += 1  # a number that is not on the list: never shown
            continue
        seen.add(pick.candidate)
        c = candidates[pick.candidate]
        suggestions.append(Suggestion(name=c["name"], why=pick.why, found_by=c["found_by"],
                                      evidence=c["evidence"], links=c["links"]))
        if len(suggestions) == SUGGESTIONS:
            break
    if dropped:
        notes.append(f"Dropped {dropped} ranking choices that did not match a candidate.")
    return Consideration(suggestions=suggestions, used_by=used_by, field=plan.field, themes=plan.themes,
                         catalog_queries=plan.catalog_queries, stats=stats, notes=notes)


def to_json(c: Consideration) -> str:
    return json.dumps(c.model_dump(), indent=2)
