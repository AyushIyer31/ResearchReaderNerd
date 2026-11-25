"""Evaluate the benchmark reader on a sample of arXiv papers, with a local and a hosted model.

    python experiments/benchmark_eval.py papers                      # pick the paper set from sample_raw.json
    python experiments/benchmark_eval.py run --models qwen2.5:14b     # phase A, one process per model
    python experiments/benchmark_eval.py run --models claude-haiku-5-5
    python experiments/benchmark_eval.py score                        # phase B: truth, replays, scores, summary

Everything is cached under experiments/out/, so a rerun only does missing work.

Phase A, per paper and model:
  extract/<id>.<model>.json        R2 from the full PDF: benchmarks with every number, tokens, seconds, text cut?
  blind/<id>.<model>.<cond>.json   R3 from title + abstract only, the paper's own benchmarks hidden.
                                   cond "refs":  candidates from the paper's references (OpenAlex) + catalogs
                                   cond "topic": candidates from a topic search limited to papers up to the
                                                 paper's year + catalogs (for papers with no references listed)
Phase B (Claude does the matching; both models rank):
  truth/<id>.json                  the union of both models' benchmark lists, deduplicated and marked
                                   public / own / not a benchmark, each with a snippet from the paper.
                                   Authors correct these by hand and set checked_by_author.
  verbatim/<id>.<model>.json       every copied number checked against the paper's text
  replay/<id>.<pool>.<cond>.<ranker>.json   a ranking replayed on the other model's candidate pool
  match/<id>.<pool>.<cond>.json    which candidate in a pool is each truth benchmark, judged once per pool
                                   so that every ranking replayed on that pool is scored the same way
  results.json, summary.md
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from researchreadernerd.analyze import AnalysisError, Prompts, analyze_paper  # noqa: E402
from researchreadernerd.cli import load_dotenv  # noqa: E402
from researchreadernerd.consider import SUGGESTIONS, Ranking, consider_benchmarks  # noqa: E402
from researchreadernerd.llm import CHARS_PER_TOKEN, DEFAULT_NUM_CTX, MAX_LOCAL_OUTPUT, ask, is_local_model, make_client  # noqa: E402
from researchreadernerd.paper import Paper, load_paper, pdf_text  # noqa: E402
from researchreadernerd.pricing import Spend  # noqa: E402
from researchreadernerd import scholar  # noqa: E402
from researchreadernerd.schema import PaperAnalysis  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
# RRN_OUT and RRN_PAPERS select another paper set with its own outputs (the post-cutoff 2026 set: out_2026, papers_2026.json)
OUT = HERE / os.environ.get("RRN_OUT", "out")
# RRN_USER1_ARXIV_BIB=1 (launch config, stages B and C): user-1 runs with a reference list also read the paper's arXiv
# bibliography (version 2, change 3). Off by default, which is the as-run harness.
USER1_ARXIV_BIB = os.environ.get("RRN_USER1_ARXIV_BIB") == "1"
SKIP_REPLAY = os.environ.get("RRN_SKIP_REPLAY") == "1"
PDFS = HERE / "pdfs"
PAPERS = HERE / os.environ.get("RRN_PAPERS", "papers.json")
MATCH_MODEL = "claude-haiku-5-5"  # judges name matches; never ranks
# Every blind condition withholds the papers citing P (use_citers=False): the replay treats each paper as the
# unpublished manuscript it once was. "refs+model+published" is the published mode, which reads them.
CONDITIONS = {"refs": {"use_references": True, "topic_search": False, "propose": False, "use_citers": False},
              "topic": {"use_references": False, "topic_search": True, "propose": False, "use_citers": False},
              # the next version: references plus the model's own proposals, each confirmed in OpenAlex
              "refs+model": {"use_references": True, "topic_search": False, "propose": True, "use_citers": False},
              "refs+model+published": {"use_references": True, "topic_search": False, "propose": True, "use_citers": True},
              # Shasha's first user: a paper not yet written, so no reference list; topic search stands in for it
              "topic+model": {"use_references": False, "topic_search": True, "propose": True, "use_citers": False}}
LOCAL_BUDGET_CHARS = int((DEFAULT_NUM_CTX - MAX_LOCAL_OUTPUT - 256) * CHARS_PER_TOKEN)

DEDUPE_PROMPT = """\
The numbered entries are benchmark names that two readers extracted from one paper, each with \
the task the reader said it evaluates. Some entries are the same dataset under different \
spellings, versions or subsets; some are not benchmarks at all (a metric, a model, a method, \
an experiment name). Group the entries that denote the same dataset or benchmark. For each \
group give the entry numbers, a standard short name, and its kind: "public" for a named \
dataset, benchmark or environment other researchers can obtain and reuse (for example \
"ImageNet", "MuJoCo HalfCheetah", "TPC-H"); "own" for data the authors collected, built or \
simulated for this paper; "not_a_benchmark" otherwise. Every entry belongs to exactly one group."""

MATCH_PROMPT = """\
You score an experiment. A paper truly used the numbered benchmarks under "Used" (each with \
alternative spellings). A search tool, which never saw the paper's full text, proposed the \
numbered candidates. For each used benchmark give the number of the candidate that is the \
same benchmark (allow different spellings, versions or subsets of the same dataset), or -1 if \
none is. Return one entry per used benchmark, in order."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Group(_Strict):
    entries: list[int]
    name: str
    kind: str = Field(description='"public", "own" or "not_a_benchmark"')


class Groups(_Strict):
    groups: list[Group]


class Match(_Strict):
    used: int
    candidate: int = Field(description="Matching candidate number, or -1.")


class Matches(_Strict):
    matches: list[Match]


# ---------- helpers ----------

def safe(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9.-]+", "-", model)


def write_atomic(path: Path, data: bytes | str) -> None:
    """A kill mid-write never leaves a truncated cache file behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.part")
    (tmp.write_bytes if isinstance(data, bytes) else tmp.write_text)(data)
    os.replace(tmp, path)


def cached(path: Path, compute):
    if path.exists():
        return json.loads(path.read_text())
    data = compute()
    write_atomic(path, json.dumps(data, indent=1, ensure_ascii=False))
    return data


def paper_year(p: dict) -> int:
    return 2000 + int(p["arxiv"][:2])  # arXiv ids start with YYMM


def get_pdf(p: dict) -> bytes:
    path = PDFS / f"{p['arxiv']}.pdf"
    if not path.exists():
        PDFS.mkdir(exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.part")
        tmp.write_bytes(load_paper(p["arxiv"]).pdf)
        os.replace(tmp, path)  # atomic, so two processes never read a half-written file
        time.sleep(3)  # polite to arXiv
    return path.read_bytes()


def norm(text: str) -> str:
    text = text.replace("−", "-").replace("±", "+-").replace("±", "+-")
    return re.sub(r"\s+", "", text).lower()


def blind_input(p: dict) -> PaperAnalysis:
    return PaperAnalysis(title=p["title"], authors=[], proposed_method="", summary=p["abstract"],
                         benchmarks=[], caveats=[])


# ---------- phase A ----------

def run_extract(p: dict, model: str, client) -> dict:
    def compute():
        t0 = time.time()
        try:
            pdf = get_pdf(p)
            text = pdf_text(pdf)
        except Exception as e:  # noqa: BLE001  an oversized or unreadable PDF (2609.38721, 8 Nov 2026) must not stop the run
            return {"error": f"{type(e).__name__}: {e}", "seconds": round(time.time() - t0, 1)}
        paper = Paper(name=p["arxiv"], pdf=pdf, source=f"https://arxiv.org/abs/{p['arxiv']}")
        effort = "medium"
        try:
            try:
                r = analyze_paper(paper, client, model=model, effort=effort)
            except AnalysisError as e:
                if "token limit" not in str(e):
                    raise
                effort = "low"  # the answer overran the output cap; a lower effort writes a shorter one
                r = analyze_paper(paper, client, model=model, effort=effort)
        except Exception as e:  # keep the overnight run going
            return {"error": f"{type(e).__name__}: {e}", "seconds": round(time.time() - t0, 1)}
        return {"analysis": r.analysis.model_dump(), "model": r.model, "effort": effort, "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens, "seconds": round(time.time() - t0, 1),
                "text_chars": len(text), "cut": is_local_model(model) and len(text) > LOCAL_BUDGET_CHARS}
    return cached(OUT / "extract" / f"{p['id']}.{safe(model)}.json", compute)


def run_blind(p: dict, model: str, cond: str, client) -> dict:
    def compute():
        spend = Spend()
        t0 = time.time()
        real_find = scholar.find_paper
        if USER1_ARXIV_BIB and CONDITIONS[cond]["use_references"]:
            # Version 2, change 3, for user 1: the paper is still looked up and dated as before (no arXiv id is passed
            # to the pipeline, so the six-month cutoff is unchanged), but find_paper gets the arXiv id so that it reads
            # the paper's own arXiv bibliography; the lookup itself is then by arXiv DOI first, title second.
            scholar.find_paper = lambda title, arxiv_id=None, **kw: real_find(title, arxiv_id or p["arxiv"], **kw)
        try:
            c = consider_benchmarks(blind_input(p), client=client, model=model, effort="medium", prompts=Prompts(),
                                    spend=spend, budget_usd=1.0, up_to_year=paper_year(p), **CONDITIONS[cond])
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}", "seconds": round(time.time() - t0, 1)}
        finally:
            scholar.find_paper = real_find
        return {"consideration": c.model_dump(), "cost": spend.cost, "input_tokens": spend.input_tokens,
                "output_tokens": spend.output_tokens, "seconds": round(time.time() - t0, 1)}
    return cached(OUT / "blind" / f"{p['id']}.{safe(model)}.{cond}.json", compute)


# ---------- phase B ----------

def run_verbatim(p: dict, model: str, extract: dict) -> dict:
    """Was each copied value really in the paper? A value passes when every number in it (28.4,
    0.3, 1.5 of "1.5K") occurs as a number in the extracted text; formatting such as spaces,
    the ± sign or a trailing % is ignored, since PDF text extraction does not preserve it."""
    def compute():
        if "error" in extract:
            return {"error": extract["error"]}
        text_numbers = set(re.findall(r"\d+(?:\.\d+)?", pdf_text(get_pdf(p)).replace(",", "")))
        rows = []
        for b in extract["analysis"]["benchmarks"]:
            for r in b["results"]:
                nums = re.findall(r"\d+(?:\.\d+)?", r["value"].replace(",", ""))
                missing = [n for n in nums if n not in text_numbers]
                rows.append({"benchmark": b["name"], "method": r["method"], "value": r["value"],
                             "is_number": bool(nums), "found": bool(nums) and not missing, "missing": missing})
        numbers = [r for r in rows if r["is_number"]]
        return {"rows": rows, "n": len(rows), "numbers": len(numbers), "numbers_found": sum(r["found"] for r in numbers)}
    return cached(OUT / "verbatim" / f"{p['id']}.{safe(model)}.json", compute)


def run_truth(p: dict, extracts: dict[str, dict], judge) -> dict:
    path = OUT / "truth" / f"{p['id']}.json"
    if path.exists():
        return json.loads(path.read_text())

    def compute():
        entries = []
        for model, ex in extracts.items():
            for b in ex.get("analysis", {}).get("benchmarks", []):
                entries.append({"model": model, "name": b["name"], "task": b["task"]})
        if not entries:
            return {"benchmarks": [], "entries": entries, "checked_by_author": False}
        listing = "\n".join(f"[{i}] {e['name']} ({e['task']})" for i, e in enumerate(entries))
        answer = ask(judge, model=MATCH_MODEL, effort="low", system=DEDUPE_PROMPT,
                     content=[{"type": "text", "text": f"Title: {p['title']}\n\nEntries:\n{listing}"}],
                     schema=Groups, max_tokens=4000)
        text = pdf_text(get_pdf(p))
        low = text.lower()
        benchmarks = []
        for g in answer.value.groups:
            idx = [i for i in g.entries if 0 <= i < len(entries)]
            names = sorted({entries[i]["name"] for i in idx} | {g.name})
            snippet = None
            for n in names:
                k = low.find(n.lower())
                if k >= 0:
                    snippet = " ".join(text[max(0, k - 120): k + len(n) + 120].split())
                    break
            benchmarks.append({"name": g.name, "aliases": names, "kind": g.kind,
                               "found_by": sorted({entries[i]["model"] for i in idx}),
                               "in_text": snippet is not None, "snippet": snippet})
        return {"benchmarks": benchmarks, "entries": entries, "checked_by_author": False}
    return cached(path, compute)


def rank_pool(p: dict, pool_model: str, cond: str, ranker: str, blind: dict, client) -> dict:
    def compute():
        c = blind.get("consideration")
        if not c or not c.get("rank_input"):
            return {"error": "no pool"}
        t0 = time.time()
        try:
            answer = ask(client, model=ranker, effort="medium", system=Prompts().rank,
                         content=[{"type": "text", "text": c["rank_input"]}], schema=Ranking, max_tokens=8000)
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}
        names, seen, picks = c["candidates"], set(), []
        for pick in answer.value.picks:
            if 0 <= pick.candidate < len(names) and pick.candidate not in seen:
                seen.add(pick.candidate)
                picks.append(names[pick.candidate])
            if len(picks) == SUGGESTIONS:
                break
        return {"suggestions": picks, "candidates": names, "seconds": round(time.time() - t0, 1),
                "input_tokens": answer.input_tokens, "output_tokens": answer.output_tokens}
    return cached(OUT / "replay" / f"{p['id']}.{safe(pool_model)}.{cond}.{safe(ranker)}.json", compute)


_OVERRIDES = None


def override(arxiv: str, used_name: str, found):
    """Judge matches rejected by hand (match_overrides.json) score as no match."""
    global _OVERRIDES
    if _OVERRIDES is None:
        f = HERE / "match_overrides.json"
        _OVERRIDES = {(o["arxiv"], o["benchmark"], o["found_as"]) for o in json.loads(f.read_text())["overrides"]} if f.exists() else set()
    return None if found and (arxiv, used_name, found) in _OVERRIDES else found


def match_pool(p: dict, key: str, truth: dict, pool: list[str], judge) -> list:
    """Which candidate in this pool is each of the paper's public benchmarks (None if absent)?
    Judged once per pool, so every ranking replayed on the same pool is scored against the
    same matches."""
    used_names = [b["name"] for b in truth["benchmarks"] if b["kind"] == "public"]
    path = OUT / "match" / f"{p['id']}.{key}.json"
    if path.exists() and json.loads(path.read_text()).get("used") != used_names:
        path.unlink()  # the truth list changed (hand corrections): judge this pool again

    def compute():
        used = [b for b in truth["benchmarks"] if b["kind"] == "public"]
        if not used or not pool:
            return {"found_as": [None] * len(used), "pool_size": len(pool), "used": used_names}
        text = ("Used:\n" + "\n".join(f"[{i}] {b['name']} (also: {', '.join(b['aliases'])})" for i, b in enumerate(used))
                + "\n\nCandidates:\n" + "\n".join(f"[{i}] {n}" for i, n in enumerate(pool)))
        answer = ask(judge, model=MATCH_MODEL, effort="low", system=MATCH_PROMPT,
                     content=[{"type": "text", "text": text}], schema=Matches, max_tokens=4000)
        by_used = {m.used: m.candidate for m in answer.value.matches}
        found = []
        for i in range(len(used)):
            hit = by_used.get(i, -1)
            found.append(pool[hit] if 0 <= hit < len(pool) else None)
        return {"found_as": found, "pool_size": len(pool), "used": used_names}
    return [override(p["arxiv"], u, f) for u, f in zip(used_names, cached(path, compute)["found_as"])]


def score_rows(truth: dict, found_as: list, suggestions: list[str]) -> list[dict]:
    """One row per public benchmark the paper used: in the pool? in the top 10? at what rank?"""
    used = [b for b in truth["benchmarks"] if b["kind"] == "public"]
    rows = []
    for b, name in zip(used, found_as):
        rank = suggestions.index(name) + 1 if name in suggestions else None
        rows.append({"benchmark": b["name"], "in_pool": name is not None, "found_as": name,
                     "in_top10": rank is not None, "rank": rank})
    return rows


# ---------- commands ----------

def cmd_papers(args) -> None:
    raw = json.loads((HERE / "sample_raw.json").read_text())
    papers = [p for p in raw if p.get("arxiv") and p.get("abstract")][: args.n]
    PAPERS.write_text(json.dumps(papers, indent=1, ensure_ascii=False))
    print(f"{len(papers)} papers -> {PAPERS}")


def refuse_if_sealed() -> None:
    """Held-out runs are sealed from the end of C1 until C2 restores them (launch.py); running or scoring meanwhile would
    re-run them or expose them to the truth-check reviewer."""
    sealed = OUT / "sealed_pre_check"
    if (sealed / "blind").exists() and not (OUT / "blind").exists():
        sys.exit(f"{OUT.name}: the runs are sealed in {sealed.name} until launch.py C2 restores them")


def cmd_run(args) -> None:
    refuse_if_sealed()
    load_dotenv(HERE.parent / ".env")
    papers = json.loads(PAPERS.read_text())[: args.limit or None]
    for model in args.models.split(","):
        client = make_client(model)
        for i, p in enumerate(papers, 1):
            print(f"[{model}] {i}/{len(papers)} {p['arxiv']} {p['title'][:60]}", flush=True)
            ex = run_extract(p, model, client)
            print(f"   extract: {len(ex.get('analysis', {}).get('benchmarks', []))} benchmarks, {ex['seconds']}s"
                  + (f" ERROR {ex['error']}" if "error" in ex else ""), flush=True)
            for cond in args.conditions.split(","):
                b = run_blind(p, model, cond, client)
                n = len(b.get("consideration", {}).get("suggestions", []))
                print(f"   blind/{cond}: {n} suggestions, {b['seconds']}s" + (f" ERROR {b['error']}" if "error" in b else ""),
                      flush=True)


def cmd_score(args) -> None:
    refuse_if_sealed()
    global ablation
    import ablation  # noqa: PLC0415  (imports this module back, so not at top level)
    load_dotenv(HERE.parent / ".env")
    models = args.models.split(",")
    papers = json.loads(PAPERS.read_text())[: args.limit or None]
    judge = make_client(MATCH_MODEL)
    clients = {m: make_client(m) for m in models}
    results = []
    for p in papers:
        extracts = {m: run_extract(p, m, clients[m]) for m in models}
        truth = run_truth(p, extracts, judge)
        entry = {"id": p["id"], "arxiv": p["arxiv"], "title": p["title"], "topic": p.get("topic"),
                 "truth": truth, "extract": {}, "blind": {}, "replay": {}}
        public = [b for b in truth["benchmarks"] if b["kind"] == "public"]
        for m in models:
            ex = extracts[m]
            vb = run_verbatim(p, m, ex)
            entry["extract"][m] = {
                "error": ex.get("error"), "seconds": ex.get("seconds"), "cut": ex.get("cut", False),
                "input_tokens": ex.get("input_tokens"), "output_tokens": ex.get("output_tokens"),
                "benchmarks": len(ex.get("analysis", {}).get("benchmarks", [])),
                "public_found": sum(m in b["found_by"] for b in public), "public_total": len(public),
                "numbers": vb.get("numbers", 0), "numbers_found": vb.get("numbers_found", 0),
            }
            for cond in CONDITIONS:
                if not (OUT / "blind" / f"{p['id']}.{safe(m)}.{cond}.json").exists():
                    continue  # scoring never runs a condition; `run --conditions` does
                b = run_blind(p, m, cond, clients[m])
                c = b.get("consideration", {})
                sug = [s["name"] for s in c.get("suggestions", [])]
                found = match_pool(p, f"{safe(m)}.{cond}", truth, c.get("candidates", []), judge)
                ranker = c.get("ranker_picks") or sug  # runs before the reserved tier (9 Nov 2026) saved only the final 10
                score = score_rows(truth, found, sug)
                for row in score:
                    row["in_top10_ranker"] = row["found_as"] in ranker
                entry["blind"][f"{m}/{cond}"] = {"seconds": b.get("seconds"), "cost": b.get("cost"),
                                                 "stats": c.get("stats"), "suggestions": sug, "ranker_picks": ranker,
                                                 "tier": c.get("tier", []), "recent_tier": c.get("recent_tier", []), "recent_candidates": c.get("recent_candidates", []), "score": score}
                # Replay this pool with every ranker (including the same model, as a repeat). RRN_SKIP_REPLAY=1 (launch
                # config): no replays; they are not part of the protocol's tests or the approved plan.
                for ranker in ([] if SKIP_REPLAY else models):
                    rp = rank_pool(p, m, cond, ranker, b, clients[ranker])
                    if "error" in rp:
                        continue
                    entry["replay"][f"pool={m}/{cond} ranker={ranker}"] = {"suggestions": rp["suggestions"],
                                                                           "score": score_rows(truth, found, rp["suggestions"])}
        entry["ablation"] = ablation.score_ablation(p, models, truth, judge, match_pool, score_rows)
        results.append(entry)
        print(f"- {p['arxiv']} {p['title'][:50]}: {len(public)} public benchmarks", flush=True)
    (OUT / "results.json").write_text(json.dumps(results, indent=1, ensure_ascii=False))
    (OUT / "summary.md").write_text(summarize(results, models))
    print((OUT / "summary.md").read_text())


def cmd_redo(args) -> None:
    """Delete runs to be retried, and what was scored from them: blind runs that ended in an error (a Claude refusal or
    API error is cached as an error) or whose literature lookups or dating stopped (OpenAlex rate limits or quota), and
    PDF readings that ended in an error other than an unreadable PDF (which is permanent; the recipe swaps that paper)."""
    removed = 0
    for f in sorted((OUT / "extract").glob("*.json")):
        err = json.loads(f.read_text()).get("error")
        if err and not err.startswith("PaperError"):
            f.unlink(); removed += 1
            print(f"redo: extract {f.name}")
    for f in sorted((OUT / "blind").glob("*.json")):
        d = json.loads(f.read_text())
        notes = " ".join(d.get("consideration", {}).get("notes", []))
        if "error" not in d and "could not be fetched" not in notes and "failed" not in notes and "Dating stopped" not in notes:
            continue
        pid, rest = f.name.split(".", 1)
        cond = f.name.split(".")[-2]
        model = rest[: -(len(cond) + 6)]
        for g in ([f] + list((OUT / "replay").glob(f"{pid}.{model}.{cond}.*.json"))
                  + list((OUT / "match").glob(f"{pid}.{model}.{cond}.json"))):
            g.unlink()
            removed += 1
        print(f"redo: {f.name}")
    print(f"removed {removed} files")


def summarize(results: list[dict], models: list[str]) -> str:
    usable = [r for r in results if any(b["kind"] == "public" for b in r["truth"]["benchmarks"])]
    lines = [f"# Benchmark reader evaluation: {len(results)} papers, {len(usable)} with at least one public benchmark", ""]
    lines += ["## Reading the paper (R2)", "",
              "How many of a paper's benchmarks each model lists from the full PDF, and whether the numbers it copies are really in the paper. "
              "The truth lists were checked by hand; a benchmark missed by both models counts against both.", "",
              "| Model | Papers | Errors | Text cut | Public benchmarks found | Numbers copied verbatim | Mean seconds | Mean tokens in / out |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for m in models:
        ex = [r["extract"][m] for r in usable]
        ok = [e for e in ex if not e["error"]]
        found = sum(e["public_found"] for e in ok); total = sum(e["public_total"] for e in ok)
        nf = sum(e["numbers_found"] for e in ok); n = sum(e["numbers"] for e in ok)
        secs = sum(e["seconds"] for e in ok) / max(len(ok), 1)
        tin = sum(e["input_tokens"] or 0 for e in ok) / max(len(ok), 1); tout = sum(e["output_tokens"] or 0 for e in ok) / max(len(ok), 1)
        lines.append(f"| {m} | {len(ok)} | {len(ex) - len(ok)} | {sum(e['cut'] for e in ok)} | {found}/{total} ({100 * found / max(total, 1):.0f}%) "
                     f"| {nf}/{n} ({100 * nf / max(n, 1):.0f}%) | {secs:.0f} | {tin:,.0f} / {tout:,.0f} |")
    lines += ["", "## Benchmarks to consider (R3), blind: title and abstract only", "",
              "Given only the title and abstract, with the paper's own benchmarks hidden: how many of them the search brings into the "
              "candidate pool, and how many the ranker puts in its 10 suggestions. The mean rank is over the papers whose benchmarks "
              "were all in the top 10, where it is defined.", "",
              "In top 10 is the final list (reserved tier of model proposals, then the ranking); ranker only is the evidence ranking's own 10, the same thing for runs made before the tier.", "",
              "| Model | Condition | Papers | In pool | In top 10 | Ranker only, in top 10 | Mean rank (papers all found) | Mean seconds |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for m in models:
        for cond in CONDITIONS:
            rows = [row for r in usable for row in r["blind"].get(f"{m}/{cond}", {}).get("score", [])]
            n_papers = sum(1 for r in usable if r["blind"].get(f"{m}/{cond}", {}).get("score"))
            pool = sum(x["in_pool"] for x in rows); top = sum(x["in_top10"] for x in rows)
            top_r = sum(x.get("in_top10_ranker", x["in_top10"]) for x in rows)
            full = [rs for rs in (r["blind"].get(f"{m}/{cond}", {}).get("score", []) for r in usable)
                    if rs and all(x["in_top10"] for x in rs)]
            ranks = [x["rank"] for rs in full for x in rs]
            secs = [r["blind"][f"{m}/{cond}"]["seconds"] for r in usable if r["blind"].get(f"{m}/{cond}", {}).get("seconds")]
            mean_rank = f"{sum(ranks) / len(ranks):.1f} ({len(full)} papers)" if ranks else "-"
            lines.append(f"| {m} | {cond} | {n_papers} | {pool}/{len(rows)} ({100 * pool / max(len(rows), 1):.0f}%) "
                         f"| {top}/{len(rows)} ({100 * top / max(len(rows), 1):.0f}%) | {top_r}/{len(rows)} ({100 * top_r / max(len(rows), 1):.0f}%) | {mean_rank} | {sum(secs) / max(len(secs), 1):.0f} |")
    lines += ["", "## Same candidates, different ranker", "",
              "Both models rank the same candidate list, so a miss can be blamed on the search or on the ranker. "
              "Each model ranks the pool the other model's search produced, and its own as a repeat. Top-10 recall of the benchmarks that are in the pool.", "",
              "| Pool from | Condition | Ranker | Benchmarks in pool | In top 10 |", "| --- | --- | --- | --- | --- |"]
    for pm in models:
        for cond in CONDITIONS:
            for rk in models:
                key = f"pool={pm}/{cond} ranker={rk}"
                rows = [row for r in usable for row in r["replay"].get(key, {}).get("score", []) if row["in_pool"]]
                top = sum(x["in_top10"] for x in rows)
                lines.append(f"| {pm} | {cond} | {rk} | {len(rows)} | {top} ({100 * top / max(len(rows), 1):.0f}%) |")
    lines += ablation.summarize_ablation(usable, models)
    lines += ["", "## Papers", "", "The 30 sampled papers with their hand-checked benchmark lists. Papers with no public benchmark are read but not scored in the blind tests.", "", "| arXiv | Title | Topic | Public benchmarks | Own | Checked |", "| --- | --- | --- | --- | --- | --- |"]
    for r in results:
        pub = [b["name"] for b in r["truth"]["benchmarks"] if b["kind"] == "public"]
        own = sum(b["kind"] == "own" for b in r["truth"]["benchmarks"])
        lines.append(f"| {r['arxiv']} | {r['title'][:60]} | {r['topic']} | {', '.join(pub)} | {own} | {'yes' if r['truth'].get('checked_by_author') else 'no'} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("papers"); a.add_argument("--n", type=int, default=30)
    a = sub.add_parser("run"); a.add_argument("--models", required=True); a.add_argument("--limit", type=int, default=0)
    a.add_argument("--conditions", default="refs,topic,refs+model",
                   help="blind conditions to run (see CONDITIONS); the published mode is opt-in: refs+model+published")
    sub.add_parser("redo")
    a = sub.add_parser("score"); a.add_argument("--models", default="qwen2.5:14b,claude-haiku-5-5"); a.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    {"papers": cmd_papers, "run": cmd_run, "redo": cmd_redo, "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    main()
