"""Where do good suggestions come from? An ablation of the benchmarks-to-consider step, after
Shasha: with and without citation tracking, with and without the model's understanding.

    python experiments/ablation.py run --models claude-haiku-5-5
    python experiments/ablation.py run --models qwen2.5:14b
    python experiments/benchmark_eval.py score      # picks the results up; adds a section to summary.md

Every run is blind, like the blind runs in benchmark_eval: the paper's own benchmarks are hidden
and the score is how many of them come back. One file per run under experiments/out/abl/:

  catalog    no citations, no model: the title and the OpenAlex topic sent to the dataset catalogs,
             hits in the order returned (model-independent, stored under model "none")
  llm        no citations: the model names benchmarks from the title and abstract alone, no search
  back-freq  backward citation tracking, no model ranking: benchmarks named by the papers this paper
             cites, ranked by how many of them use each (the "refs" blind run's pool, re-ranked)
  fwd-freq   forward citation tracking, no model ranking: benchmarks named by the papers that cite this
             paper and that existed at least six months before it, ranked by how many use each
  fwd-llm    the same forward pool, ranked by the model
  union      the "refs" pool plus the model's own llm proposals as extra candidates (marked as the
             model's knowledge, unverified), ranked by the model

The "refs" blind run is the backward-tracking-with-model cell and "topic" the topic-search-with-model
cell. In every cell the model still reads the related papers' abstracts to pull out benchmark names;
"without the model" means it neither writes the queries nor ranks.

Forward tracking needs a date for each benchmark: the judge model names the paper that introduced it
and its year, OpenAlex gives that paper's year, and the earlier of the two is used. Dates are cached in
experiments/out/fwd/dates.json so each benchmark is dated once; `redate` re-applies the cutoff to the
forward pools after a change to the rule.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from pydantic import BaseModel, ConfigDict

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from benchmark_eval import HERE, MATCH_MODEL, OUT, PAPERS, cached, safe  # noqa: E402
from researchreadernerd import scholar  # noqa: E402
from researchreadernerd.analyze import Prompts  # noqa: E402
from researchreadernerd.catalogs import search_all  # noqa: E402
from researchreadernerd.cli import load_dotenv  # noqa: E402
from researchreadernerd.consider import PAPERS_PER_EXTRACTION_CALL, SUGGESTIONS, Extraction, Ranking, _key  # noqa: E402
from researchreadernerd.llm import ask, make_client  # noqa: E402

CITING_PAPERS = 60
MONTHS_BEFORE = 6
MODEL_CONDS = ["llm", "back-freq", "fwd-freq", "fwd-llm", "union"]

LLM_ONLY_PROMPT = """\
You advise the authors of the paper below on benchmarks to evaluate on. From its title and abstract \
alone, name up to 10 public benchmarks, datasets or evaluation suites that papers on this problem are \
expected to use, most important first, by their standard names. Choose only established, obtainable \
benchmarks; do not invent names."""

DATE_PROMPT = """\
For each numbered benchmark or dataset name, give the title of the paper or report that introduced it \
and the year it became available. If you do not know it, give an empty title and the year 0."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Names(_Strict):
    benchmarks: list[str]


class Dated(_Strict):
    item: int
    source_title: str
    year: int


class Dates(_Strict):
    items: list[Dated]


# ---------- helpers ----------

def abl_path(p: dict, model: str, cond: str) -> Path:
    return OUT / "abl" / f"{p['id']}.{safe(model)}.{cond}.json"


def dedupe(names: list[str]) -> list[str]:
    out, seen = [], set()
    for n in names:
        k = _key(n)
        if k and k not in seen:
            seen.add(k)
            out.append(n.strip())
    return out


def cutoff_year(p: dict) -> int:
    """A benchmark counts if it existed MONTHS_BEFORE months before the paper (arXiv ids start YYMM)."""
    year, month = 2000 + int(p["arxiv"][:2]), int(p["arxiv"][2:4]) - MONTHS_BEFORE
    return year - 1 if month <= 0 else year


# ---------- the conditions ----------

def run_catalog(p: dict) -> dict:
    def compute():
        t0 = time.time()
        queries = [p["title"]] + ([p["topic"]] if p.get("topic") else [])
        hits, notes = search_all(queries)
        names = dedupe([h.name for h in hits])
        return {"pool": names, "suggestions": names[:SUGGESTIONS], "queries": queries, "notes": notes,
                "seconds": round(time.time() - t0, 1)}
    return cached(abl_path(p, "none", "catalog"), compute)


def run_llm_only(p: dict, model: str, client) -> dict:
    def compute():
        t0 = time.time()
        text = f"Title: {p['title']}\nAbstract: {p['abstract']}"
        answer = ask(client, model=model, effort="medium", system=LLM_ONLY_PROMPT,
                     content=[{"type": "text", "text": text}], schema=Names, max_tokens=2000)
        names = dedupe(answer.value.benchmarks)[:SUGGESTIONS]
        return {"pool": names, "suggestions": names, "seconds": round(time.time() - t0, 1),
                "input_tokens": answer.input_tokens, "output_tokens": answer.output_tokens}
    return cached(abl_path(p, model, "llm"), compute)


_LINE = re.compile(r"^\[(\d+)\] (.*?) \| (.*)$")


def run_back_freq(p: dict, model: str) -> dict:
    """Re-rank the 'refs' blind pool by how many of the paper's references use each candidate
    (plus bibliography and topic counts); catalog hits, which have no count, keep their order."""
    def compute():
        path = OUT / "blind" / f"{p['id']}.{safe(model)}.refs.json"
        c = json.loads(path.read_text()).get("consideration", {}) if path.exists() else {}
        pool = c.get("candidates", [])
        if not c.get("rank_input") or not pool:
            return {"pool": pool, "suggestions": [], "note": "no refs pool"}
        counts = [0] * len(pool)
        for line in c["rank_input"].split("Candidates:\n", 1)[-1].splitlines():
            m = _LINE.match(line)
            if not m or not 0 <= int(m.group(1)) < len(pool):
                continue
            ev = m.group(3)
            n = sum(int(x) for x in re.findall(r"Used by (\d+) of the paper's references", ev))
            n += sum(int(x) for x in re.findall(r"is cited by (\d+) of the", ev))
            n += sum(int(x) for x in re.findall(r"Used by (\d+) of the most-cited papers", ev))
            counts[int(m.group(1))] = n
        order = sorted(range(len(pool)), key=lambda i: -counts[i])  # stable: ties keep the pool order
        return {"pool": pool, "suggestions": [pool[i] for i in order[:SUGGESTIONS]],
                "counts": {pool[i]: counts[i] for i in order if counts[i]}}
    return cached(abl_path(p, model, "back-freq"), compute)


def citing_papers(p: dict) -> dict:
    """OpenAlex: the most-cited papers that cite this one (model-independent). Not cached on failure."""
    def compute():
        found = scholar.find_paper(p["title"], p["arxiv"])
        if not found:
            return {"note": "not in OpenAlex", "works": []}
        works = scholar.citing_works(found[0].id, CITING_PAPERS)
        return {"openalex": found[0].id,
                "works": [{"id": w.id, "title": w.title, "abstract": w.abstract, "year": w.year, "url": w.url}
                          for w in works if w.title]}
    return cached(OUT / "fwd" / f"{p['id']}.citing.json", compute)


def date_benchmarks(names: list[str], judge) -> dict[str, int | None]:
    """Year each benchmark became available: OpenAlex's year for the source paper the judge names,
    else the judge's year, else None. Cached across papers and models."""
    path = OUT / "fwd" / "dates.json"
    dates = json.loads(path.read_text()) if path.exists() else {}
    todo = dedupe([n for n in names if _key(n) not in dates])
    try:
        for start in range(0, len(todo), 40):
            chunk = todo[start:start + 40]
            listing = "\n".join(f"[{i}] {n}" for i, n in enumerate(chunk))
            answer = ask(judge, model=MATCH_MODEL, effort="low", system=DATE_PROMPT,
                         content=[{"type": "text", "text": listing}], schema=Dates, max_tokens=6000)
            got = {d.item: d for d in answer.value.items}
            for i, n in enumerate(chunk):
                d = got.get(i)
                entry = {"name": n, "source_title": (d.source_title.strip() if d else ""),
                         "llm_year": (d.year if d else 0), "openalex_year": None, "openalex_title": None}
                if entry["source_title"]:
                    w = scholar.find_by_title(entry["source_title"])
                    if w:
                        entry["openalex_year"], entry["openalex_title"] = w.year, w.title
                dates[_key(n)] = entry
    finally:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dates, indent=1, ensure_ascii=False))
    return {n: resolve_year(dates[_key(n)]) for n in names if _key(n) in dates}


def resolve_year(entry: dict) -> int | None:
    """The earlier of OpenAlex's year and the judge's: OpenAlex sometimes returns a re-indexed copy
    of the source paper with a recent year (CUB-200-2011 came back as 2024)."""
    years = [y for y in (entry.get("openalex_year"), entry.get("llm_year")) if y]
    return min(years) if years else None


def run_forward(p: dict, model: str, client, judge) -> dict:
    """The forward pool: benchmarks named in the abstracts of papers citing this one, counted and
    dated, those newer than the cutoff set aside. Cached per model (the model reads the abstracts)."""
    def compute():
        works = citing_papers(p).get("works", [])
        cutoff = cutoff_year(p)
        base = {"cutoff_year": cutoff, "citing_papers": len(works), "pool": [], "candidates": [], "excluded": []}
        if not works:
            return {**base, "papers_read": 0, "seconds": 0, "note": "no citing papers in OpenAlex"}
        t0 = time.time()
        prompts = Prompts()
        mentions: dict[str, list[str]] = {}
        tin = tout = 0
        for start in range(0, len(works), PAPERS_PER_EXTRACTION_CALL):
            chunk = works[start:start + PAPERS_PER_EXTRACTION_CALL]
            text = "\n\n".join(f"[{i}] {w['title']}\n{(w['abstract'] or '')[:1200]}" for i, w in enumerate(chunk))
            answer = ask(client, model=model, effort="low", system=prompts.extract,
                         content=[{"type": "text", "text": text}], schema=Extraction, max_tokens=8000)
            tin += answer.input_tokens
            tout += answer.output_tokens
            for item in answer.value.papers:
                if 0 <= item.paper < len(chunk):
                    mentions[chunk[item.paper]["id"]] = item.benchmarks
        display: dict[str, Counter] = defaultdict(Counter)
        users: dict[str, set[str]] = defaultdict(set)
        for pid, names in mentions.items():
            for n in names:
                k = _key(n)
                if k:
                    display[k][n.strip()] += 1
                    users[k].add(pid)
        names = {k: display[k].most_common(1)[0][0] for k in display}
        years = date_benchmarks(list(names.values()), judge)
        candidates, excluded = [], []
        for k, name in sorted(names.items(), key=lambda kv: -len(users[kv[0]])):
            row = {"name": name, "users": len(users[k]), "year": years.get(name)}
            (excluded if row["year"] and row["year"] > cutoff else candidates).append(row)
        return {**base, "pool": [c["name"] for c in candidates], "candidates": candidates, "excluded": excluded,
                "papers_read": len(mentions), "seconds": round(time.time() - t0, 1),
                "input_tokens": tin, "output_tokens": tout}
    return cached(OUT / "fwd" / f"{p['id']}.{safe(model)}.json", compute)


def run_fwd_freq(p: dict, model: str, fwd: dict) -> dict:
    def compute():
        return {"pool": fwd["pool"], "suggestions": fwd["pool"][:SUGGESTIONS], "ranked_by": "number of citing papers using it"}
    return cached(abl_path(p, model, "fwd-freq"), compute)


def run_fwd_llm(p: dict, model: str, fwd: dict, client) -> dict:
    def compute():
        pool = fwd["pool"]
        if not pool:
            return {"pool": [], "suggestions": [], "note": "empty pool"}
        t0 = time.time()
        listing = "\n".join(f"[{i}] {c['name']} | Used by {c['users']} of the {fwd['citing_papers']} papers that cite this paper"
                            + (f" (introduced {c['year']})" if c["year"] else "") for i, c in enumerate(fwd["candidates"]))
        rank_input = (f"Title: {p['title']}\nSummary: {p['abstract']}\nProposed method: \nBenchmarks used:\n- none found\n\n"
                      f"Candidates:\n{listing}")
        answer = ask(client, model=model, effort="medium", system=Prompts().rank,
                     content=[{"type": "text", "text": rank_input}], schema=Ranking, max_tokens=8000)
        picks, seen = [], set()
        for pick in answer.value.picks:
            if 0 <= pick.candidate < len(pool) and pick.candidate not in seen:
                seen.add(pick.candidate)
                picks.append(pool[pick.candidate])
            if len(picks) == SUGGESTIONS:
                break
        return {"pool": pool, "suggestions": picks, "rank_input": rank_input, "seconds": round(time.time() - t0, 1),
                "input_tokens": answer.input_tokens, "output_tokens": answer.output_tokens}
    return cached(abl_path(p, model, "fwd-llm"), compute)


def run_union(p: dict, model: str, client) -> dict:
    """The refs pool with the model's own proposals added as candidates, ranked by the model."""
    def compute():
        refs_path = OUT / "blind" / f"{p['id']}.{safe(model)}.refs.json"
        llm_path = abl_path(p, model, "llm")
        c = json.loads(refs_path.read_text()).get("consideration", {}) if refs_path.exists() else {}
        proposals = json.loads(llm_path.read_text()).get("pool", []) if llm_path.exists() else []
        pool = list(c.get("candidates", []))
        lines = []
        if c.get("rank_input"):
            lines = [ln for ln in c["rank_input"].split("Candidates:\n", 1)[-1].splitlines() if _LINE.match(ln)]
        keys = {_key(n) for n in pool}
        for n in proposals:
            if _key(n) and _key(n) not in keys:
                keys.add(_key(n))
                lines.append(f"[{len(pool)}] {n} | Named by the model from the paper's title and abstract alone; not yet checked against the literature")
                pool.append(n)
        if not pool:
            return {"pool": [], "suggestions": [], "note": "empty pool"}
        t0 = time.time()
        rank_input = (f"Title: {p['title']}\nSummary: {p['abstract']}\nProposed method: \nBenchmarks used:\n- none found\n\n"
                      f"Candidates:\n" + "\n".join(lines))
        answer = ask(client, model=model, effort="medium", system=Prompts().rank,
                     content=[{"type": "text", "text": rank_input}], schema=Ranking, max_tokens=8000)
        picks, seen = [], set()
        for pick in answer.value.picks:
            if 0 <= pick.candidate < len(pool) and pick.candidate not in seen:
                seen.add(pick.candidate)
                picks.append(pool[pick.candidate])
            if len(picks) == SUGGESTIONS:
                break
        return {"pool": pool, "suggestions": picks, "proposals_added": len(pool) - len(c.get("candidates", [])),
                "rank_input": rank_input, "seconds": round(time.time() - t0, 1),
                "input_tokens": answer.input_tokens, "output_tokens": answer.output_tokens}
    return cached(abl_path(p, model, "union"), compute)


# ---------- scoring hooks, called from benchmark_eval.cmd_score / summarize ----------

def pool_key(model: str, cond: str) -> str:
    """Runs that share a candidate pool share one match file."""
    if cond == "catalog":
        return "none.catalog"
    if cond == "back-freq":
        return f"{safe(model)}.refs"
    if cond.startswith("fwd"):
        return f"{safe(model)}.fwd"
    return f"{safe(model)}.{cond}"


def score_ablation(p: dict, models: list[str], truth: dict, judge, match_pool, score_rows) -> dict:
    out = {}
    runs = [("none", "catalog")] + [(m, c) for m in models for c in MODEL_CONDS]
    for model, cond in runs:
        path = abl_path(p, model, cond)
        if not path.exists():
            continue
        d = json.loads(path.read_text())
        found = match_pool(p, pool_key(model, cond), truth, d.get("pool", []), judge)
        out[f"{model}/{cond}"] = {"suggestions": d.get("suggestions", []), "score": score_rows(truth, found, d.get("suggestions", []))}
    return out


CELLS = [  # (citation tracking, model understanding, condition, where the rows live)
    ("none", "none", "catalog", "ablation"),
    ("none", "yes", "llm", "ablation"),
    ("backward", "none", "back-freq", "ablation"),
    ("backward", "yes", "refs", "blind"),
    ("forward", "none", "fwd-freq", "ablation"),
    ("forward", "yes", "fwd-llm", "ablation"),
    ("topic search", "yes", "topic", "blind"),
    ("backward + model knowledge", "yes", "union", "ablation"),
]


def summarize_ablation(usable: list[dict], models: list[str]) -> list[str]:
    lines = ["", "## Where suggestions come from (ablation)", "",
             "The same blind recall for each source of candidates on its own, to see where good suggestions come from. "
             "Citation tracking: none, backward (papers this paper cites), "
             "forward (papers citing it, benchmarks that existed six months before it). Model understanding: whether the model "
             "writes the queries and ranks; in every cell it reads related abstracts to pull out names. "
             "Catalog-only uses no model at all.", "",
             "| Citation tracking | Model understanding | Condition | Model | Papers | In pool | In top 10 |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for tracking, understanding, cond, where in CELLS:
        for m in (["none"] if cond == "catalog" else models):
            key = f"{m}/{cond}"
            per_paper = [r.get(where, {}).get(key, {}).get("score", []) for r in usable]
            rows = [x for rs in per_paper for x in rs]
            n_papers = sum(1 for r in usable if key in r.get(where, {}))
            if not n_papers:
                continue
            pool = sum(x["in_pool"] for x in rows)
            top = sum(x["in_top10"] for x in rows)
            lines.append(f"| {tracking} | {understanding} | {cond} | {'-' if m == 'none' else m} | {n_papers} "
                         f"| {pool}/{len(rows)} ({100 * pool / max(len(rows), 1):.0f}%) | {top}/{len(rows)} ({100 * top / max(len(rows), 1):.0f}%) |")
    return lines


# ---------- command ----------

def cmd_run(args) -> None:
    load_dotenv(HERE.parent / ".env")
    papers = json.loads(PAPERS.read_text())[: args.limit or None]
    judge = make_client(MATCH_MODEL)
    if "catalog" in args.conds:
        for i, p in enumerate(papers, 1):
            try:
                c = run_catalog(p)
                print(f"[catalog] {i}/{len(papers)} {p['arxiv']}: {len(c['pool'])} hits, {c['seconds']}s", flush=True)
            except Exception as e:
                print(f"[catalog] {i}/{len(papers)} {p['arxiv']}: ERROR {type(e).__name__}: {e}", flush=True)
    for model in args.models.split(","):
        client = make_client(model)
        for i, p in enumerate(papers, 1):
            print(f"[{model}] {i}/{len(papers)} {p['arxiv']} {p['title'][:60]}", flush=True)
            for cond in args.conds:
                try:
                    if cond == "llm":
                        r = run_llm_only(p, model, client)
                        print(f"   llm: {len(r['suggestions'])} names, {r['seconds']}s", flush=True)
                    elif cond == "back-freq":
                        r = run_back_freq(p, model)
                        print(f"   back-freq: {len(r['pool'])} candidates, {len(r['suggestions'])} ranked", flush=True)
                    elif cond == "union":
                        r = run_union(p, model, client)
                        print(f"   union: {len(r['pool'])} candidates ({r.get('proposals_added', 0)} from the model), {r.get('seconds', 0)}s", flush=True)
                    elif cond == "fwd":
                        f = run_forward(p, model, client, judge)
                        run_fwd_freq(p, model, f)
                        r = run_fwd_llm(p, model, f, client)
                        print(f"   fwd: {f['citing_papers']} citing papers, {len(f['pool'])} candidates, "
                              f"{len(f['excluded'])} too new, {f['seconds']}s + rank {r.get('seconds', 0)}s", flush=True)
                except Exception as e:
                    print(f"   {cond}: ERROR {type(e).__name__}: {e}", flush=True)


def cmd_redate(args) -> None:
    """Re-apply the cutoff to every forward pool from the cached dates; drop the dependent runs when a pool changes."""
    path = OUT / "fwd" / "dates.json"
    dates = json.loads(path.read_text()) if path.exists() else {}
    papers = {p["id"]: p for p in json.loads(PAPERS.read_text())}
    changed = 0
    for f in sorted((OUT / "fwd").glob("*.json")):
        if f.name == "dates.json" or f.name.endswith(".citing.json"):
            continue
        d = json.loads(f.read_text())
        pid, model = f.name.split(".", 1)[0], f.stem.split(".", 1)[1]
        rows = sorted(d["candidates"] + d["excluded"], key=lambda r: -r["users"])
        cutoff = cutoff_year(papers[pid])
        for r in rows:
            r["year"] = resolve_year(dates.get(_key(r["name"]), {}))
        candidates = [r for r in rows if not (r["year"] and r["year"] > cutoff)]
        excluded = [r for r in rows if r["year"] and r["year"] > cutoff]
        pool = [r["name"] for r in candidates]
        if pool == d["pool"]:
            continue
        changed += 1
        f.write_text(json.dumps({**d, "pool": pool, "candidates": candidates, "excluded": excluded}, indent=1, ensure_ascii=False))
        for g in [OUT / "abl" / f"{pid}.{model}.fwd-freq.json", OUT / "abl" / f"{pid}.{model}.fwd-llm.json",
                  OUT / "match" / f"{pid}.{model}.fwd.json"]:
            g.unlink(missing_ok=True)
        print(f"redate: {f.name}: {len(d['pool'])} -> {len(pool)} candidates")
    print(f"{changed} pools changed; rerun `run --conds fwd` to rebuild their rankings")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("redate")
    a = sub.add_parser("run")
    a.add_argument("--models", required=True)
    a.add_argument("--conds", default="catalog,llm,back-freq,fwd,union", help="comma-separated subset of catalog,llm,back-freq,fwd,union")
    a.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    if args.cmd == "redate":
        cmd_redate(args)
        return
    args.conds = args.conds.split(",")
    cmd_run(args)


if __name__ == "__main__":
    main()
