"""Leave-one-out test (Shasha's user 1', 8 Nov 2026).

For each public benchmark B that paper P used: remove B from the paper as the tool read it (the
benchmark entry, every mention of its names in the summary and method, and its dataset paper from
the reference list we hand to OpenAlex), run the tool in manuscript mode on what is left, and ask
whether B comes back among the 10 suggestions, or at least in the candidate pool. The figure of
merit is the share of the 117 benchmarks recovered. Citing papers stay withheld (the paper is
replayed as the unpublished manuscript it was); co-usage and topic lookups are dated to the
paper's year as in the other replays.

What the removal cannot do: take B out of the model's memory. These are 2018 to 2021 papers the
model has read, so a recovered benchmark may be recalled rather than inferred. The report splits
recoveries by where the tool says it found them (references, co-usage, bibliographies, the model's
own proposal) so that part can be seen.

OpenAlex responses are cached on disk for this harness only (out/oa_cache), since each paper is
rerun once per benchmark with an almost identical reference list.

  python leave_one_out.py run [--limit N] [--papers W1,W2] [--dry-run]
  python leave_one_out.py score
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_eval import HERE, MATCH_MODEL, MATCH_PROMPT, OUT, PAPERS, Matches, cached, override, paper_year, safe, write_atomic  # noqa: E402
from researchreadernerd import scholar  # noqa: E402
from researchreadernerd.analyze import Prompts  # noqa: E402
from researchreadernerd.cli import load_dotenv  # noqa: E402
from researchreadernerd.consider import consider_benchmarks  # noqa: E402
from researchreadernerd.llm import ask, make_client  # noqa: E402
from researchreadernerd.pricing import Spend  # noqa: E402
from researchreadernerd.schema import PaperAnalysis  # noqa: E402

MODEL = "claude-haiku-5-5"
LOO = OUT / "loo"
LOO_MATCH = OUT / "loo_match"
CACHE = OUT / "oa_cache"
STOP = 0.95  # quota.py fraction at which runs stop (5 cents left)
SEED = 7
PLACEHOLDER = "[withheld]"
STATS = Counter()


# ---------- OpenAlex response cache (harness only) ----------

_real_get = scholar._get


def _cached_get(path: str, params: dict | None = None, *, retries: int = 6) -> dict:
    shown = {k: v for k, v in (params or {}).items() if k not in ("api_key", "mailto")}
    key = hashlib.sha1(json.dumps([path, shown], sort_keys=True, default=str).encode()).hexdigest()
    f = CACHE / f"{key}.json"
    if f.exists():
        STATS["cache_hits"] += 1
        return json.loads(f.read_text())
    data = _real_get(path, params, retries=retries)
    STATS["openalex_calls"] += 1
    STATS["openalex_usd"] += 0.001 if "search" in shown else 0.0001  # OpenAlex prices: search $0.001, list $0.0001
    write_atomic(f, json.dumps(data))
    return data


scholar._get = _cached_get

S2_CACHE = OUT / "s2_cache"  # version 2: Semantic Scholar responses, cached the same way (a paper is read once per benchmark)
_real_s2_get = scholar._s2_get


def _cached_s2_get(path: str, params: dict | None = None, *, retries: int = 6) -> dict:
    key = hashlib.sha1(json.dumps([path, params or {}], sort_keys=True, default=str).encode()).hexdigest()
    f = S2_CACHE / f"{key}.json"
    if f.exists():
        STATS["s2_cache_hits"] += 1
        return json.loads(f.read_text())
    data = _real_s2_get(path, params, retries=retries)
    STATS["s2_calls"] += 1
    S2_CACHE.mkdir(parents=True, exist_ok=True)
    write_atomic(f, json.dumps(data))
    return data


scholar._s2_get = _cached_s2_get


# ---------- withholding the benchmark's dataset paper from the reference list ----------

WITHHOLD: dict = {"titles": [], "aliases": [], "dropped": []}
_real_find_paper = scholar.find_paper


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def _alias_patterns(aliases: list[str]) -> list[re.Pattern]:
    pats = []
    for a in sorted(set(aliases), key=len, reverse=True):
        if len(a.strip()) < 3:
            continue
        pats.append(re.compile(r"(?<![A-Za-z0-9])" + re.escape(a.strip()) + r"(?![A-Za-z0-9])", re.I))
    return pats


def _title_is_source(title: str) -> bool:
    t = _norm(title)
    if not t:
        return False
    if any(t == _norm(src) for src in WITHHOLD["titles"]):
        return True
    # a reference whose title names the benchmark (five letters or more, as a whole word) is its dataset paper
    # or a paper about it; either way the authors' pointer to the benchmark goes with the benchmark
    return any(len(a.strip()) >= 5 and p.search(title or "")
               for a in WITHHOLD["alias_names"] for p in _alias_patterns([a]))


def _find_paper_withholding(title: str, arxiv_id: str | None = None):
    found = _real_find_paper(title, arxiv_id)
    if found is None or not (WITHHOLD["titles"] or WITHHOLD["aliases"]):
        return found
    work, ref_ids = found
    works = {w.id: w for w in scholar.get_works(list(ref_ids))} if ref_ids else {}
    keep = []
    for rid in ref_ids:
        w = works.get(rid)
        if w is not None and _title_is_source(w.title):
            WITHHOLD["dropped"].append({"id": rid, "title": w.title, "year": w.year})
            continue
        keep.append(rid)
    return work, keep


scholar.find_paper = _find_paper_withholding


# ---------- removing the benchmark from the paper as read ----------

def scrub(obj, pats: list[re.Pattern], counter: Counter):
    if isinstance(obj, str):
        out = obj
        for p in pats:
            out, n = p.subn(PLACEHOLDER, out)
            counter["mentions"] += n
        return out
    if isinstance(obj, list):
        return [scrub(x, pats, counter) for x in obj]
    if isinstance(obj, dict):
        return {k: scrub(v, pats, counter) for k, v in obj.items()}
    return obj


def names_match(name: str, pats: list[re.Pattern], aliases: list[str]) -> bool:
    n = _norm(name)
    return any(n == _norm(a) for a in aliases) or any(p.search(name) for p in pats)


def remove_benchmark(analysis: dict, bench: dict) -> tuple[dict, dict]:
    """The analysis without benchmark `bench`: its entries dropped, its names blanked everywhere else."""
    aliases = list(dict.fromkeys([bench["name"], *bench.get("aliases", [])]))
    pats = _alias_patterns(aliases)
    info: dict = {"aliases": aliases, "dropped_entries": [], "mentions": 0}
    kept = []
    for b in analysis["benchmarks"]:
        if names_match(b["name"], pats, aliases):
            info["dropped_entries"].append(b["name"])
        else:
            kept.append(b)
    counter = Counter()
    rest = scrub({k: v for k, v in analysis.items() if k != "benchmarks"}, pats, counter)
    rest["benchmarks"] = scrub(kept, pats, counter)
    info["mentions"] = counter["mentions"]
    return rest, info


# ---------- the runs ----------

def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:48] or "x"


def quota() -> float:
    try:
        out = subprocess.run([sys.executable, "quota.py"], cwd=HERE, capture_output=True, text=True, timeout=60)
        return float(out.stdout.strip())
    except Exception:  # noqa: BLE001
        return -1.0


def load_truth(p: dict) -> dict:
    path = OUT / "truth" / f"{p['id']}.json"
    return json.loads(path.read_text()) if path.exists() else {"benchmarks": []}


def source_titles_for(p: dict, bench: dict) -> list[str]:
    """The dataset paper the tool found for this benchmark in the cached full-paper run (used_by.source)."""
    titles = []
    aliases = [bench["name"], *bench.get("aliases", [])]
    pats = _alias_patterns(aliases)
    for suffix in (".model", "", ".model.published"):
        path = OUT / "full" / f"{p['id']}.{safe(MODEL)}{suffix}.json"
        if not path.exists():
            continue
        c = json.loads(path.read_text()).get("consideration", {})
        for u in c.get("used_by", []):
            if u.get("source") and names_match(u["benchmark"], pats, aliases):
                titles.append(u["source"]["title"])
    return list(dict.fromkeys(titles))


def run_one(p: dict, bench: dict, analysis: dict, client, dry: bool = False) -> dict:
    path = LOO / f"{p['id']}.{safe(MODEL)}.{slug(bench['name'])}.json"
    if path.exists() and not dry:
        return json.loads(path.read_text())
    stripped, info = remove_benchmark(analysis, bench)
    info["read_by_tool"] = bool(info["dropped_entries"])
    info["source_titles"] = source_titles_for(p, bench)
    WITHHOLD.update({"titles": info["source_titles"], "alias_names": info["aliases"],
                     "aliases": _alias_patterns(info["aliases"]), "dropped": []})
    if dry:
        found = scholar.find_paper(stripped["title"], p["arxiv"])
        info["dropped_refs"] = WITHHOLD["dropped"]
        info["refs_left"] = len(found[1]) if found else None
        info["stripped_summary"] = stripped["summary"]
        info["benchmarks_left"] = [b["name"] for b in stripped["benchmarks"]]
        return info
    spend = Spend()
    t0 = time.time()
    try:
        c = consider_benchmarks(PaperAnalysis(**stripped), client=client, model=MODEL, effort="medium",
                                prompts=Prompts(), spend=spend, budget_usd=1.0, arxiv_id=p["arxiv"],
                                up_to_year=paper_year(p), use_references=True, topic_search=False, propose=True,
                                use_citers=False)
    except Exception as e:  # noqa: BLE001
        data = {"error": f"{type(e).__name__}: {e}", "seconds": round(time.time() - t0, 1), "removed": info,
                "dropped_refs": WITHHOLD["dropped"]}
        WITHHOLD.update({"titles": [], "aliases": [], "dropped": []})
        return data  # not cached: an error is retried next time
    data = {"consideration": c.model_dump(), "cost": spend.cost, "input_tokens": spend.input_tokens,
            "output_tokens": spend.output_tokens, "seconds": round(time.time() - t0, 1), "removed": info,
            "dropped_refs": WITHHOLD["dropped"]}
    WITHHOLD.update({"titles": [], "aliases": [], "dropped": []})
    write_atomic(path, json.dumps(data, indent=1, ensure_ascii=False))
    return data


def match_one(p: dict, bench: dict, pool: list[str], judge) -> str | None:
    """Which candidate in the pool is this benchmark, if any (the same judge and prompt as the other tests)."""
    path = LOO_MATCH / f"{p['id']}.{safe(MODEL)}.{slug(bench['name'])}.json"

    def compute():
        if not pool:
            return {"found_as": None, "pool_size": 0}
        text = (f"Used:\n[0] {bench['name']} (also: {', '.join(bench.get('aliases', []))})"
                + "\n\nCandidates:\n" + "\n".join(f"[{i}] {n}" for i, n in enumerate(pool)))
        answer = ask(judge, model=MATCH_MODEL, effort="low", system=MATCH_PROMPT,
                     content=[{"type": "text", "text": text}], schema=Matches, max_tokens=2000)
        hit = next((m.candidate for m in answer.value.matches if m.used == 0), -1)
        return {"found_as": pool[hit] if 0 <= hit < len(pool) else None, "pool_size": len(pool)}
    return override(p["arxiv"], bench["name"], cached(path, compute)["found_as"])


def jobs(papers: list[dict]) -> list[tuple[dict, dict, dict]]:
    out = []
    for p in papers:
        ex_path = OUT / "extract" / f"{p['id']}.{safe(MODEL)}.json"
        if not ex_path.exists():
            continue
        ex = json.loads(ex_path.read_text())
        if "error" in ex:
            continue
        truth = load_truth(p)
        for b in truth["benchmarks"]:
            if b.get("kind") == "public":
                out.append((p, b, ex["analysis"]))
    return out


def cmd_run(args) -> None:
    load_dotenv(HERE.parent / ".env")
    papers = json.loads(PAPERS.read_text())
    if args.papers:
        want = set(args.papers.split(","))
        papers = [p for p in papers if p["id"] in want or p["arxiv"] in want]
    todo = jobs(papers)[: args.limit or None]
    client = None if args.dry_run else make_client(MODEL)
    print(f"{len(todo)} (paper, benchmark) runs", flush=True)
    for i, (p, b, analysis) in enumerate(todo, 1):
        path = LOO / f"{p['id']}.{safe(MODEL)}.{slug(b['name'])}.json"
        if path.exists() and not args.dry_run:
            continue
        if not args.dry_run:
            q = quota()
            if q >= STOP:
                print(f"stopping: OpenAlex quota at {q:.4f}", flush=True)
                sys.exit(2)
        r = run_one(p, b, analysis, client, dry=args.dry_run)
        if args.dry_run:
            print(f"[{i}] {p['arxiv']} minus {b['name']!r}: entries dropped {r['dropped_entries']}, "
                  f"{r['mentions']} mentions blanked, refs dropped {[d['title'][:60] for d in r['dropped_refs']]}, "
                  f"{r['refs_left']} refs left; source titles {r['source_titles']}", flush=True)
            continue
        c = r.get("consideration", {})
        sug = [s["name"] for s in c.get("suggestions", [])]
        print(f"[{i}/{len(todo)}] {p['arxiv']} minus {b['name']!r}: {len(c.get('candidates', []))} candidates, "
              f"{len(sug)} suggestions, {len(r.get('dropped_refs', []))} refs withheld, {r['seconds']}s"
              + (f" ERROR {r['error']}" if "error" in r else ""), flush=True)
    print(f"openalex calls {STATS['openalex_calls']}, cache hits {STATS['cache_hits']}", flush=True)


# ---------- scoring ----------

def wilson(k: int, n: int, z: float = 1.645) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def bootstrap_papers(pairs: list[tuple[int, int]], rounds: int = 10000, level: float = 0.90) -> tuple[float, float]:
    rng = random.Random(SEED)
    vals = []
    for _ in range(rounds):
        sample = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        k, n = sum(a for a, _ in sample), sum(b for _, b in sample)
        vals.append(k / n if n else 0.0)
    vals.sort()
    tail = (1 - level) / 2
    return (vals[int(tail * rounds)], vals[int((1 - tail) * rounds) - 1])


def pc(k: int, n: int) -> str:
    return f"{k}/{n} ({100 * k / n:.0f}%)" if n else "-"


def ci(k: int, n: int, pairs: list[tuple[int, int]]) -> str:
    lo, hi = wilson(k, n)
    blo, bhi = bootstrap_papers(pairs) if pairs else (0.0, 0.0)
    return f"{100 * blo:.0f}-{100 * bhi:.0f}% papers, {100 * lo:.0f}-{100 * hi:.0f}% benchmarks"


def cited_counts() -> dict[tuple[str, str], tuple[int, int]]:
    """How many cited and citing papers used each (paper, benchmark), from the shared tables."""
    path = OUT / "shasha_tables.json"
    if not path.exists():
        return {}
    t = json.loads(path.read_text())
    out = {}
    for p in t.get("papers", []):
        for o in p.get("own", []):
            out[(p["arxiv"], _norm(o["benchmark"]))] = (o.get("cited", 0), o.get("citing", 0))
    return out


def cmd_score(args) -> None:
    load_dotenv(HERE.parent / ".env")
    judge = make_client(MATCH_MODEL)
    papers = json.loads(PAPERS.read_text())
    counts = cited_counts()
    rows = []
    for p, b, analysis in jobs(papers):
        path = LOO / f"{p['id']}.{safe(MODEL)}.{slug(b['name'])}.json"
        if not path.exists():
            continue
        r = json.loads(path.read_text())
        c = r.get("consideration", {})
        pool = c.get("candidates", [])
        sug = [s["name"] for s in c.get("suggestions", [])]
        ranker = c.get("ranker_picks") or sug  # runs before the reserved tier saved only the final 10
        found_as = match_one(p, b, pool, judge)
        rank = sug.index(found_as) + 1 if found_as in sug else None
        where = next((s["found_by"] for s in c.get("suggestions", []) if s["name"] == found_as), []) if rank else []
        used_section = c.get("rank_input", "").split("Candidates:")[0]
        leak = any(pat.search(used_section) for pat in _alias_patterns(r["removed"]["aliases"]))
        cc = counts.get((p["arxiv"], _norm(b["name"])), (None, None))
        rows.append({"arxiv": p["arxiv"], "title": p["title"], "benchmark": b["name"], "in_pool": found_as is not None,
                     "found_as": found_as, "in_top10": rank is not None, "rank": rank, "found_by": where,
                     "in_top10_ranker": found_as is not None and found_as in ranker, "tier": c.get("tier", []), "recent_tier": c.get("recent_tier", []), "recent_candidates": c.get("recent_candidates", []),
                     "read_by_tool": r["removed"]["read_by_tool"], "mentions_blanked": r["removed"]["mentions"],
                     "refs_withheld": [d["title"] for d in r.get("dropped_refs", [])],
                     "cited_users": cc[0], "citing_users": cc[1], "leak_in_used_list": leak,
                     "candidates": len(pool), "suggestions": len(sug), "seconds": r.get("seconds")})
    write_atomic(OUT / "loo_results.json", json.dumps(rows, indent=1, ensure_ascii=False))

    def block(title: str, sel: list[dict]) -> list[str]:
        n = len(sel)
        k10 = sum(r["in_top10"] for r in sel)
        kr = sum(r.get("in_top10_ranker", r["in_top10"]) for r in sel)
        kp = sum(r["in_pool"] for r in sel)
        by_paper: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
        for r in sel:
            by_paper[r["arxiv"]][0] += r["in_top10"]
            by_paper[r["arxiv"]][1] += r["in_pool"]
            by_paper[r["arxiv"]][2] += 1
        pairs10 = [(v[0], v[2]) for v in by_paper.values()]
        pairsp = [(v[1], v[2]) for v in by_paper.values()]
        return [f"| {title} | {len(by_paper)} | {pc(k10, n)} | {ci(k10, n, pairs10)} | {pc(kr, n)} | {pc(kp, n)} | {ci(kp, n, pairsp)} |"]

    lines = ["# Leave-one-out test (user 1')", "",
             f"Each of the {len(rows)} public benchmarks was removed in turn from its paper as the tool had read it: the "
             "benchmark entry, every mention of its names in the summary and method, and any reference whose title is "
             "its dataset paper or names it. The tool then ran in manuscript mode (references read, model proposals on, "
             "citing papers withheld, lookups dated to the paper's year). Recovered = the removed benchmark is among the "
             f"10 suggestions; in pool = it is among the candidates the ranker saw. Model: {MODEL}. Intervals are 90%: "
             "a percentile bootstrap over papers (10,000 rounds, seed 7) and Wilson over benchmarks.", "",
             "Recovered in 10 is the final list (reserved tier of model proposals, then the ranking); ranker only is the evidence ranking's own 10.", "",
             "| Benchmarks | Papers | Recovered in 10 | 90% CI | Ranker only | In pool | 90% CI |", "| --- | --- | --- | --- | --- | --- | --- |"]
    lines += block("All", rows)
    lines += block("Read by the tool from the PDF (so removal changed the input)", [r for r in rows if r["read_by_tool"]])
    lines += block("Missed by the tool's reading (only the reference was withheld)", [r for r in rows if not r["read_by_tool"]])
    lines += block("Used by 1+ cited paper", [r for r in rows if (r["cited_users"] or 0) >= 1])
    lines += block("Used by no cited paper", [r for r in rows if r["cited_users"] == 0])
    lines += block("Used by 2+ citing papers", [r for r in rows if (r["citing_users"] or 0) >= 2])
    src = Counter()
    for r in rows:
        if r["in_top10"]:
            for s in r["found_by"] or ["unknown"]:
                src[s] += 1
    lines += ["", "## Where the recovered benchmarks came from", "",
              "A suggestion can carry more than one source, so the counts can exceed the recoveries. 'model knowledge' is "
              "the model's own proposal confirmed in OpenAlex: the part that may be recall rather than inference.", "",
              "| Source | Recovered suggestions |", "| --- | --- |"]
    lines += [f"| {k} | {v} |" for k, v in src.most_common()]
    leaks = [r for r in rows if r["leak_in_used_list"]]
    withheld = sum(1 for r in rows if r["refs_withheld"])
    lines += ["", "## Checks", "",
              f"- Runs where a reference was withheld: {withheld} of {len(rows)}; the rest had no reference whose title "
              "matched the benchmark or its dataset paper.",
              f"- Runs where a name of the removed benchmark still appeared in the 'benchmarks used' list the ranker saw: "
              f"{len(leaks)}" + (" (" + ", ".join(f"{r['arxiv']} {r['benchmark']}" for r in leaks[:10]) + ")" if leaks else ""),
              f"- Mean candidates per run: {sum(r['candidates'] for r in rows) / max(len(rows), 1):.0f}; mean seconds: "
              f"{sum(r['seconds'] or 0 for r in rows) / max(len(rows), 1):.0f}."]
    lines += ["", "## Per paper", "", "| arXiv | Title | Benchmarks | Recovered in 10 | In pool |", "| --- | --- | --- | --- | --- |"]
    by_paper: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_paper[r["arxiv"]].append(r)
    for arxiv, sel in by_paper.items():
        lines.append(f"| {arxiv} | {sel[0]['title'][:60]} | {len(sel)} | {sum(r['in_top10'] for r in sel)} | {sum(r['in_pool'] for r in sel)} |")
    lines += ["", "## Every benchmark", "", "| arXiv | Benchmark | Rank | Found as | Source | Cited users | Refs withheld |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        lines.append(f"| {r['arxiv']} | {r['benchmark']} | {r['rank'] or ('pool' if r['in_pool'] else '-')} | "
                     f"{r['found_as'] or ''} | {', '.join(r['found_by'])} | {r['cited_users'] if r['cited_users'] is not None else ''} | "
                     f"{len(r['refs_withheld'])} |")
    (OUT / "loo_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:16]))
    print(f"\n-> {OUT / 'loo_summary.md'}, {OUT / 'loo_results.json'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--papers", default="", help="comma-separated OpenAlex ids or arXiv ids")
    r.add_argument("--dry-run", action="store_true", help="show what would be removed; no model calls")
    r.set_defaults(func=cmd_run)
    s = sub.add_parser("score")
    s.set_defaults(func=cmd_score)
    args = ap.parse_args()
    for d in (LOO, LOO_MATCH, CACHE):
        d.mkdir(parents=True, exist_ok=True)
    args.func(args)


if __name__ == "__main__":
    main()
