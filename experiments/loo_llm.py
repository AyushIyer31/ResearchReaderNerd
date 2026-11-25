"""Haiku-alone baseline for the leave-one-out test (user 1'), 9 Nov 2026.

For each (paper, withheld benchmark) run in out/loo, give the model the same paper description the tool's
ranker saw (title, summary, method, the remaining 'benchmarks used' list, field, themes; the withheld
benchmark already scrubbed) and ask for up to 10 public benchmarks the paper would be expected to evaluate on
beyond those listed. No literature search. Recovered = the withheld benchmark is among the 10, judged by the
same name matcher as everywhere else.

  RRN_OUT=out RRN_PAPERS=papers.json python loo_llm.py
  RRN_OUT=out_2026 RRN_PAPERS=papers_2026.json python loo_llm.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ablation import Names, dedupe  # noqa: E402
from benchmark_eval import HERE, MATCH_MODEL, MATCH_PROMPT, OUT, PAPERS, Matches, cached, override, safe, write_atomic  # noqa: E402
from leave_one_out import LOO, MODEL, bootstrap_papers, jobs, pc, slug, wilson  # noqa: E402
from researchreadernerd.cli import load_dotenv  # noqa: E402
from researchreadernerd.llm import ask, make_client  # noqa: E402

LLM_DIR = OUT / "loo_llm"
LLM_MATCH = OUT / "loo_llm_match"
PROMPT = """\
You advise the authors of the paper described below on benchmarks to evaluate on. The description lists the \
benchmarks the paper already uses. From the description alone, name up to 10 further public benchmarks, datasets \
or evaluation suites that a paper on this problem is expected to use, most important first, by their standard \
names. Do not repeat the ones listed. Choose only established, obtainable benchmarks; do not invent names."""


def run_llm(p: dict, bench: dict, used_section: str, client) -> dict:
    def compute():
        t0 = time.time()
        a = ask(client, model=MODEL, effort="medium", system=PROMPT,
                content=[{"type": "text", "text": used_section.strip()}], schema=Names, max_tokens=2000)
        return {"suggestions": dedupe(a.value.benchmarks)[:10], "seconds": round(time.time() - t0, 1),
                "input_tokens": a.input_tokens, "output_tokens": a.output_tokens}
    return cached(LLM_DIR / f"{p['id']}.{safe(MODEL)}.{slug(bench['name'])}.json", compute)


def match_llm(p: dict, bench: dict, pool: list[str], judge) -> str | None:
    def compute():
        if not pool:
            return {"found_as": None}
        text = (f"Used:\n[0] {bench['name']} (also: {', '.join(bench.get('aliases', []))})"
                + "\n\nCandidates:\n" + "\n".join(f"[{i}] {n}" for i, n in enumerate(pool)))
        a = ask(judge, model=MATCH_MODEL, effort="low", system=MATCH_PROMPT,
                content=[{"type": "text", "text": text}], schema=Matches, max_tokens=2000)
        hit = next((m.candidate for m in a.value.matches if m.used == 0), -1)
        return {"found_as": pool[hit] if 0 <= hit < len(pool) else None}
    return override(p["arxiv"], bench["name"], cached(LLM_MATCH / f"{p['id']}.{safe(MODEL)}.{slug(bench['name'])}.json", compute)["found_as"])


def main() -> None:
    load_dotenv(HERE.parent / ".env")
    client, judge = make_client(MODEL), make_client(MATCH_MODEL)
    papers = json.loads(PAPERS.read_text())
    tool = {(r["arxiv"], r["benchmark"]): r for r in json.loads((OUT / "loo_results.json").read_text())}
    rows = []
    todo = jobs(papers)
    for i, (p, b, _analysis) in enumerate(todo, 1):
        path = LOO / f"{p['id']}.{safe(MODEL)}.{slug(b['name'])}.json"
        if not path.exists():
            continue
        c = json.loads(path.read_text())["consideration"]
        used_section = c["rank_input"].split("Candidates:")[0]
        r = run_llm(p, b, used_section, client)
        found = match_llm(p, b, r["suggestions"], judge)
        t = tool.get((p["arxiv"], b["name"]), {})
        rows.append({"arxiv": p["arxiv"], "title": p["title"], "benchmark": b["name"], "llm_found_as": found,
                     "llm_in_top10": found is not None, "llm_rank": r["suggestions"].index(found) + 1 if found else None,
                     "tool_in_top10": bool(t.get("in_top10")), "tool_in_pool": bool(t.get("in_pool")),
                     "cited_users": t.get("cited_users"), "read_by_tool": t.get("read_by_tool")})
        print(f"[{i}/{len(todo)}] {p['arxiv']} minus {b['name']!r}: llm {'HIT' if found else 'miss'} ({found}), tool {'HIT' if t.get('in_top10') else 'miss'}", flush=True)
    write_atomic(OUT / "loo_llm_results.json", json.dumps(rows, indent=1, ensure_ascii=False))

    def line(label, sel):
        n = len(sel)
        papers_ = sorted({r["arxiv"] for r in sel})
        def agg(key):
            k = sum(1 for r in sel if r[key])
            pairs = [(sum(1 for r in sel if r["arxiv"] == a and r[key]), sum(1 for r in sel if r["arxiv"] == a)) for a in papers_]
            lo, hi = wilson(k, n); blo, bhi = bootstrap_papers(pairs)
            return f"{pc(k, n)} | {100*lo:.0f}-{100*hi:.0f}% benchmarks, {100*blo:.0f}-{100*bhi:.0f}% papers"
        return f"| {label} | {len(papers_)} | {agg('llm_in_top10')} | {agg('tool_in_top10')} |"

    lines = [f"# Leave-one-out (user 1'): Haiku alone against the tool, {PAPERS.name}", "",
             f"Same withheld-benchmark runs as loo_summary.md. Haiku alone ({MODEL}) is given the paper description the tool's ranker saw "
             "(title, summary, method, remaining benchmarks, field, themes; the withheld benchmark scrubbed) and names up to 10 further "
             "benchmarks with no literature search. The tool column is the full pipeline from loo_summary.md. Recovered = the withheld "
             "benchmark is among the 10. Intervals are 90%: Wilson over benchmarks, percentile bootstrap over papers.", "",
             "| Benchmarks | Papers | Haiku alone recovered | 90% CI | Tool recovered | 90% CI |", "| --- | --- | --- | --- | --- | --- |",
             line("All", rows)]
    if any(r["cited_users"] is not None for r in rows):
        lines += [line("Used by 1+ cited paper", [r for r in rows if r["cited_users"]]),
                  line("Used by no cited paper", [r for r in rows if r["cited_users"] == 0])]
    both = sum(1 for r in rows if r["llm_in_top10"] and r["tool_in_top10"]); only_llm = sum(1 for r in rows if r["llm_in_top10"] and not r["tool_in_top10"])
    only_tool = sum(1 for r in rows if r["tool_in_top10"] and not r["llm_in_top10"]); neither = [r for r in rows if not r["llm_in_top10"] and not r["tool_in_top10"]]
    lines += ["", f"- Both recover: {both}; only Haiku alone: {only_llm}; only the tool: {only_tool}; neither: {len(neither)} (of which in the tool's pool: {sum(1 for r in neither if r['tool_in_pool'])}).", "",
              "## Every benchmark", "", "| arXiv | Benchmark | Haiku alone | Tool | Haiku found as |", "| --- | --- | --- | --- | --- |",
              *[f"| {r['arxiv']} | {r['benchmark']} | {r['llm_rank'] or '-'} | {'yes' if r['tool_in_top10'] else ('pool' if r['tool_in_pool'] else '-')} | {r['llm_found_as'] or ''} |" for r in rows]]
    (OUT / "loo_llm_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:12]))


if __name__ == "__main__":
    main()
