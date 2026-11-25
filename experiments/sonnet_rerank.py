"""A stronger model for the one ranking call, offline on the saved pools (11 Nov 2026).

For every saved user-1 (refs+model) and leave-one-out run, the exact text the Haiku ranker saw (rank_input) is sent
unchanged, with the unchanged RANK_PROMPT, to another model (default claude-sonnet-5-5). Its 10 picks replace the
ranker's picks; the reserved tier of confirmed proposals is then applied exactly as live. Scores reuse the judge's saved
name matches per benchmark (found_as), so no new matching: a benchmark counts as recovered when its matched candidate
is in the list. Nothing touches OpenAlex.

  RRN_OUT=out_2026 RRN_PAPERS=papers_2026.json python sonnet_rerank.py [--model claude-sonnet-5-5] [--limit N]
      -> OUT/rerank_<model>/ (cached picks), OUT/sonnet_rerank.md, OUT/sonnet_rerank_rows.json
  python sonnet_rerank.py summarize out_old_v2 out_2026 out_2026b   -> out/report_2026-11-08/sonnet_rerank.md
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src")); sys.path.insert(0, str(HERE))
from researchreadernerd.analyze import RANK_PROMPT  # noqa: E402
from researchreadernerd.cli import load_dotenv  # noqa: E402
from researchreadernerd.consider import SUGGESTIONS, Ranking  # noqa: E402
from researchreadernerd.llm import ask, make_client  # noqa: E402
from compare_sets import wilson  # noqa: E402
# slug is copied from leave_one_out.py rather than imported: importing leave_one_out routes every OpenAlex call
# through its response cache in OUT/oa_cache, which in a stage A replay is a hashed as-run input folder (14 Nov 2026).
def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:48] or "x"

OUT = HERE / os.environ.get("RRN_OUT", "out_2026")
PAPERS = HERE / os.environ.get("RRN_PAPERS", "papers_2026.json")
REPORT = HERE / "out" / "report_2026-11-08"
HAIKU = "claude-haiku-5-5"
DESIGNS = ["Haiku ranker alone", "Haiku ranker + tier (live)", "New ranker alone", "New ranker + tier"]


def rerank(c: dict, client, model: str, cache: Path) -> list[str]:
    if cache.exists():
        d = json.loads(cache.read_text())
        return d["picks"] if isinstance(d, dict) else d
    a = ask(client, model=model, effort="medium", system=RANK_PROMPT, content=[{"type": "text", "text": c["rank_input"]}],
            schema=Ranking, max_tokens=8000)
    names, seen = [], set()
    for pick in a.value.picks:
        if 0 <= pick.candidate < len(c["candidates"]) and pick.candidate not in seen:
            seen.add(pick.candidate); names.append(c["candidates"][pick.candidate])
        if len(names) == SUGGESTIONS:
            break
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"picks": names, "model": a.model, "in": a.input_tokens, "out": a.output_tokens}))
    return names


def lists(c: dict, new_picks: list[str]) -> dict[str, list[str]]:
    tier, ranker = c.get("tier", []), c.get("ranker_picks", [])
    rest = lambda base, picks: (base + [n for n in picks if n not in base])[:SUGGESTIONS]
    return dict(zip(DESIGNS, [ranker[:SUGGESTIONS], rest(tier, ranker), new_picks[:SUGGESTIONS], rest(tier, new_picks)]))


def jobs_for(papers: dict) -> list:
    jobs = []  # (test, arxiv, benchmark, found_as, consideration, cache name)
    for p in json.loads((OUT / "results.json").read_text()):
        if not p.get("blind"):
            continue
        c = json.loads((OUT / "blind" / f"{p['id']}.{HAIKU}.refs+model.json").read_text())["consideration"]
        for s in p["blind"][f"{HAIKU}/refs+model"]["score"]:
            jobs.append(("user 1", p["arxiv"], s["benchmark"], s.get("found_as"), c, f"{p['id']}.refs+model.json"))
    by_arxiv = {p["arxiv"]: p["id"] for p in papers.values()}
    for r in json.loads((OUT / "loo_results.json").read_text()):
        pid = by_arxiv[r["arxiv"]]
        f = OUT / "loo" / f"{pid}.{HAIKU}.{slug(r['benchmark'])}.json"
        if f.exists():
            d = json.loads(f.read_text())
            if d.get("consideration", {}).get("rank_input"):
                jobs.append(("user 1'", r["arxiv"], r["benchmark"], r.get("found_as"), d["consideration"], f.name))
    return jobs


def cell(grp, k, ci=False):
    if not grp:
        return "-"
    h = sum(1 for r in grp if r[k]); s = f"{100 * h / len(grp):.0f}% ({h}/{len(grp)})"
    if ci:
        a, b = wilson(h, len(grp)); s += f" [{100 * a:.0f}-{100 * b:.0f}]"
    return s


def table(rows, title, old=False, ci=False):
    lines = [f"## {title}", ""]
    for test in ("user 1", "user 1'"):
        sel = [r for r in rows if r["test"] == test]
        if not sel:
            continue
        lines += [f"### {test}: {len(sel)} benchmarks", "", "| Benchmarks introduced | n | In pool | " + " | ".join(DESIGNS) + " |",
                  "| --- | --- | --- | " + " | ".join("---" for _ in DESIGNS) + " |"]
        groups = [("all", sel)] if old else [("2024 or earlier", [r for r in sel if r["year"] and r["year"] <= 2024]),
                                             ("2025", [r for r in sel if r["year"] == 2025]), ("2026", [r for r in sel if r["year"] == 2026]), ("all", sel)]
        for label, grp in groups:
            if grp:
                lines.append(f"| {label} | {len(grp)} | {cell(grp, 'in_pool', ci)} | " + " | ".join(cell(grp, d, ci) for d in DESIGNS) + " |")
        lines.append("")
    return lines


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", nargs="?", default="run"); ap.add_argument("sets", nargs="*")
    ap.add_argument("--model", default="claude-sonnet-5-5"); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--threads", type=int, default=6)
    args = ap.parse_args()
    if args.cmd == "summarize":
        lines = ["# A stronger ranker on the saved pools: Sonnet 5.5 against Haiku 5.5 for the one ranking call", "",
                 "Same runs, same pools, same ranker input and prompt, same judge matches; only the model making the ranking call differs. "
                 "'+ tier' applies the reserved tier of confirmed model proposals exactly as the live tool does. Intervals are 90% Wilson over benchmarks.", ""]
        allnew = []
        for s in args.sets:
            rows = json.loads((HERE / s / "sonnet_rerank_rows.json").read_text())
            old = s == "out_old_v2"
            lines += table(rows, f"{s}: {len({r['arxiv'] for r in rows})} papers", old=old, ci=True)
            if not old:
                allnew += rows
        if allnew:
            lines += table(allnew, f"Post-cutoff pooled: {len({r['arxiv'] for r in allnew})} papers", ci=True)
        (REPORT / "sonnet_rerank.md").write_text("\n".join(lines)); print("\n".join(lines)); return
    load_dotenv(HERE.parent / ".env"); client = make_client(args.model)
    papers = {p["id"]: p for p in json.loads(PAPERS.read_text())}
    truth_year = {}
    for p in papers.values():
        f = OUT / "truth" / f"{p['id']}.json"
        if f.exists():
            for b in json.loads(f.read_text())["benchmarks"]:
                truth_year[(p["arxiv"], b["name"])] = b.get("year")
    jobs = jobs_for(papers)
    if args.limit:
        jobs = jobs[: args.limit]
    cdir = OUT / f"rerank_{args.model}"
    uniq = {j[5]: j[4] for j in jobs}
    print(f"{len(jobs)} benchmark rows, {len(uniq)} rankings to make with {args.model}", flush=True)
    with ThreadPoolExecutor(args.threads) as pool:
        picks = dict(zip(uniq, pool.map(lambda kv: rerank(kv[1], client, args.model, cdir / kv[0]), uniq.items())))
    rows = []
    for test, ax, bench, fa, c, name in jobs:
        L = lists(c, picks[name])
        rows.append({"test": test, "arxiv": ax, "benchmark": bench, "year": truth_year.get((ax, bench)), "in_pool": bool(fa),
                     **{k: bool(fa) and fa in v for k, v in L.items()}})
    (OUT / "sonnet_rerank_rows.json").write_text(json.dumps(rows, indent=1))
    toks = [json.loads(f.read_text()) for f in cdir.glob("*.json")]
    lines = [f"# Ranking call by {args.model} on the saved pools, {OUT.name}", "",
             f"{len(toks)} rankings; tokens in {sum(t.get('in', 0) for t in toks):,}, out {sum(t.get('out', 0) for t in toks):,}.", ""]
    lines += table(rows, OUT.name, old=OUT.name == "out_old_v2")
    (OUT / "sonnet_rerank.md").write_text("\n".join(lines)); print("\n".join(lines))


if __name__ == "__main__":
    main()
