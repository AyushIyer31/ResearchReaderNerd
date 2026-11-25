"""Version-2 ranking on the saved pools, offline (11 Nov 2026): date every undated candidate (one Haiku call per run,
OpenAlex confirms each source title), show the years to the ranker, rank with Sonnet and the year-aware RANK_PROMPT,
apply the reserved tier as live. Scores reuse the judge's saved matches. The two retrieval changes (arXiv bibliography,
second proposal call) change the pool and cannot be replayed here; they need a live run.

Costs OpenAlex (one title lookup per undated literature candidate, at most 60 per run, cached per title in OUT/dating/)
and Haiku + Sonnet tokens. Run only through launch.py (stage A), under the approved cap.

  RRN_OUT=out_2026 RRN_PAPERS=papers_2026.json python v2_offline.py [--limit N]
      -> OUT/v2_rerank/ (cached), OUT/v2_offline_rows.json, OUT/v2_offline.md
  python v2_offline.py summarize out_2026 out_2026b -> out/report_2026-11-08/v2_offline.md
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src")); sys.path.insert(0, str(HERE))
from researchreadernerd import scholar  # noqa: E402
from researchreadernerd.analyze import DATE_PROMPT, RANK_PROMPT  # noqa: E402
from researchreadernerd.cli import load_dotenv  # noqa: E402
from researchreadernerd.consider import MAX_DATE_LOOKUPS, SUGGESTIONS, DatedNames, Ranking  # noqa: E402
from researchreadernerd.llm import ask, make_client  # noqa: E402
from compare_sets import wilson  # noqa: E402
# slug is copied from leave_one_out.py rather than imported: importing leave_one_out routes every OpenAlex call
# through its response cache in OUT/oa_cache, which in a stage A replay is a hashed as-run input folder (14 Nov 2026).
def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:48] or "x"
from sonnet_rerank import HAIKU, jobs_for, cell  # noqa: E402

OUT = HERE / os.environ.get("RRN_OUT", "out_2026")
PAPERS = HERE / os.environ.get("RRN_PAPERS", "papers_2026.json")
REPORT = Path(os.environ.get("RRN_LAUNCH_REPORT") or HERE / "out" / "report_2026-11-08")  # launch runner: its own report folder
RANKER = "claude-sonnet-5-5"
DESIGNS = ["Haiku ranker + tier (live v1)", "Sonnet, no years, + tier", "v2: Sonnet, dated, + tier"]


def dated_input(c: dict, years: dict[str, int]) -> str:
    head, _, listing = c["rank_input"].partition("Candidates:\n")
    lines = []
    for line in listing.splitlines():
        name = line.split("] ", 1)[1].split(" | ", 1)[0] if "] " in line and " | " in line else None
        y = (c.get("first_seen") or {}).get(name) or years.get(name)
        lines.append(f"{line}; first seen: {y or 'unknown'}")
    return head + "Candidates:\n" + "\n".join(lines)


def catalog_only(c: dict) -> set[str]:
    """Candidates whose only evidence is a catalog keyword hit (strength 0 in the live pipeline), from the rank_input listing."""
    out = set()
    for line in c["rank_input"].partition("Candidates:\n")[2].splitlines():
        if "] " in line and " | " in line:
            name, ev = line.split("] ", 1)[1].split(" | ", 1)
            if all(e.strip().startswith("Found in ") and "by keyword search" in e for e in ev.split("; ")):
                out.add(name)
    return out


def date_run(c: dict, haiku, cache_dir: Path, title_cache: dict) -> dict[str, int]:
    """Years for the run's undated candidates: Haiku names a source paper, OpenAlex confirms; else Haiku's year."""
    # As the live pipeline (consider.py, 11 Nov): only literature candidates are dated (catalog keyword hits, strength 0,
    # are not), and at most MAX_DATE_LOOKUPS OpenAlex title lookups per run (13 Nov 2026).
    undated = [n for n in c["candidates"] if not (c.get("first_seen") or {}).get(n) and n not in catalog_only(c)]
    f = cache_dir / "dates.json"
    done = json.loads(f.read_text()) if f.exists() else {}
    todo = [n for n in undated if n not in done]
    if todo:
        a = ask(haiku, model=HAIKU, effort="low", system=DATE_PROMPT, content=[{"type": "text", "text": "\n".join(f"[{i}] {n}" for i, n in enumerate(todo))}],
                schema=DatedNames, max_tokens=8000)
        got = {d.item: d for d in a.value.items}
        lookups = 0
        for i, n in enumerate(todo):
            d = got.get(i)
            year, src = None, None
            if d and d.source_title.strip() and lookups < MAX_DATE_LOOKUPS:
                lookups += 1
                t = d.source_title.strip()
                if t not in title_cache:
                    w = scholar.find_by_title(t)
                    title_cache[t] = w.year if w and w.year else None
                    cache_dir.parent.mkdir(parents=True, exist_ok=True)
                    tf = cache_dir.parent / "title_years.json"; tf.write_text(json.dumps(title_cache))
                if title_cache[t]:
                    year, src = title_cache[t], "openalex"
            if year is None and d and d.year > 0:
                year, src = d.year, "model"
            done[n] = {"year": year, "source": src}
        cache_dir.mkdir(parents=True, exist_ok=True); f.write_text(json.dumps(done))
        (cache_dir / "dating_call.json").write_text(json.dumps({"model": a.model, "in": a.input_tokens, "out": a.output_tokens, "lookups": lookups}))
    return {n: v["year"] for n, v in done.items() if v.get("year")}


def rank(text: str, client, cache: Path, c: dict) -> list[str]:
    if cache.exists():
        return json.loads(cache.read_text())["picks"]
    a = ask(client, model=RANKER, effort="medium", system=RANK_PROMPT, content=[{"type": "text", "text": text}], schema=Ranking, max_tokens=8000)
    names, seen = [], set()
    for pick in a.value.picks:
        if 0 <= pick.candidate < len(c["candidates"]) and pick.candidate not in seen:
            seen.add(pick.candidate); names.append(c["candidates"][pick.candidate])
        if len(names) == SUGGESTIONS:
            break
    cache.parent.mkdir(parents=True, exist_ok=True); cache.write_text(json.dumps({"picks": names, "in": a.input_tokens, "out": a.output_tokens}))
    return names


def table(rows, title, ci=True):
    L = [f"## {title}", ""]
    for test in ("user 1", "user 1'"):
        sel = [r for r in rows if r["test"] == test]
        if not sel:
            continue
        L += [f"### {test}: {len(sel)} benchmarks", "", "| Benchmarks introduced | n | In pool | " + " | ".join(DESIGNS) + " |", "| --- | --- | --- | " + " | ".join("---" for _ in DESIGNS) + " |"]
        for label, grp in [("2024 or earlier", [r for r in sel if r["year"] and r["year"] <= 2024]), ("2025", [r for r in sel if r["year"] == 2025]),
                           ("2026", [r for r in sel if r["year"] == 2026]), ("all", sel)]:
            if grp:
                L.append(f"| {label} | {len(grp)} | {cell(grp, 'in_pool', ci)} | " + " | ".join(cell(grp, d, ci) for d in DESIGNS) + " |")
        L.append("")
    return L


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", nargs="?", default="run"); ap.add_argument("sets", nargs="*"); ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    if args.cmd == "summarize":
        rows = []
        for s in args.sets:
            rows += json.loads((HERE / s / "v2_offline_rows.json").read_text())
        L = ["# Version 2 ranking on the saved pools: dating + Sonnet + year-aware prompt, against v1", "",
             "Same pools and judge matches. v1 = Haiku ranking with the reserved tier (the live tool). Sonnet without years = the earlier re-rank. "
             "v2 = every undated candidate dated (Haiku names the source paper, OpenAlex confirms the year), years shown to Sonnet with the recency rule in RANK_PROMPT, tier as live. "
             "The retrieval changes are not included (they need a live run).", ""] + table(rows, f"Post-cutoff pooled: {len({r['arxiv'] for r in rows})} papers")
        REPORT.mkdir(parents=True, exist_ok=True); (REPORT / "v2_offline.md").write_text("\n".join(L)); print("\n".join(L)); return
    load_dotenv(HERE.parent / ".env"); haiku = make_client(HAIKU); sonnet = make_client(RANKER)
    papers = {p["id"]: p for p in json.loads(PAPERS.read_text())}
    truth_year, public = {}, set()
    for p in papers.values():
        f = OUT / "truth" / f"{p['id']}.json"
        if f.exists():
            for b in json.loads(f.read_text())["benchmarks"]:
                truth_year[(p["arxiv"], b["name"])] = b.get("year")
                if b.get("kind") == "public":
                    public.add((p["arxiv"], b["name"]))
    jobs = jobs_for(papers)
    if args.limit:
        jobs = jobs[: args.limit]
    tf = OUT / "dating" / "title_years.json"; title_cache = json.loads(tf.read_text()) if tf.exists() else {}
    uniq = {j[5]: j[4] for j in jobs}
    print(f"{len(jobs)} rows, {len(uniq)} runs", flush=True)
    picks, failed = {}, []
    for name, c in uniq.items():
        try:  # a failed call (a refusal, an API or OpenAlex error) skips this run; the runner's next pass retries it, the rest cached
            years = date_run(c, haiku, OUT / "dating" / name.replace(".json", ""), title_cache)
            plain = rank(c["rank_input"], sonnet, OUT / f"rerank_{RANKER}" / name, c)  # the earlier Sonnet re-rank, cached
            v2 = rank(dated_input(c, years), sonnet, OUT / "v2_rerank" / name, c)
        except Exception as e:  # noqa: BLE001
            failed.append(name); print(f"failed {name}: {type(e).__name__}: {e}", flush=True)
            continue
        picks[name] = (plain, v2)
    if failed:
        sys.exit(f"{len(failed)} of {len(uniq)} runs failed; rows not written (run again to retry them)")
    rows = []
    for test, ax, bench, fa, c, name in jobs:
        if (ax, bench) not in public:  # rows follow the current truth lists (14 Nov: EmulatorBench settled as own)
            continue
        tier, ranker = c.get("tier", []), c.get("ranker_picks", [])
        rest = lambda base, p: (base + [n for n in p if n not in base])[:SUGGESTIONS]
        lists = dict(zip(DESIGNS, [rest(tier, ranker), rest(tier, picks[name][0]), rest(tier, picks[name][1])]))
        rows.append({"test": test, "arxiv": ax, "benchmark": bench, "year": truth_year.get((ax, bench)), "in_pool": bool(fa), **{k: bool(fa) and fa in v for k, v in lists.items()}})
    (OUT / "v2_offline_rows.json").write_text(json.dumps(rows, indent=1))
    (OUT / "v2_offline.md").write_text("\n".join(table(rows, OUT.name))); print("\n".join(table(rows, OUT.name, ci=False)))


if __name__ == "__main__":
    main()
