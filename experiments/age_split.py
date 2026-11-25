"""Recall by the year a benchmark was introduced, for the post-cutoff paper sets (10 Nov 2026).

For each public benchmark in a scored paper: model alone (ablation 'llm'), the tool in the author test
(refs+model: final list, ranker-only list, pool), the leave-one-out test (tool, pool) and Haiku alone on the
same scrubbed paper. Year = the earliest year among the source papers the tool's evidence quotes for that
benchmark (rank_input of the blind and leave-one-out runs); else Haiku's own year for the name
(DATE_PROMPT); else a hand assignment from years_manual.json; else undated, shown as its own row.

  python age_split.py out_2026 papers_2026.json out_2026b papers_2026b.json
      -> out/report_2026-11-08/age_rows_<set>.json, age_table_v3.md (per set and both sets pooled)
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "src"))
from compare_sets import wilson  # noqa: E402

MODEL = "claude-haiku-5-5"
REPORT = HERE / "out" / "report_2026-11-08"
# RRN_LAUNCH_REPORT (launch runner, 13 Nov 2026): write rows, the table and the Haiku year cache to this folder instead of the
# report folder, so a launch stage never overwrites the as-run files; the as-run year cache is read when none is there.
AGE_OUT = Path(os.environ.get("RRN_LAUNCH_REPORT") or REPORT)
MANUAL = HERE / "years_manual.json"
PAT = re.compile(r"“([^”]*)” \((\d{4})\)")


def evidence_years(out: Path) -> dict[str, set[int]]:
    years: dict[str, set[int]] = {}
    for f in glob.glob(str(out / "blind" / "*.json")) + glob.glob(str(out / "loo" / "*.json")):
        c = json.load(open(f)).get("consideration") or {}
        for line in (c.get("rank_input") or "").partition("Candidates:")[2].splitlines():
            m = re.match(r"^\[\d+\] (.*?) \| (.*)$", line)
            if m:
                for _, y in PAT.findall(m.group(2)):
                    years.setdefault(m.group(1).strip(), set()).add(int(y))
    return years


MISSING: dict[str, list[str]] = {}


def rows_for(out: Path) -> list[dict]:
    results = json.loads((out / "results.json").read_text())
    loo = {}
    if (out / "loo_results.json").exists():
        loo = {(r["arxiv"], r["benchmark"]): r for r in json.loads((out / "loo_results.json").read_text())}
    loo_llm = {}
    if (out / "loo_llm_results.json").exists():
        loo_llm = {(r["arxiv"], r["benchmark"]): r for r in json.loads((out / "loo_llm_results.json").read_text())}
    years = evidence_years(out)
    forced = {}
    if (out / "memory_probe_forced.json").exists():
        forced = {(r["arxiv"], r["benchmark"]): r["hit"] for r in json.loads((out / "memory_probe_forced.json").read_text())}
    rows = []
    MISSING.setdefault(out.name, [])
    for p in results:
        used = [b["name"] for b in p["truth"]["benchmarks"] if b["kind"] == "public"]
        if not used:
            continue
        def scores(part, key):  # a run that failed every retry is missing: its cells are left out of n and listed
            run = (p.get(part) or {}).get(key)
            if run is None or "score" not in run:
                MISSING[out.name].append(f"{p['arxiv']} {key}")
                return None
            return {s["benchmark"]: s for s in run["score"]}
        tool, topic, llm = scores("blind", f"{MODEL}/refs+model"), scores("blind", f"{MODEL}/topic+model"), scores("ablation", f"{MODEL}/llm")
        probe = {}
        pf = out / "match" / f"{p['id']}.{MODEL}.probe.json"
        if pf.exists():
            m = json.loads(pf.read_text())
            probe = dict(zip(m.get("used", []), m.get("found_as", [])))
        unresolved = {b["name"] for b in p["truth"]["benchmarks"] if b.get("unresolved")}
        stored = {b["name"]: (b.get("year"), b.get("year_source")) for b in p["truth"]["benchmarks"] if b.get("year")}
        hit = lambda d, n, k="in_top10": None if d is None else bool(d.get(n, {}).get(k))  # noqa: E731
        for n in used:
            l = loo.get((p["arxiv"], n)); ll = loo_llm.get((p["arxiv"], n))
            found = {s["found_as"] for s in ((tool or {}).get(n), (topic or {}).get(n)) if s and s.get("found_as")}
            if l and l.get("found_as"):
                found.add(l["found_as"])
            ys = set()
            for f in found:
                ys |= years.get(f, set())
            rows.append({"set": out.name, "id": p["id"], "arxiv": p["arxiv"], "benchmark": n,
                         "llm": hit(llm, n),
                         "tool": hit(tool, n),
                         "tool_ranker": None if tool is None else bool(tool.get(n, {}).get("in_top10_ranker", tool.get(n, {}).get("in_top10"))),
                         "tool_pool": hit(tool, n, "in_pool"),
                         "topic": hit(topic, n),
                         "probe": (bool(probe.get(n)) if pf.exists() else None), "forced": forced.get((p["arxiv"], n)),
                         "unresolved": n in unresolved,
                         "loo_tool": l["in_top10"] if l else None, "loo_pool": l["in_pool"] if l else None,
                         "loo_ranker": l.get("in_top10_ranker", l["in_top10"]) if l else None,
                         "loo_llm": ll["llm_in_top10"] if ll else None,
                         "year": stored[n][0] if n in stored else (min(ys) if ys else None),
                         "year_source": stored[n][1] if n in stored else ("evidence" if ys else None)})
    return rows


def date_with_haiku(rows: list[dict]) -> None:
    from ablation import DATE_PROMPT, Dates
    from researchreadernerd.cli import load_dotenv
    from researchreadernerd.llm import ask, make_client
    load_dotenv(HERE.parent / ".env")
    cache = AGE_OUT / "years_haiku.json"
    seed = cache if cache.exists() else REPORT / "years_haiku.json"
    known = json.loads(seed.read_text()) if seed.exists() else {}
    names = sorted({r["benchmark"] for r in rows if r["year"] is None and r["benchmark"] not in known})
    if names:
        client = make_client(MODEL)
        text = "\n".join(f"[{i}] {n}" for i, n in enumerate(names))
        a = ask(client, model=MODEL, effort="medium", system=DATE_PROMPT, content=[{"type": "text", "text": text}], schema=Dates, max_tokens=4000)
        for d in a.value.items:
            if 0 <= d.item < len(names):
                known[names[d.item]] = {"year": d.year or None, "source_title": d.source_title}
        for n in names:
            known.setdefault(n, {"year": None, "source_title": ""})
        cache.write_text(json.dumps(known, indent=1, ensure_ascii=False))
    for r in rows:
        if r["year"] is None and known.get(r["benchmark"], {}).get("year"):
            r["year"], r["year_source"] = known[r["benchmark"]]["year"], "haiku"


def apply_prior(rows: list[dict]) -> None:
    """Keep the years already sent to the group for the first 2026 set (new_set_years.json, 10 Nov)."""
    prior = REPORT / "new_set_years.json"
    if not prior.exists():
        return
    sent = {(r["arxiv"], r["benchmark"]): r for r in json.loads(prior.read_text())}
    for r in rows:
        s = sent.get((r["arxiv"], r["benchmark"]))
        if s and s.get("year"):
            r["year"], r["year_source"] = s["year"], s.get("year_source") or "evidence"
        elif s:
            r["year"], r["year_source"] = None, None


def apply_manual(rows: list[dict]) -> None:
    manual = json.loads(MANUAL.read_text()) if MANUAL.exists() else {}
    for r in rows:
        if r["year"] is None and manual.get(r["benchmark"]):
            r["year"], r["year_source"] = manual[r["benchmark"]], "manual"


def cell(sel, key, interval=False):
    have = [r for r in sel if r[key] is not None]
    k = sum(1 for r in have if r[key]); n = len(have)
    if not n:
        return "-"
    s = f"{100 * k / n:.0f}% ({k}/{n})"
    if interval:
        a, b = wilson(k, n)
        s += f" [{100 * a:.0f}-{100 * b:.0f}]"
    return s


def table(rows: list[dict], title: str, interval=False) -> list[str]:
    slices = [("2024 or earlier", [r for r in rows if r["year"] and r["year"] <= 2024]),
              ("2025", [r for r in rows if r["year"] == 2025]),
              ("2026", [r for r in rows if r["year"] == 2026]),
              ("undated (no source paper found, model does not know it)", [r for r in rows if not r["year"]]),
              ("all", rows)]
    lines = [f"### {title}", "",
             "| Benchmarks introduced | n | User 1: Haiku alone | User 1: tool, final 10 | User 1: ranker only | User 1: in pool | User 1, topic+model: tool, final 10 "
             "| User 1': Haiku alone | User 1': tool, final 10 | User 1': ranker only | User 1': in pool | Memory probe, declining | Memory probe, forced |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for label, sel in slices:
        if not sel:
            continue
        lines.append(f"| {label} | {len(sel)} | " + " | ".join(cell(sel, k, interval) for k in
                     ("llm", "tool", "tool_ranker", "tool_pool", "topic", "loo_llm", "loo_tool", "loo_ranker", "loo_pool", "probe", "forced")) + " |")
    return lines + [""]


def main():
    args = sys.argv[1:]
    sets = [(HERE / args[i], HERE / args[i + 1]) for i in range(0, len(args), 2)]
    all_rows = []
    lines = ["# Recall by benchmark age, post-cutoff papers", "",
             "Year = the year stored in the checked truth list; else the earliest source paper quoted by the tool's evidence for the benchmark; "
             "else hand-assigned (years_manual.json); else Haiku's year for the name. User 1 = title and abstract; user 1' = leave-one-out on the read paper. 'Ranker only' is the evidence "
             "ranking's own 10 without the reserved tier. Cells with a leave-one-out run still missing are left out of that column's n. "
             "Intervals in the pooled table are 90% Wilson over benchmarks.", ""]
    for out, _ in sets:
        rows = rows_for(out)
        # apply_prior(rows): retired 11 Nov 2026; every truth entry now carries a sourced year, which rows_for prefers.
        apply_manual(rows)  # protocol section 3: a hand assignment with a cited source comes before the model's memory
        date_with_haiku(rows)
        AGE_OUT.mkdir(parents=True, exist_ok=True)
        (AGE_OUT / f"age_rows_{out.name}.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))
        from collections import Counter
        src = Counter(r["year_source"] or "undated" for r in rows)
        lines += table(rows, f"{out.name}: {len({r['id'] for r in rows})} papers, {len(rows)} benchmarks; years from " +
                       ", ".join(f"{v} {k}" for k, v in src.most_common()))
        undated = [r["benchmark"] for r in rows if not r["year"]]
        lines += [f"Undated in {out.name}: " + ", ".join(undated), ""]
        if MISSING.get(out.name):
            lines += [f"Runs missing in {out.name} after every retry (their cells are left out of n): " + ", ".join(MISSING[out.name]), ""]
        all_rows += rows
    if len(sets) > 1:
        lines += table(all_rows, f"Both post-cutoff sets pooled: {len({r['id'] for r in all_rows})} papers, {len(all_rows)} benchmarks", interval=True)
    elif sets:
        lines += table(all_rows, f"{sets[0][0].name} with 90% Wilson intervals: {len({r['id'] for r in all_rows})} papers, {len(all_rows)} benchmarks", interval=True)
    unres = [r for r in all_rows if r.get("unresolved")]
    if unres:  # protocol section 3: unresolved entries stay in, are listed, and the headline is also given without them
        lines += [f"Unresolved availability, kept as public: " + ", ".join(f"{r['benchmark']} ({r['arxiv']})" for r in unres), ""]
        lines += table([r for r in all_rows if not r.get("unresolved")], f"Without the {len(unres)} unresolved entries", interval=True)
    (AGE_OUT / os.environ.get("RRN_AGE_TABLE", "age_table_v3.md")).write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
