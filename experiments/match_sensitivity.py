"""Name-match sensitivity and paired per-paper uncertainty (11 Nov 2026, after Harsh's review).

1. Exact against family matching. The judge maps each truth benchmark to at most one suggestion, allowing spelling,
   version and subset variants ('family'). 'Exact' counts a hit only when the matched suggestion equals the truth name
   or one of its aliases after normalisation. Both systems, both tests, both sets.
2. Paired per-paper difference. For each paper, (Haiku hits - tool hits) / benchmarks; the mean over papers with a
   percentile bootstrap (10,000 rounds, seed 7) gives a 90% interval on the paired difference.

  python match_sensitivity.py -> out/report_2026-11-08/match_sensitivity.md (the development sets, as run)
  python match_sensitivity.py held-out out_heldout papers_heldout.json -> $RRN_LAUNCH_REPORT/match_sensitivity_held-out.md
      (launch, 14 Nov 2026: one named set; adds the user-1 topic+model rows, protocol section 4 reports both conditions)
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPORT = HERE / "out" / "report_2026-11-08"
MODEL = "claude-haiku-5-5"
SETS = {"old set": [("out_old_v2", "papers.json")], "post-cutoff": [("out_2026", "papers_2026.json"), ("out_2026b", "papers_2026b.json")]}
TESTS = ("user 1", "user 1'")
MISSING: list[str] = []
if len(sys.argv) > 1:  # held-out mode: label, then (output folder, paper list) pairs
    SETS = {sys.argv[1]: [(sys.argv[i], sys.argv[i + 1]) for i in range(2, len(sys.argv) - 1, 2)]}
    TESTS = ("user 1", "user 1, topic+model", "user 1'")
    REPORT = Path(os.environ.get("RRN_LAUNCH_REPORT") or REPORT)


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


def load(label):
    rows = []
    for out, pj in SETS[label]:
        aliases = {}
        for p in json.loads((HERE / pj).read_text()):
            f = HERE / out / "truth" / f"{p['id']}.json"
            if f.exists():
                for b in json.loads(f.read_text())["benchmarks"]:
                    aliases[(p["arxiv"], b["name"])] = {norm(a) for a in [b["name"]] + b.get("aliases", [])}
        res = json.loads((HERE / out / "results.json").read_text())
        for p in res:
            if not p.get("blind"):
                continue
            if f"{MODEL}/llm" not in (p.get("ablation") or {}):
                MISSING.append(f"{p['arxiv']} Haiku alone"); continue  # a run that failed every retry is left out and listed
            llm = {s["benchmark"]: s for s in p["ablation"][f"{MODEL}/llm"]["score"]}
            for test, cond in (("user 1", "refs+model"), ("user 1, topic+model", "topic+model")):
                if test not in TESTS:
                    continue
                if f"{MODEL}/{cond}" not in p["blind"]:
                    MISSING.append(f"{p['arxiv']} {cond}"); continue
                tool = {s["benchmark"]: s for s in p["blind"][f"{MODEL}/{cond}"]["score"]}
                for b in tool:
                    rows.append({"test": test, "paper": p["arxiv"], "benchmark": b,
                                 "tool": tool[b]["in_top10"], "tool_found": tool[b].get("found_as"),
                                 "llm": llm[b]["in_top10"], "llm_found": llm[b].get("found_as"), "al": aliases[(p["arxiv"], b)]})
        loo = {(r["arxiv"], r["benchmark"]): r for r in json.loads((HERE / out / "loo_results.json").read_text())}
        lll = {(r["arxiv"], r["benchmark"]): r for r in json.loads((HERE / out / "loo_llm_results.json").read_text())}
        for k, r in loo.items():
            l = lll.get(k, {})
            rows.append({"test": "user 1'", "paper": k[0], "benchmark": k[1], "tool": r["in_top10"], "tool_found": r.get("found_as"),
                         "llm": l.get("llm_in_top10", False), "llm_found": l.get("llm_found_as"), "al": aliases.get(k, set())})
    for r in rows:
        r["tool_exact"] = bool(r["tool"] and norm(r["tool_found"]) in r["al"])
        r["llm_exact"] = bool(r["llm"] and norm(r["llm_found"]) in r["al"])
    return rows


def pct(rows, k):
    return f"{100 * sum(1 for r in rows if r[k]) / len(rows):.0f}% ({sum(1 for r in rows if r[k])}/{len(rows)})"


def paired(rows, a="llm", b="tool", rounds=10000, seed=7, estimator="benchmark"):
    """Paired difference (a minus b) with a percentile bootstrap over papers. estimator="benchmark": pooled hits over
    the resampled papers' benchmarks (benchmark-weighted; a ratio estimator). estimator="paper": the mean over papers
    of each paper's own recall difference (every paper weighted equally)."""
    papers = {}
    for r in rows:
        papers.setdefault(r["paper"], []).append(r)
    ids = list(papers)
    def diff(sel):
        if estimator == "paper":
            return 100 * sum((sum(r[a] for r in papers[pid]) - sum(r[b] for r in papers[pid])) / len(papers[pid]) for pid in sel) / len(sel)
        h = t = n = 0
        for pid in sel:
            for r in papers[pid]:
                h += r[a]; t += r[b]; n += 1
        return 100 * (h - t) / n
    rng = random.Random(seed)
    boots = sorted(diff([rng.choice(ids) for _ in ids]) for _ in range(rounds))
    return diff(ids), boots[int(0.05 * rounds)], boots[int(0.95 * rounds)], sum(1 for pid in ids if sum(r[a] for r in papers[pid]) > sum(r[b] for r in papers[pid])), \
        sum(1 for pid in ids if sum(r[a] for r in papers[pid]) < sum(r[b] for r in papers[pid])), len(ids)


def main():
    L = ["# Name-match sensitivity and paired per-paper uncertainty", "",
         "Family = the judge's matching as used in every table (spelling, version and subset variants allowed). Exact = the matched suggestion equals the truth name or an alias after normalisation. "
         "Paired differences are given under two estimators (see below); 'papers Haiku ahead / tool ahead' counts papers where one side recovered strictly more.", "",
         "| Set | Test | n | Haiku alone, family | Haiku alone, exact | Tool, family | Tool, exact |", "| --- | --- | --- | --- | --- | --- | --- |"]
    P = ["", "Two paired estimators, both bootstrapped over papers (10,000 rounds, seed 7). Benchmark-weighted: hits pooled over the resampled papers' benchmarks (a ratio estimator; the one quoted in earlier notes). "
         "Paper-weighted: the mean over papers of each paper's own recall difference.", "",
         "| Set | Test | Papers | Benchmark-weighted: Haiku minus tool | 90% CI | Paper-weighted: Haiku minus tool | 90% CI | Papers Haiku ahead / tool ahead |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    loose = []
    for label in SETS:
        rows = load(label)
        for test in TESTS:
            sel = [r for r in rows if r["test"] == test]
            if not sel:
                continue
            L.append(f"| {label} | {test} | {len(sel)} | {pct(sel, 'llm')} | {pct(sel, 'llm_exact')} | {pct(sel, 'tool')} | {pct(sel, 'tool_exact')} |")
            d, lo, hi, ha, ta, n = paired(sel)
            d2, lo2, hi2, _, _, _ = paired(sel, estimator="paper")
            P.append(f"| {label} | {test} | {n} | {d:+.1f} | {lo:+.1f} to {hi:+.1f} | {d2:+.1f} | {lo2:+.1f} to {hi2:+.1f} | {ha} / {ta} |")
            if label != "old set":
                loose += [(test, r["paper"], r["benchmark"], "tool", r["tool_found"]) for r in sel if r["tool"] and not r["tool_exact"]]
                loose += [(test, r["paper"], r["benchmark"], "Haiku", r["llm_found"]) for r in sel if r["llm"] and not r["llm_exact"]]
    L += P + ["", ("## Post-cutoff hits" if len(sys.argv) == 1 else f"## {sys.argv[1]} hits") + " that are family matches, not exact (for the hand-check of the judge)", "",
              "| Test | Paper | Truth benchmark | System | Matched suggestion |", "| --- | --- | --- | --- | --- |"]
    for t, pap, b, sy, fa in sorted(loose):
        L.append(f"| {t} | {pap} | {b} | {sy} | {fa} |")
    if MISSING:
        L += ["", "Runs missing after every retry, left out of the rows above: " + ", ".join(MISSING)]
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / ("match_sensitivity.md" if len(sys.argv) == 1 else f"match_sensitivity_{sys.argv[1]}.md")).write_text("\n".join(L) + "\n")
    print("\n".join(L[:20])); print(f"... {len(loose)} family-only hits listed")


if __name__ == "__main__":
    main()
