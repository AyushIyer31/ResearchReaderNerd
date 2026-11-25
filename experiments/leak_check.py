"""Leakage checks on the leave-one-out runs (9 Nov 2026), from the saved files only.

1. Does any name of the withheld benchmark survive in the paper description the ranker saw?  (loo_results: leak_in_used_list)
2. Is the source paper P itself listed as a user of the recovered benchmark, i.e. did the tool read P's own record back?
3. How often does P appear anywhere in the evidence (as a user of its other, unhidden benchmarks; expected and harmless).

  RRN_OUT=out RRN_PAPERS=papers.json python leak_check.py ; RRN_OUT=out_2026 RRN_PAPERS=papers_2026.json python leak_check.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_eval import OUT, PAPERS, safe  # noqa: E402
from leave_one_out import MODEL, slug  # noqa: E402


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def main() -> None:
    papers = {p["arxiv"]: p for p in json.loads(PAPERS.read_text())}
    res = json.loads((OUT / "loo_results.json").read_text())
    n = len(res); rec = sum(1 for r in res if r["in_top10"])
    used_list_leaks = sum(1 for r in res if r["leak_in_used_list"])
    self_as_evidence = []; self_anywhere = 0
    for r in res:
        p = papers[r["arxiv"]]
        c = json.loads((OUT / "loo" / f"{p['id']}.{safe(MODEL)}.{slug(r['benchmark'])}.json").read_text())["consideration"]
        t = norm(p["title"])[:50]
        if any(t in norm(x["title"]) for u in c["used_by"] for x in u["papers"]):
            self_anywhere += 1
        if r["found_as"]:
            ub = next((u for u in c["used_by"] if u["benchmark"] == r["found_as"]), None)
            if ub and any(t in norm(x["title"]) for x in ub["papers"]):
                self_as_evidence.append((r["arxiv"], r["benchmark"], r["in_top10"]))
    lines = [f"# Leakage checks, {PAPERS.name}", "",
             f"- Runs: {n}; recovered in 10: {rec}.",
             f"- Withheld benchmark's name (or alias) still present in the description the ranker saw: {used_list_leaks} runs.",
             f"- Source paper P listed as a user of the recovered or in-pool withheld benchmark (P's own record read back): {len(self_as_evidence)} runs.",
             f"- P appears somewhere in the evidence as a user of its other, unhidden benchmarks: {self_anywhere} runs (expected; P uses them).",
             "- The tool never sees P's PDF in this test: it receives the scrubbed analysis (title, summary, method, remaining benchmarks) and P's reference list from OpenAlex with the withheld benchmark's dataset paper removed. Tables and captions are therefore not an input.",
             ""]
    (OUT / "leak_check.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
