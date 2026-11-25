"""Memory probe (8 Nov 2026): does the model remember the paper, or infer from the topic?

Given only a paper's title (no abstract), the model is asked which benchmarks that paper evaluated on. A
paper the model has read can be answered from memory; a paper posted after its training cutoff cannot. Run
on both sets (papers.json, 2018 to 2021; papers_2026.json, August to October 2026) and compare: the gap is
the memory component in the model-alone figures.

  RRN_OUT=out RRN_PAPERS=papers.json python memory_probe.py
  RRN_OUT=out_2026 RRN_PAPERS=papers_2026.json python memory_probe.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from pydantic import BaseModel, ConfigDict

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_eval import HERE, MATCH_MODEL, OUT, PAPERS, cached, match_pool, safe  # noqa: E402
from researchreadernerd.cli import load_dotenv  # noqa: E402
from researchreadernerd.llm import ask, make_client  # noqa: E402

MODEL = "claude-haiku-5-5"
PROBE_PROMPT = """\
You are asked about a specific published research paper, identified only by its title. If you know this paper, \
list the benchmarks, datasets or evaluation environments it reports experiments on, by their standard names, \
up to 10. If you do not know the paper or its experiments, return an empty list rather than guessing; do not \
infer from the topic. Also say whether you recognise the paper."""


class Probe(BaseModel):
    model_config = ConfigDict(extra="forbid")
    recognised: bool
    benchmarks: list[str]


def run_probe(p: dict, client) -> dict:
    def compute():
        t0 = time.time()
        answer = ask(client, model=MODEL, effort="medium", system=PROBE_PROMPT,
                     content=[{"type": "text", "text": f"Title: {p['title']}"}], schema=Probe, max_tokens=1500)
        names = list(dict.fromkeys(n.strip() for n in answer.value.benchmarks if n.strip()))[:10]
        return {"recognised": answer.value.recognised, "benchmarks": names, "seconds": round(time.time() - t0, 1)}
    return cached(OUT / "probe" / f"{p['id']}.{safe(MODEL)}.json", compute)


def main() -> None:
    load_dotenv(HERE.parent / ".env")
    papers = json.loads(PAPERS.read_text())
    client, judge = make_client(MODEL), make_client(MATCH_MODEL)
    (OUT / "probe").mkdir(parents=True, exist_ok=True)
    n_bench = hits = recognised = named = 0
    rows = []
    for p in papers:
        truth_path = OUT / "truth" / f"{p['id']}.json"
        truth = json.loads(truth_path.read_text()) if truth_path.exists() else {"benchmarks": []}
        used = [b for b in truth["benchmarks"] if b["kind"] == "public"]
        r = run_probe(p, client)
        found = match_pool(p, f"{safe(MODEL)}.probe", truth, r["benchmarks"], judge) if used else []
        k = sum(1 for f in found if f)
        n_bench += len(used)
        hits += k
        recognised += r["recognised"]
        named += bool(r["benchmarks"])
        rows.append(f"| {p['arxiv']} | {p['title'][:55]} | {'yes' if r['recognised'] else 'no'} | {len(r['benchmarks'])} | {k}/{len(used)} |")
        print(f"{p['arxiv']} recognised={r['recognised']} named={len(r['benchmarks'])} correct={k}/{len(used)}", flush=True)
    lines = [f"# Memory probe: {PAPERS.name}", "",
             f"Title only, no abstract: does {MODEL} say it knows the paper, and does it name the benchmarks the paper used? "
             "Correct = named benchmarks matched to the hand-checked (or auto) truth list by the usual judge.", "",
             f"- Papers: {len(papers)}; says it recognises the paper: {recognised}; names at least one benchmark: {named}",
             f"- Benchmarks correctly recalled from the title alone: {hits}/{n_bench} ({100 * hits / max(n_bench, 1):.0f}%)", "",
             "| arXiv | Title | Recognised | Named | Correct / used |", "| --- | --- | --- | --- | --- |", *rows]
    (OUT / "memory_probe.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:6]))


if __name__ == "__main__":
    main()
