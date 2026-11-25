"""Forced title-only probe (9 Nov 2026): the control for memory_probe.py. Same title-only input, but the model must
name up to 10 benchmarks the paper most likely evaluated on, with no option to decline. If the old-set rate is no
higher than the new-set rate on benchmarks the model knows, the title-only score is inference, not recall.

  RRN_OUT=out RRN_PAPERS=papers.json python memory_probe_forced.py ; RRN_OUT=out_2026 RRN_PAPERS=papers_2026.json python memory_probe_forced.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ablation import Names, dedupe  # noqa: E402
from benchmark_eval import HERE, MATCH_MODEL, OUT, PAPERS, cached, match_pool, safe  # noqa: E402
from researchreadernerd.cli import load_dotenv  # noqa: E402
from researchreadernerd.llm import ask, make_client  # noqa: E402

MODEL = "claude-haiku-5-5"
PROMPT = """\
You are given only the title of a research paper. Name up to 10 public benchmarks, datasets or evaluation environments \
the paper most likely reports experiments on, by their standard names, most likely first. Answer from whatever you \
know about the paper and from its topic; always give a list."""


def main() -> None:
    load_dotenv(HERE.parent / ".env")
    papers = json.loads(PAPERS.read_text())
    client, judge = make_client(MODEL), make_client(MATCH_MODEL)
    rows, hits, n = [], 0, 0
    for p in papers:
        tp = OUT / "truth" / f"{p['id']}.json"
        truth = json.loads(tp.read_text()) if tp.exists() else {"benchmarks": []}
        used = [b for b in truth["benchmarks"] if b["kind"] == "public"]
        def compute():
            a = ask(client, model=MODEL, effort="medium", system=PROMPT, content=[{"type": "text", "text": f"Title: {p['title']}"}], schema=Names, max_tokens=1500)
            return {"benchmarks": dedupe(a.value.benchmarks)[:10]}
        r = cached(OUT / "probe_forced" / f"{p['id']}.{safe(MODEL)}.json", compute)
        found = match_pool(p, f"{safe(MODEL)}.probe_forced", truth, r["benchmarks"], judge) if used else []
        for b, f in zip(used, found):
            rows.append({"arxiv": p["arxiv"], "benchmark": b["name"], "hit": bool(f), "found_as": f})
        hits += sum(1 for f in found if f); n += len(used)
        print(f"{p['arxiv']} named={len(r['benchmarks'])} correct={sum(1 for f in found if f)}/{len(used)}", flush=True)
    (OUT / "memory_probe_forced.json").write_text(json.dumps(rows, indent=1))
    print(f"forced title-only, {PAPERS.name}: {hits}/{n} ({100 * hits / max(n, 1):.0f}%)")


if __name__ == "__main__":
    main()
