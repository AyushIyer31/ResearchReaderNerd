"""Draw the post-cutoff paper set (8 Nov 2026): computer-science arXiv preprints posted August to October 2026,
after Claude Haiku 5.5's June 2026 training cutoff, so the model cannot have read them. Mirrors papers.json:
OpenAlex preprints with an abstract and a reference list of 15 or more, at most 3 per topic, chosen by a seeded
random sample. Writes papers_2026.json with the same fields as papers.json.

  python sample_2026.py [--n 30] [--seed 7]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import httpx2

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
from researchreadernerd.cli import load_dotenv  # noqa: E402
from researchreadernerd.scholar import _abstract  # noqa: E402

ARXIV_SOURCE = "S4306400194"
FILTER = ("from_publication_date:{start},to_publication_date:{end},type:preprint,"
          f"primary_topic.field.id:fields/17,referenced_works_count:>15,primary_location.source.id:{ARXIV_SOURCE},has_abstract:true")
SELECT = "id,title,publication_year,publication_date,doi,cited_by_count,referenced_works_count,primary_topic,abstract_inverted_index,ids"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--per-topic", type=int, default=3)
    ap.add_argument("--start", default="2026-08-01")
    ap.add_argument("--end", default="2026-10-31")
    ap.add_argument("--out", default="papers_2026.json")
    ap.add_argument("--exclude", default="", help="comma-separated paper files whose arXiv ids are skipped")
    args = ap.parse_args()
    skip = {p["arxiv"] for f in args.exclude.split(",") if f for p in json.loads((HERE / f).read_text())} | {"2609.38721"}
    load_dotenv(HERE.parent / ".env")
    key = os.environ["OPENALEX_API_KEY"]
    r = httpx2.get("https://api.openalex.org/works", timeout=90,
                   params={"filter": FILTER.format(start=args.start, end=args.end), "sample": 150, "seed": args.seed, "per_page": 150, "select": SELECT, "api_key": key})
    r.raise_for_status()
    # The draw's full response, kept as the record of the sample (OpenAlex's seeded sample changes as its index grows).
    (HERE / args.out.replace("papers", "draw_response")).write_bytes(r.content)
    raw, picked, per_topic = [], [], Counter()
    for w in r.json()["results"]:
        m = re.search(r"arxiv\.(\d{4}\.\d{4,5})", (w.get("doi") or "") + " " + (w["ids"].get("arxiv") or ""), re.I)
        if not m or m.group(1) in skip:
            continue
        abstract = _abstract(w.get("abstract_inverted_index"))
        if len(abstract) < 300:
            continue
        topic = (w.get("primary_topic") or {}).get("display_name", "")
        row = {"id": w["id"].rsplit("/", 1)[-1], "arxiv": m.group(1), "title": w["title"], "year": w["publication_year"],
               "doi": w.get("doi"), "cited_by": w["cited_by_count"], "refs": w["referenced_works_count"], "topic": topic,
               "abstract": abstract, "published": w["publication_date"]}
        raw.append(row)
        if per_topic[topic] < args.per_topic and len(picked) < args.n:
            per_topic[topic] += 1
            picked.append(row)
    (HERE / args.out.replace("papers", "sample_raw")).write_text(json.dumps(raw, indent=1, ensure_ascii=False))
    (HERE / args.out).write_text(json.dumps(picked, indent=1, ensure_ascii=False))
    print(f"{len(raw)} candidates, {len(picked)} picked -> {args.out}")
    for p in picked:
        print(f"  {p['arxiv']} {p['published']} refs={p['refs']} cited={p['cited_by']} | {p['topic'][:40]} | {p['title'][:60]}")


if __name__ == "__main__":
    main()
