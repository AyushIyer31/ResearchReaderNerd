"""Free sources vs OpenAlex on the five things the tool uses OpenAlex for (Shasha / Harsh, 24 Nov 2026).

Development papers only (post-cutoff sets a and b); no Claude calls; the held-out windows (4-8 and 9-13 Nov 2026)
are not touched. OpenAlex is read from the existing response caches first (read only); live OpenAlex calls stay
inside the free $1 daily budget (hard stop at $0.80 used, read from the rate-limit endpoint). Semantic Scholar is
used without a key, slowly, with backoff on 429. New responses go to out/free_source_cache (not a hashed input).

Gold, independent of OpenAlex: for each public truth benchmark, the entry in the paper's own arXiv bibliography
whose text names the benchmark (its dataset paper, as the authors cite it).

Rules fixed before running: a free route wins a function if it is within 5 points of OpenAlex on the gold measure
(or better) at a workable call count; any function it loses is listed as a gap that still needs OpenAlex.

  python free_source_test.py            # runs all five checks, writes out/report_2026-11-08/free_source_test.{md,json}
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import sys
import time
import urllib.parse
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import httpx2

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
from researchreadernerd.cli import load_dotenv  # noqa: E402
from researchreadernerd.scholar import titles_match, _clean  # noqa: E402

load_dotenv(HERE.parent / ".env")
OA_KEY = os.environ.get("OPENALEX_API_KEY", "")
CACHE = HERE / "out" / "free_source_cache"
CACHE.mkdir(parents=True, exist_ok=True)
OA_CACHES = [HERE / s / "oa_cache" for s in ("out_2026", "out_2026b", "out_old_v2")] + [HERE / "out" / "oa_cache"]
HTML_CACHE = HERE / "out" / "arxiv_html_cache"
REPORT = HERE / "out" / "report_2026-11-08"
SETS = [("out_2026", "papers_2026.json"), ("out_2026b", "papers_2026b.json")]
N_PAPERS, TITLES_PER_PAPER, SEED = 10, 15, 7
OA_STOP_USD = 0.80
S2_MAX_CALLS = 2500
UA = {"User-Agent": "ResearchReaderNerd-free-source-test/0.1"}

STATS: Counter = Counter()
TIMES: dict[str, list[float]] = {}
_last: dict[str, float] = {}


# ---------- HTTP with disk cache, pacing and backoff ----------

def _key(name: str, url: str, params: dict) -> Path:
    return CACHE / f"{name}-{hashlib.sha1(json.dumps([url, params], sort_keys=True).encode()).hexdigest()}.json"


def fetch(name: str, url: str, params: dict | None = None, *, gap: float = 1.0, text: bool = False, tries: int = 6):
    """GET with a per-service gap between calls; 429/5xx back off (exponential, honours Retry-After). Cached on disk.
    Returns (status, body) where body is parsed JSON (or text), or None on failure."""
    params = dict(params or {})
    f = _key(name, url, params)
    if f.exists():
        d = json.loads(f.read_text())
        return d["status"], d["body"]
    wait = 2.0
    status, body = None, None
    for attempt in range(tries):
        dt = gap - (time.time() - _last.get(name, 0))
        if dt > 0:
            time.sleep(dt)
        t0 = time.time()
        try:
            r = httpx2.get(url, params=params, headers=UA, timeout=40, follow_redirects=True)
            status = r.status_code
        except httpx2.HTTPError as e:
            status = f"error {type(e).__name__}"
        _last[name] = time.time()
        STATS[f"{name}_calls"] += 1
        TIMES.setdefault(name, []).append(time.time() - t0)
        if status == 200:
            body = r.text if text else r.json()
            break
        if status in (429, 500, 502, 503, 504) or isinstance(status, str):
            STATS[f"{name}_{status}"] += 1
            ra = r.headers.get("Retry-After") if not isinstance(status, str) else None
            time.sleep(min(max(wait, float(ra) if ra and ra.isdigit() else 0), 90))
            wait *= 2
            continue
        break  # 400/404: final
    if status in (200, 404, 400):
        f.write_text(json.dumps({"status": status, "body": body}))
    return status, body


# ---------- OpenAlex: existing caches first, then live within the free budget ----------

def oa_spent() -> float:
    r = httpx2.get("https://api.openalex.org/rate-limit", params={"api_key": OA_KEY}, timeout=20).json()["rate_limit"]
    return r["daily_used_usd"]


_OA_USED = [0.0]


def oa(path: str, params: dict) -> dict:
    shown = dict(params)
    key = hashlib.sha1(json.dumps([path, shown], sort_keys=True, default=str).encode()).hexdigest()
    for c in OA_CACHES:
        if (c / f"{key}.json").exists():
            STATS["openalex_cache_hits"] += 1
            return json.loads((c / f"{key}.json").read_text())
    f = CACHE / f"oa-{key}.json"
    if f.exists():
        STATS["openalex_cache_hits"] += 1
        return json.loads(f.read_text())
    price = 0.001 if "search" in json.dumps(shown) else (0.0 if path.startswith("/works/") else 0.0001)
    if _OA_USED[0] + price > OA_STOP_USD:
        STATS["openalex_skipped_budget"] += 1
        return {"_skipped": True}
    status, body = fetch("openalex", "https://api.openalex.org" + path, {**shown, "api_key": OA_KEY}, gap=0.15)
    _OA_USED[0] += price
    if status == 200:
        f.write_text(json.dumps(body))
        return body
    return {}


# ---------- gold from the arXiv bibliography ----------

_BIBITEM = re.compile(r'<li[^>]*class="[^"]*ltx_bibitem[^"]*"[^>]*>(.*?)</li>', re.S)
_BIBBLOCK = re.compile(r'<span[^>]*class="[^"]*ltx_bibblock[^"]*"[^>]*>(.*?)</span>', re.S)
_TAG = re.compile(r"<[^>]+>")


def bibliography(arxiv_id: str) -> list[dict]:
    import html as H
    f = HTML_CACHE / f"{arxiv_id}.html"
    if f.exists():
        page = f.read_text(errors="ignore")
    else:  # the paper's own arXiv HTML (free), cached under out/free_source_cache
        st, page = fetch("arxivhtml", f"https://arxiv.org/html/{arxiv_id}", {}, gap=1.5, text=True)
        if st != 200 or not page:
            return []
    out = []
    for item in _BIBITEM.findall(page):
        blocks = [H.unescape(re.sub(r"\s+", " ", _TAG.sub(" ", b)).strip(" .")) for b in _BIBBLOCK.findall(item)]
        blocks = [b for b in blocks if b]
        if not blocks:
            continue
        title = blocks[1] if len(blocks) >= 2 else blocks[0]
        whole = " ".join(blocks)
        ax = re.search(r"\b(\d{4}\.\d{4,5})\b", whole)
        if len(title.split()) >= 3:
            out.append({"title": title, "text": whole, "arxiv": ax.group(1) if ax else None})
    return out


def names(b: dict) -> list[str]:
    return [a for a in dict.fromkeys([b["name"], *b.get("aliases", [])]) if len(a.strip()) >= 4]


def gold_for(truth: dict, bib: list[dict]) -> list[dict]:
    gold = []
    for b in truth.get("benchmarks", []):
        if b.get("kind") != "public":
            continue
        pats = [re.compile(r"(?<![A-Za-z0-9])" + re.escape(a) + r"(?![A-Za-z0-9])", re.I) for a in names(b)]
        hit = next((e for e in bib if any(p.search(e["title"]) for p in pats)), None)  # the entry whose title names it
        if hit and not any(g["title"] == hit["title"] for g in gold):
            gold.append({"benchmark": b["name"], **hit})
    return gold


# ---------- the five checks ----------

def s2(path: str, params: dict):
    if STATS["s2_calls"] >= S2_MAX_CALLS:
        STATS["s2_skipped_cap"] += 1
        return "cap", None
    return fetch("s2", "https://api.semanticscholar.org/graph/v1" + path, params, gap=1.2)


def title_lookups(titles: list[str]) -> dict:
    rows = []
    for t in titles:
        r = {"title": t}
        d = oa("/works", {"filter": f"title.search:{_clean(t)}", "sort": "cited_by_count:desc", "per-page": 5,
                          "select": "id,title,abstract_inverted_index,publication_year,doi,cited_by_count"})
        if d.get("_skipped"):
            r["openalex"] = None
        else:
            m = next((w for w in d.get("results", []) if titles_match(w.get("title") or "", t)), None)
            r["openalex"] = bool(m)
            r["openalex_id"] = m["id"].rsplit("/", 1)[-1] if m else None
            r["openalex_rank1"] = bool(d.get("results")) and titles_match(d["results"][0].get("title") or "", t)
        st, body = s2("/paper/search/match", {"query": t, "fields": "title,year,externalIds,citationCount"})
        if st == 200 and body and body.get("data"):
            m = body["data"][0]
            r["s2"] = titles_match(m.get("title") or "", t)
            r["s2_id"] = m.get("paperId") if r["s2"] else None
        elif st in (404, 200):
            r["s2"] = False
        else:
            r["s2"] = None  # blocked / failed
            r["s2_status"] = st
        st, body = fetch("crossref", "https://api.crossref.org/works",
                         {"query.bibliographic": t, "rows": 5, "select": "DOI,title"}, gap=1.1)
        if st == 200:
            items = body["message"]["items"]
            r["crossref"] = any(titles_match((i.get("title") or [""])[0], t) for i in items)
            r["crossref_rank1"] = bool(items) and titles_match((items[0].get("title") or [""])[0], t)
        else:
            r["crossref"] = None
        rows.append(r)
    return rows


def references(p: dict, gold: list[dict]) -> dict:
    ax = p["arxiv"]
    out = {"paper": ax, "gold": len(gold)}
    d = oa(f"/works/doi:10.48550/arXiv.{ax}", {"select": "id,title,referenced_works"})
    ids = [w.rsplit("/", 1)[-1] for w in d.get("referenced_works") or []]
    titles = []
    for i in range(0, len(ids), 50):
        r = oa("/works", {"filter": f"openalex_id:{'|'.join(ids[i:i + 50])}", "per-page": 50,
                          "select": "id,title,abstract_inverted_index,publication_year,doi,cited_by_count,referenced_works"})
        titles += [w.get("title") or "" for w in r.get("results", [])]
    out["openalex_refs"] = len(ids)
    out["openalex_gold"] = sum(any(titles_match(t, g["title"]) for t in titles) for g in gold)
    st, body = s2(f"/paper/arXiv:{ax}/references", {"fields": "title,year", "limit": 1000})
    if st == 200 and body is not None:
        s2t = [(x.get("citedPaper") or {}).get("title") or "" for x in body.get("data") or []]
        out["s2_refs"] = len(s2t)
        out["s2_gold"] = sum(any(titles_match(t, g["title"]) for t in s2t) for g in gold)
        out["s2_elided"] = body.get("data") is None
    else:
        out["s2_refs"] = None
        out["s2_status"] = st
    st, body = fetch("opencitations", f"https://api.opencitations.net/index/v2/references/doi:10.48550/arXiv.{ax}",
                     {}, gap=0.5)
    out["opencitations_refs"] = len(body) if st == 200 and isinstance(body, list) else (0 if st in (200, 404) else None)
    st, _ = fetch("crossref", f"https://api.crossref.org/works/10.48550/arXiv.{ax}", {}, gap=1.1)
    out["crossref_has_record"] = st == 200
    out["arxiv_bibliography_entries"] = len(bibliography(ax))
    return out


def co_usage(src: dict) -> dict:
    """Top 15 most-cited papers citing a benchmark's source paper: OpenAlex in one call; S2 by paging all citations."""
    out = {"benchmark": src["benchmark"], "title": src["title"]}
    d = oa("/works", {"filter": f"cites:{src['openalex_id']}", "sort": "cited_by_count:desc", "per-page": 15,
                      "select": "id,title,abstract_inverted_index,publication_year,doi,cited_by_count,referenced_works"})
    oa_top = [w.get("title") or "" for w in d.get("results", [])]
    out["openalex_calls"] = 1
    out["openalex_total_citers"] = (d.get("meta") or {}).get("count")
    pid = f"arXiv:{src['arxiv']}" if src.get("arxiv") else src.get("s2_id")
    if not pid:
        out["s2"] = "no id"
        return out
    allc, calls, offset, st = [], 0, 0, None
    while True:
        st, body = s2(f"/paper/{pid}/citations", {"fields": "title,year,citationCount", "limit": 1000, "offset": offset})
        calls += 1
        if st != 200 or body is None:
            break
        data = body.get("data") or []
        allc += [x.get("citingPaper") or {} for x in data]
        if body.get("next") is None or not data:
            break
        offset = body["next"]
        if calls >= 20:
            out["s2_stopped_at_20_pages"] = True
            break
    out["s2_calls"] = calls
    out["s2_status"] = st
    out["s2_citers_returned"] = len(allc)
    s2_top = [c.get("title") or "" for c in sorted(allc, key=lambda c: -(c.get("citationCount") or 0))[:15]]
    out["overlap_top15"] = sum(any(titles_match(a, b) for b in s2_top) for a in oa_top) if oa_top and s2_top else None
    out["openalex_top15"] = len(oa_top)
    return out


def topic(query: str, gold_titles: list[str], year: int) -> dict:
    out = {"query": query}
    d = oa("/works", {"filter": f"title_and_abstract.search:{_clean(query)},publication_year:<{year + 1}",
                      "sort": "cited_by_count:desc", "per-page": 10,
                      "select": "id,title,abstract_inverted_index,publication_year,doi,cited_by_count,referenced_works"})
    oa_t = [w.get("title") or "" for w in d.get("results", [])]
    st, body = s2("/paper/search/bulk", {"query": query, "year": f"-{year}", "sort": "citationCount:desc",
                                         "fields": "title,year,citationCount"})
    s2_t = [x.get("title") or "" for x in (body or {}).get("data") or []][:10] if st == 200 else None
    out["openalex_n"] = len(oa_t)
    out["s2_n"] = len(s2_t) if s2_t is not None else None
    out["s2_status"] = st
    out["overlap10"] = sum(any(titles_match(a, b) for b in s2_t) for a in oa_t) if s2_t else None
    out["openalex_gold"] = sum(any(titles_match(t, g) for g in gold_titles) for t in oa_t)
    out["s2_gold"] = sum(any(titles_match(t, g) for g in gold_titles) for t in s2_t) if s2_t is not None else None
    return out


def sample_check(start: str, end: str) -> dict:
    out = {"window": f"{start} to {end}"}
    d = oa("/works", {"filter": f"from_publication_date:{start},to_publication_date:{end},type:preprint,"
                                f"primary_topic.field.id:fields/17,referenced_works_count:>15,"
                                f"primary_location.source.id:S4306400194,has_abstract:true", "per-page": 1, "select": "id"})
    out["openalex_eligible"] = (d.get("meta") or {}).get("count")
    s, e = start.replace("-", ""), end.replace("-", "")
    st, body = fetch("arxiv", "https://export.arxiv.org/api/query",
                     {"search_query": f"cat:cs.* AND submittedDate:[{s}0000 TO {e}2359]", "max_results": 1},
                     gap=3.1, text=True)
    if st == 200:
        m = re.search(r"<opensearch:totalResults[^>]*>(\d+)<", body)
        out["arxiv_api_cs_in_window"] = int(m.group(1)) if m else None
    else:
        out["arxiv_api_status"] = st
    st, body = s2("/paper/search/bulk", {"query": "model", "publicationDateOrYear": f"{start}:{end}",
                                         "fieldsOfStudy": "Computer Science", "venue": "arXiv.org", "fields": "title"})
    out["s2_bulk_total"] = body.get("total") if st == 200 and body else None
    out["s2_bulk_status"] = st
    st, body = fetch("crossref", "https://api.crossref.org/works",
                     {"filter": f"from-pub-date:{start},until-pub-date:{end},type:posted-content", "sample": 20,
                      "select": "DOI,publisher"}, gap=1.1)
    if st == 200:
        items = body["message"]["items"]
        out["crossref_sample_n"] = len(items)
        out["crossref_sample_arxiv"] = sum(i["DOI"].lower().startswith("10.48550") for i in items)
        out["crossref_sample_publishers"] = Counter(i.get("publisher", "?") for i in items).most_common(4)
    else:
        out["crossref_status"] = st
    return out


# ---------- driver ----------

def main() -> None:
    _OA_USED[0] = oa_spent()
    start_used = _OA_USED[0]
    eligible = []
    for out_dir, plist in SETS:
        for p in json.loads((HERE / plist).read_text()):
            tf = HERE / out_dir / "truth" / f"{p['id']}.json"
            bib = bibliography(p["arxiv"])
            if not tf.exists() or not bib:
                continue
            gold = gold_for(json.loads(tf.read_text()), bib)
            if len(gold) >= 2:
                eligible.append((out_dir, p, bib, gold))
    rng = random.Random(SEED)
    chosen = rng.sample(eligible, min(N_PAPERS, len(eligible)))
    print(f"eligible papers {len(eligible)}, chosen {len(chosen)}", flush=True)

    res = {"papers": [], "titles": [], "references": [], "co_usage": [], "topic": [], "sample": None}
    for out_dir, p, bib, gold in chosen:
        res["papers"].append({"set": out_dir, "arxiv": p["arxiv"], "title": p["title"], "gold": [g["benchmark"] for g in gold]})
        others = [e["title"] for e in bib if not any(e["title"] == g["title"] for g in gold)]
        titles = [g["title"] for g in gold] + random.Random(p["arxiv"]).sample(others, min(len(others), max(0, TITLES_PER_PAPER - len(gold))))
        print(f"[{p['arxiv']}] titles {len(titles)}", flush=True)
        trows = title_lookups(titles)
        for r in trows:
            r["paper"], r["gold"] = p["arxiv"], any(r["title"] == g["title"] for g in gold)
        res["titles"] += trows
        print(f"[{p['arxiv']}] references", flush=True)
        res["references"].append(references(p, gold))
        # co-usage: the first two gold sources that OpenAlex and S2 both resolved
        for g in gold[:2]:
            r = next((x for x in trows if x["title"] == g["title"]), {})
            if r.get("openalex_id"):
                print(f"[{p['arxiv']}] co-usage {g['benchmark']}", flush=True)
                res["co_usage"].append(co_usage({**g, "openalex_id": r["openalex_id"], "s2_id": r.get("s2_id")}))
        blind = HERE / out_dir / "blind" / f"{p['id']}.claude-haiku-5-5.topic+model.json"
        queries = (json.loads(blind.read_text()).get("consideration") or {}).get("catalog_queries", [])[:3] if blind.exists() else []
        for q in queries:
            row = topic(q, [g["title"] for g in gold], p.get("year") or 2026)
            row["paper"] = p["arxiv"]
            res["topic"].append(row)
        json.dump({**res, "stats": STATS}, open(REPORT / "free_source_test.json", "w"), indent=1, default=str)
    print("sample check", flush=True)
    res["sample"] = sample_check("2026-10-19", "2026-10-23")
    res["stats"] = dict(STATS)
    res["latency_s"] = {k: round(sum(v) / len(v), 2) for k, v in TIMES.items() if v}
    res["openalex_usd_today"] = {"before": start_used, "after": oa_spent()}
    (REPORT / "free_source_test.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({"stats": res["stats"], "latency": res["latency_s"], "openalex_usd": res["openalex_usd_today"]}))


if __name__ == "__main__":
    main()
