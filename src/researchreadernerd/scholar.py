"""Scholarly lookups: a paper's references, papers citing a work, title search and topic search.

Hybrid sources (version 2, 24 Nov 2026; free-source comparison in experiments/free_source_test.md): every search goes
to Semantic Scholar (title match, topic search, a paper's reference list), and each paper found there is read from
OpenAlex by its DOI, arXiv id or MAG id, a lookup by id that OpenAlex does not charge for. OpenAlex keeps only its
list queries (works by id in batches, and the papers citing a work, sorted by citations), which fit its free daily
budget. Every Work is therefore an OpenAlex record, as before: callers, caches and the leave-one-out withholding see
OpenAlex ids only. Set S2_API_KEY for Semantic Scholar's keyed rate (1 request a second), OPENALEX_API_KEY for OpenAlex.
"""

from __future__ import annotations

import json
import os
import random
import re
import threading
import time
import urllib.parse
from dataclasses import dataclass

import httpx2

API = "https://api.openalex.org"
S2_API = "https://api.semanticscholar.org/graph/v1"
S2_GAP = float(os.environ.get("RRN_S2_GAP", "1.05"))  # seconds between Semantic Scholar calls (keyed limit: 1 a second)
_S2_LOCK = threading.Lock()
_S2_LAST = [0.0]
MAX_ID_LOOKUPS = 120  # Semantic Scholar references read from OpenAlex by id, per paper (free lookups)
_FIELDS = "id,title,abstract_inverted_index,publication_year,doi,cited_by_count"
_WITH_REFS = _FIELDS + ",referenced_works"


@dataclass(frozen=True)
class Work:
    id: str  # short OpenAlex id, e.g. W2626778328
    title: str
    abstract: str
    year: int | None
    url: str  # DOI link when there is one, else the OpenAlex page
    cited_by: int = 0
    references: tuple[str, ...] = ()  # short ids of the works it cites (its bibliography)


class ScholarError(Exception):
    """OpenAlex could not be reached or kept refusing requests."""


FREE_GUARD_MARGIN = 0.02  # free budget kept in reserve by the request guard
_FREE = {"read_at": 0.0, "remaining": None, "since": 0.0}
_FREE_LOCK = threading.Lock()


def _price(path: str, params: dict) -> float:
    """OpenAlex's price for a request (rate-limit card of 13 Nov 2026): a work by id is free, a filter list $0.0001, a
    search (the search parameter or a .search filter) $0.001."""
    if "search" in params or ".search:" in str(params.get("filter", "")):
        return 0.001
    return 0.0 if path.startswith("/works/") else 0.0001


class FreeBudgetSpent(Exception):
    """RRN_OPENALEX_FREE_ONLY: the free daily budget cannot cover this request; it was not sent."""


def _free_guard(path: str, params: dict) -> None:
    """RRN_OPENALEX_FREE_ONLY=1 (launch runner): before each priced request, refuse it unless OpenAlex's free daily budget,
    as last read from the rate-limit endpoint less what this process has spent since, covers it with a margin; refuse
    everything if the key holds prepaid credit (the launch key holds none, so OpenAlex itself cannot bill beyond the free
    budget). The budget is re-read every 20 s and whenever 2 cents have been spent since the last read."""
    if os.environ.get("RRN_OPENALEX_FREE_ONLY") != "1":
        return
    price = _price(path, params)
    if price == 0.0:
        return
    with _FREE_LOCK:
        if _FREE["remaining"] is None or time.time() - _FREE["read_at"] > 20 or _FREE["since"] >= 0.02:
            try:
                r = httpx2.get(f"{API}/rate-limit", params={"api_key": os.environ.get("OPENALEX_API_KEY", "")}, timeout=20).json()["rate_limit"]
            except Exception as e:  # noqa: BLE001  fail closed
                raise FreeBudgetSpent(f"OpenAlex free budget could not be read ({type(e).__name__}); request not sent") from e
            if (r.get("prepaid_remaining_usd") or 0) > 0:
                raise FreeBudgetSpent("the OpenAlex key holds prepaid credit; the launch runs only with a key that holds none")
            _FREE.update(read_at=time.time(), remaining=float(r["daily_remaining_usd"]), since=0.0)
        if _FREE["remaining"] - _FREE["since"] < price + FREE_GUARD_MARGIN:
            raise FreeBudgetSpent(f"OpenAlex free daily budget spent (${_FREE['remaining'] - _FREE['since']:.3f} left); request not sent")
        _FREE["since"] += price


def _get(path: str, params: dict | None = None, *, retries: int = 6) -> dict:
    params = dict(params or {})
    try:
        _free_guard(path, params)
    except FreeBudgetSpent as e:
        _ledger(path, params, kind="free_guard_refused", detail=str(e)[:120])
        raise ScholarError(f"OpenAlex: {e}.") from e
    if os.environ.get("OPENALEX_API_KEY"):
        params["api_key"] = os.environ["OPENALEX_API_KEY"]
    if os.environ.get("OPENALEX_MAILTO"):
        params["mailto"] = os.environ["OPENALEX_MAILTO"]
    delay = 2.0
    for attempt in range(retries + 1):
        wait = delay
        try:
            response = httpx2.get(f"{API}{path}", params=params, timeout=30)
        except httpx2.HTTPError as e:
            error = str(e)
        else:
            if response.status_code == 200:
                _ledger(path, params)
                return response.json()
            if response.status_code == 404:
                return {}
            error = f"HTTP {response.status_code}"
            if response.status_code not in (429, 500, 502, 503, 504):
                break
            retry_after = response.headers.get("Retry-After")  # OpenAlex says how long to back off when rate-limited
            if retry_after and retry_after.isdigit():
                wait = max(wait, float(retry_after))
                if wait > 300:  # the daily quota is spent; waiting is pointless
                    hours = wait / 3600
                    raise ScholarError(f"OpenAlex daily request quota exhausted; it resets in {hours:.1f} h.")
        if attempt < retries:
            time.sleep(min(wait, 60))
            delay *= 2
    raise ScholarError(f"OpenAlex request failed ({error}).")


def _s2_get(path: str, params: dict | None = None, *, retries: int = 20) -> dict:
    """A Semantic Scholar Graph API response ({} when it has no such paper). Calls are spaced S2_GAP seconds apart
    across threads. A 429 (Semantic Scholar refuses part of the calls even below the keyed rate) is retried after a
    short wait of 2-4 s, or Retry-After when given; a 5xx or network error backs off exponentially up to 30 s. Raises
    ScholarError after `retries` refusals in a row."""
    params = dict(params or {})
    headers = {"x-api-key": os.environ["S2_API_KEY"]} if os.environ.get("S2_API_KEY") else {}
    backoff = 2.0
    error = "no response"
    for attempt in range(retries + 1):
        with _S2_LOCK:
            wait = S2_GAP - (time.time() - _S2_LAST[0])
            if wait > 0:
                time.sleep(wait)
            try:
                response = httpx2.get(f"{S2_API}{path}", params=params, headers=headers, timeout=40)
            except httpx2.HTTPError as e:
                response, error = None, str(e)
            _S2_LAST[0] = time.time()
        if response is not None:
            if response.status_code == 200:
                _ledger(path, params, kind="s2")
                return response.json()
            if response.status_code == 404:
                _ledger(path, params, kind="s2")
                return {}
            error = f"HTTP {response.status_code}"
            if response.status_code not in (429, 500, 502, 503, 504):
                break
        if attempt == retries:
            break
        retry_after = response.headers.get("Retry-After") if response is not None else None
        if response is not None and response.status_code == 429:
            pause = float(retry_after) if retry_after and retry_after.isdigit() else 2.0 + random.random() * 2.0
        else:
            pause, backoff = backoff, min(backoff * 2, 30.0)
        time.sleep(min(pause, 60))
    raise ScholarError(f"Semantic Scholar request failed ({error}).")


def _s2_clean(text: str) -> str:
    # + | - " * ( ) ~ are operators in Semantic Scholar's search syntax.
    return re.sub(r"\s+", " ", re.sub(r'[+|\-"*()~]+', " ", text)).strip()


def _openalex_by_ids(ext: dict, title: str = "") -> dict | None:
    """The OpenAlex record (with its bibliography) of a Semantic Scholar paper, by DOI, then arXiv id, then MAG id:
    lookups by id, which OpenAlex does not charge for. A record whose title does not match `title` is passed over
    (an id can point to another work: OpenAlex resolves some arXiv DOIs to the wrong record). When no id gives a
    record with that title, OpenAlex's title search is the fallback (a search query, $0.001: the only searches left on
    OpenAlex, counted in the ledger as search_filter). None when that fails too."""
    idents = []
    if ext.get("DOI"):
        idents.append(f"doi:{ext['DOI']}")
    if ext.get("ArXiv"):
        idents.append(f"doi:10.48550/arXiv.{ext['ArXiv']}")
    if ext.get("MAG"):
        idents.append(f"mag:{ext['MAG']}")
    for ident in idents:
        raw = _get(f"/works/{ident}", {"select": _WITH_REFS})
        if raw and raw.get("id") and (not title or titles_match(raw.get("title") or "", title)):
            return raw
    if title and _clean(title):
        found = _get("/works", {"filter": f"title.search:{_clean(title)}", "sort": "cited_by_count:desc", "per-page": 5,
                                "select": _WITH_REFS}).get("results", [])
        return next((w for w in found if titles_match(w.get("title") or "", title)), None)
    return None


def _ledger(path: str, params: dict, kind: str | None = None, detail: str | None = None) -> None:
    """RRN_OPENALEX_LEDGER=<file> (launch runner): one JSON line per billable OpenAlex response, with its endpoint class
    (singleton: a work by id; list: a filter; search_filter: a filter on title.search or title_and_abstract.search;
    search: the search parameter; s2: a Semantic Scholar call, free; s2_unmapped: a Semantic Scholar paper OpenAlex has
    no record for), for the per-step endpoint worksheet. Unset, nothing is written."""
    ledger = os.environ.get("RRN_OPENALEX_LEDGER")
    if not ledger:
        return
    filt = str(params.get("filter", ""))
    kind = kind or ("search" if "search" in params else "search_filter" if ".search:" in filt else "list" if path == "/works" or "filter" in params
                    else "singleton")
    from datetime import datetime, timezone
    line = {"t": datetime.now(timezone.utc).isoformat(timespec="seconds"), "path": path, "class": kind} | ({"detail": detail} if detail else {})
    with open(ledger, "a") as f:
        f.write(json.dumps(line) + "\n")


def _abstract(inverted: dict | None) -> str:
    """OpenAlex stores abstracts as {word: [positions]}; rebuild the text."""
    if not inverted:
        return ""
    words = sorted((pos, word) for word, positions in inverted.items() for pos in positions)
    return " ".join(word for _, word in words)


def _work(raw: dict) -> Work:
    short_id = raw["id"].rsplit("/", 1)[-1]
    return Work(
        id=short_id,
        title=raw.get("title") or "",
        abstract=_abstract(raw.get("abstract_inverted_index")),
        year=raw.get("publication_year"),
        url=raw.get("doi") or f"https://openalex.org/{short_id}",
        cited_by=raw.get("cited_by_count") or 0,
        references=tuple(w.rsplit("/", 1)[-1] for w in raw.get("referenced_works") or []),
    )


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower())) - {"the", "a", "an", "of", "and", "for", "in", "on", "to", "with"}


def titles_match(a: str, b: str) -> bool:
    """True when two titles share most of their words (tolerates punctuation and case)."""
    wa, wb = _words(a), _words(b)
    return bool(wa and wb) and len(wa & wb) / min(len(wa), len(wb)) >= 0.8


ARXIV_HTML = "https://arxiv.org/html/{id}"
MAX_ARXIV_REF_LOOKUPS = 40  # bibliography entries OpenAlex did not list that are looked up by title, per paper
_BIBITEM = re.compile(r'<li[^>]*class="[^"]*ltx_bibitem[^"]*"[^>]*>(.*?)</li>', re.S)
_BIBBLOCK = re.compile(r'<span[^>]*class="[^"]*ltx_bibblock[^"]*"[^>]*>(.*?)</span>', re.S)
_TAG = re.compile(r"<[^>]+>")


_ARXIV_HTML: dict[str, str] = {}  # per process: a leave-one-out harness looks the same paper up once per benchmark
_ARXIV_LAST = [0.0]


def _fetch_arxiv_html(arxiv_id: str) -> str:
    """The arXiv HTML rendering of a paper, or '' when there is none (older papers, or arXiv unreachable). Fetched at
    most once per paper per process, at least a second apart; a missing page or a failed fetch is written to the
    OpenAlex ledger (classes arxiv_html_none, arxiv_html_failed) so that a run where change 3 did not apply is visible."""
    if arxiv_id in _ARXIV_HTML:
        return _ARXIV_HTML[arxiv_id]
    wait = 1.0 - (time.time() - _ARXIV_LAST[0])
    if wait > 0:
        time.sleep(wait)
    try:
        r = httpx2.get(ARXIV_HTML.format(id=arxiv_id), headers={"User-Agent": "ResearchReaderNerd/0.1"},
                       follow_redirects=True, timeout=30)
    except httpx2.HTTPError as e:
        _ledger(arxiv_id, {}, kind="arxiv_html_failed", detail=type(e).__name__)
        return ""
    finally:
        _ARXIV_LAST[0] = time.time()
    if r.status_code == 200:
        _ARXIV_HTML[arxiv_id] = r.text
        return r.text
    _ledger(arxiv_id, {}, kind="arxiv_html_none" if r.status_code == 404 else "arxiv_html_failed", detail=str(r.status_code))
    if r.status_code == 404:
        _ARXIV_HTML[arxiv_id] = ""
    return ""


def arxiv_bibliography(arxiv_id: str) -> list[str]:
    """Titles of the entries in the paper's own bibliography, read from its arXiv HTML rendering. OpenAlex links
    a 2026 preprint's references with a lag, and the benchmark papers it cites are the newest ones, so the
    authors' own list is the only complete source (11 Nov 2026: OpenAlex listed 29 of Prime Agent's 45+)."""
    html = _fetch_arxiv_html(re.sub(r"v\d+$", "", arxiv_id))
    titles: list[str] = []
    for item in _BIBITEM.findall(html):
        blocks = [re.sub(r"\s+", " ", _TAG.sub(" ", b)).strip(" .") for b in _BIBBLOCK.findall(item)]
        blocks = [b for b in blocks if b]
        if not blocks:
            continue
        # bibtex styles render author list, title, venue as successive blocks; a one-block entry is the whole line
        title = blocks[1] if len(blocks) >= 2 else blocks[0]
        title = _html_unescape(title)
        if len(title.split()) >= 3:
            titles.append(title)
    return titles


def _html_unescape(text: str) -> str:
    import html as _html
    return _html.unescape(text)


def find_paper(title: str, arxiv_id: str | None = None, *, arxiv_refs: bool = True) -> tuple[Work, list[str]] | None:
    """The paper itself, plus the OpenAlex ids of the works it references: OpenAlex's list, then the entries of
    Semantic Scholar's reference list that OpenAlex did not link (read from OpenAlex by id), then any entry of the
    paper's own arXiv bibliography still missing that has a record (looked up by title)."""
    raw = None
    bare = re.sub(r"v\d+$", "", arxiv_id) if arxiv_id else None
    if bare:
        raw = _get(f"/works/doi:10.48550/arXiv.{bare}", {"select": _WITH_REFS}) or None
    s2_id = f"arXiv:{bare}" if bare else None
    if raw is None and title:
        m = _s2_match(title)
        if m:
            raw = _openalex_by_ids(m.get("externalIds") or {}, m.get("title") or "")
            s2_id = m.get("paperId")
    if raw is None:
        return None
    work = _work(raw)
    refs = list(work.references)
    known = [w.title for w in get_works(refs)] + [work.title] if refs or s2_id or (bare and arxiv_refs) else []
    if s2_id is None and work.url.startswith("https://doi.org/"):
        s2_id = "DOI:" + work.url.removeprefix("https://doi.org/")
    if s2_id:
        data = _s2_get(f"/paper/{s2_id}/references", {"fields": "title,externalIds", "limit": 1000})
        looked = 0
        for item in data.get("data") or []:
            cited = item.get("citedPaper") or {}
            t = cited.get("title") or ""
            if not t or any(titles_match(t, k) for k in known):
                continue
            if looked >= MAX_ID_LOOKUPS:
                break
            looked += 1
            r = _openalex_by_ids(cited.get("externalIds") or {}, t)
            if r is None:
                _ledger("/paper/references", {}, kind="s2_unmapped")
                continue
            w = _work(r)
            if w.id != work.id and w.id not in refs:
                refs.append(w.id)
                known.append(w.title)
    if bare and arxiv_refs:
        bib = arxiv_bibliography(bare)
        if bib:
            looked = 0
            for t in bib:
                if any(titles_match(t, k) for k in known):
                    continue
                if looked >= MAX_ARXIV_REF_LOOKUPS:
                    break
                looked += 1
                w = find_by_title(t)
                if w and w.id != work.id and w.id not in refs:
                    refs.append(w.id)
                    known.append(w.title)
    return work, refs


def get_works(ids: list[str]) -> list[Work]:
    """The works with these ids, each with its bibliography."""
    out: list[Work] = []
    for start in range(0, len(ids), 50):
        batch = "|".join(ids[start : start + 50])
        data = _get("/works", {"filter": f"openalex_id:{batch}", "per-page": 50, "select": _WITH_REFS})
        out += [_work(w) for w in data.get("results", [])]
    return out


def _s2_match(title: str) -> dict | None:
    """Semantic Scholar's best title match, if its title really matches."""
    q = _s2_clean(title)
    if not q:
        return None
    data = _s2_get("/paper/search/match", {"query": q, "fields": "title,externalIds"})
    return next((c for c in data.get("data") or [] if titles_match(c.get("title") or "", title)), None)


def find_by_title(title: str) -> Work | None:
    """The work whose title matches, e.g. a benchmark's source paper: found by Semantic Scholar's title match, read
    from OpenAlex by id."""
    m = _s2_match(title)
    if m is None:
        return None
    raw = _openalex_by_ids(m.get("externalIds") or {}, title)
    if raw is None:
        _ledger("/paper/search/match", {}, kind="s2_unmapped")
        return None
    return _work(raw)


def citing_works(work_id: str, limit: int = 20) -> list[Work]:
    """The most-cited papers that cite `work_id`, each with its bibliography."""
    data = _get("/works", {"filter": f"cites:{work_id}", "sort": "cited_by_count:desc",
                           "per-page": limit, "select": _WITH_REFS})
    return [_work(w) for w in data.get("results", [])]


def topic_works(query: str, limit: int = 10, *, up_to_year: int | None = None) -> list[Work]:
    """The most-cited papers matching `query` in title or abstract (Semantic Scholar bulk search, sorted by citations,
    published up to `up_to_year`), each read from OpenAlex by id with its bibliography. Papers OpenAlex has no record
    for are skipped; at most 3 x limit are looked at."""
    q = _s2_clean(query)
    if not q:
        return []
    params = {"query": q, "sort": "citationCount:desc", "fields": "title,externalIds,year"}
    if up_to_year:
        params["year"] = f"-{up_to_year}"
    data = _s2_get("/paper/search/bulk", params)
    out: list[Work] = []
    for p in (data.get("data") or [])[: 3 * limit]:
        if len(out) >= limit:
            break
        raw = _openalex_by_ids(p.get("externalIds") or {}, p.get("title") or "")
        if raw is None:
            _ledger("/paper/search/bulk", {}, kind="s2_unmapped")
            continue
        w = _work(raw)
        if up_to_year and w.year and w.year > up_to_year:
            continue  # OpenAlex dates the record later than Semantic Scholar did (e.g. the published version)
        if w.id not in {x.id for x in out}:
            out.append(w)
    return out


def _clean(title: str) -> str:
    # Commas, colons and pipes have meaning in OpenAlex filter values.
    # Unquote first: a percent-encoded comma, colon or pipe would otherwise survive the cleanup and land inside the
    # filter value, where OpenAlex reads it as syntax (Harsh Kashyap, PR #1).
    return " ".join(re.sub(r"[,:|]+", " ", urllib.parse.unquote(title)).split())
