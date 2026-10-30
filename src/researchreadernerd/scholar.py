"""OpenAlex lookups: a paper's references, papers citing a work, and title search.

OpenAlex is free and needs no key for this volume (about 15-20 requests per analysis);
set OPENALEX_API_KEY or OPENALEX_MAILTO for higher limits.
"""

from __future__ import annotations

import os
import re
import time
import urllib.parse
from dataclasses import dataclass

import httpx2

API = "https://api.openalex.org"
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


def _get(path: str, params: dict | None = None, *, retries: int = 4) -> dict:
    params = dict(params or {})
    if os.environ.get("OPENALEX_API_KEY"):
        params["api_key"] = os.environ["OPENALEX_API_KEY"]
    if os.environ.get("OPENALEX_MAILTO"):
        params["mailto"] = os.environ["OPENALEX_MAILTO"]
    delay = 1.0
    for attempt in range(retries + 1):
        try:
            response = httpx2.get(f"{API}{path}", params=params, timeout=30)
        except httpx2.HTTPError as e:
            error = str(e)
        else:
            if response.status_code == 200:
                return response.json()
            if response.status_code == 404:
                return {}
            error = f"HTTP {response.status_code}"
            if response.status_code not in (429, 500, 502, 503, 504):
                break
        if attempt < retries:
            time.sleep(delay)
            delay *= 2
    raise ScholarError(f"OpenAlex request failed ({error}).")


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


def find_paper(title: str, arxiv_id: str | None = None) -> tuple[Work, list[str]] | None:
    """The paper itself, plus the OpenAlex ids of the works it references."""
    raw = None
    if arxiv_id:
        bare = re.sub(r"v\d+$", "", arxiv_id)
        raw = _get(f"/works/doi:10.48550/arXiv.{bare}", {"select": _WITH_REFS}) or None
    if raw is None and title:
        found = _get("/works", {"filter": f"title.search:{_clean(title)}", "per-page": 3,
                                "select": _WITH_REFS}).get("results", [])
        raw = next((w for w in found if titles_match(w.get("title") or "", title)), None)
    if raw is None:
        return None
    work = _work(raw)
    return work, list(work.references)


def get_works(ids: list[str]) -> list[Work]:
    """The works with these ids, each with its bibliography."""
    out: list[Work] = []
    for start in range(0, len(ids), 50):
        batch = "|".join(ids[start : start + 50])
        data = _get("/works", {"filter": f"openalex_id:{batch}", "per-page": 50, "select": _WITH_REFS})
        out += [_work(w) for w in data.get("results", [])]
    return out


def find_by_title(title: str) -> Work | None:
    """The most-cited work whose title matches, e.g. a benchmark's source paper."""
    found = _get("/works", {"filter": f"title.search:{_clean(title)}", "sort": "cited_by_count:desc",
                            "per-page": 5, "select": _FIELDS}).get("results", [])
    match = next((w for w in found if titles_match(w.get("title") or "", title)), None)
    return _work(match) if match else None


def citing_works(work_id: str, limit: int = 20) -> list[Work]:
    """The most-cited papers that cite `work_id`, each with its bibliography."""
    data = _get("/works", {"filter": f"cites:{work_id}", "sort": "cited_by_count:desc",
                           "per-page": limit, "select": _WITH_REFS})
    return [_work(w) for w in data.get("results", [])]


def _clean(title: str) -> str:
    # Commas, colons and pipes have meaning in OpenAlex filter values.
    return urllib.parse.unquote(re.sub(r"[,:|]+", " ", title)).strip()
