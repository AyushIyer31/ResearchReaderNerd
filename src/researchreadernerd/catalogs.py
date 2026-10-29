"""Search public benchmark and dataset catalogs by keyword.

Each search returns CatalogHit records with a link to the dataset page. A catalog that
fails (network error, changed API) returns nothing and is reported, never fatal.
"""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass
from functools import lru_cache

import httpx2

TIMEOUT = 20


@dataclass(frozen=True)
class CatalogHit:
    catalog: str
    name: str
    description: str
    url: str


# TPC publishes a fixed, small set of benchmarks (tpc.org, active list as of Oct 2026).
TPC_BENCHMARKS = [
    ("TPC-C", "online transaction processing (OLTP), order-entry workload with concurrent transactions", "tpcc"),
    ("TPC-E", "online transaction processing (OLTP), brokerage-firm workload", "tpce"),
    ("TPC-H", "decision support, ad-hoc business queries and concurrent data modifications", "tpch"),
    ("TPC-DS", "decision support, queries and data maintenance on large data volumes, big data systems", "tpcds"),
    ("TPC-DI", "data integration, ETL from varied sources into a unified model", "tpcdi"),
    ("TPCx-BB", "big data analytics, 30 queries over Hadoop-based systems in a retail context", "tpcx-bb"),
    ("TPCx-HS", "big data, Hadoop-compatible sort performance and availability", "tpcx-hs"),
    ("TPCx-V", "virtualized database server performance", "tpcx-v"),
    ("TPCx-HCI", "hyper-converged infrastructure, virtualized database workloads", "tpcx-hci"),
    ("TPCx-IoT", "Internet of Things data ingestion and queries", "tpcx-iot"),
    ("TPCx-AI", "end-to-end artificial intelligence and machine learning pipelines", "tpcx-ai"),
]


def _terms(query: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", query.lower()) if len(w) > 2}


def _matches(query: str, text: str) -> bool:
    terms = _terms(query)
    words = _terms(text)
    return bool(terms) and len(terms & words) >= max(1, (len(terms) + 1) // 2)


def search_tpc(query: str, limit: int = 5) -> list[CatalogHit]:
    return [
        CatalogHit("TPC", name, desc, f"https://www.tpc.org/{slug}/")
        for name, desc, slug in TPC_BENCHMARKS
        if _matches(query, f"{name} {desc}")
    ][:limit]


@dataclass(frozen=True)
class _UcrDataset:
    name: str
    type: str  # Image, Sensor, Motion, ECG, ... (15 types in the 2018 archive)
    summary: str
    size: int  # train + test series


@lru_cache(maxsize=1)
def _ucr_rows() -> tuple[_UcrDataset, ...]:
    """Every dataset in the UCR Time Series Classification Archive (128 in the 2018 release)."""
    text = httpx2.get("https://www.cs.ucr.edu/~eamonn/time_series_data_2018/DataSummary.csv", timeout=TIMEOUT).text
    rows = []
    for row in csv.DictReader(io.StringIO(text.lstrip("\ufeff"))):
        row = {k.strip(): (v or "").strip() for k, v in row.items() if k}
        if not row.get("Name"):
            continue
        size = sum(int(row[k]) for k in ("Train", "Test") if row.get(k, "").isdigit())
        summary = f"{row.get('Type', '')} time series; {row.get('Class', '?')} classes, length {row.get('Length', '?')}"
        rows.append(_UcrDataset(row["Name"], row.get("Type", ""), summary, size))
    return tuple(rows)


# Words every UCR dataset satisfies: on their own they would match all 128 datasets.
_UCR_GENERIC = {"time", "series", "classification", "classify", "data", "dataset", "datasets", "archive",
                "univariate", "benchmark", "benchmarks", "temporal", "sequence", "sequences", "ucr"}

# Query words that point to a UCR dataset type.
_UCR_TYPE_WORDS = {
    "accelerometer": "motion", "gesture": "motion", "activity": "motion", "movement": "motion",
    "heart": "ecg", "cardiac": "ecg", "electrocardiogram": "ecg",
    "appliance": "device", "appliances": "device", "electricity": "device", "energy": "power",
    "spectroscopy": "spectro", "spectrograph": "spectro", "spectral": "spectrum",
    "gps": "trajectory", "eye": "eog", "insect": "epg", "blood": "hemodynamics", "sensors": "sensor",
}


def search_ucr(query: str, limit: int = 5) -> list[CatalogHit]:
    """UCR datasets whose name or type matches a specific query term, best matches first.

    Generic words ("time series classification") are ignored, so a query made only of
    them returns nothing rather than an arbitrary handful of the 128 datasets.
    """
    terms = _terms(query) - _UCR_GENERIC
    types = {_UCR_TYPE_WORDS.get(t, t) for t in terms}
    scored = []
    for d in _ucr_rows():
        name = d.name.lower()
        score = 2 * sum(t in name for t in terms) + (d.type.lower() in types)
        if score:
            scored.append((score, d.size, d))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [
        CatalogHit("UCR Time Series Archive", d.name, d.summary,
                   f"https://www.timeseriesclassification.com/description.php?Dataset={d.name}")
        for _, _, d in scored[:limit]
    ]


def search_uci(query: str, limit: int = 5) -> list[CatalogHit]:
    data = httpx2.get("https://archive.ics.uci.edu/api/datasets/list", params={"search": query}, timeout=TIMEOUT).json()
    return [
        CatalogHit("UCI Machine Learning Repository", d["name"], "", f"https://archive.ics.uci.edu/dataset/{d['id']}")
        for d in (data.get("data") or [])[:limit]
    ]


def search_huggingface(query: str, limit: int = 5) -> list[CatalogHit]:
    data = httpx2.get("https://huggingface.co/api/datasets",
                      params={"search": query, "limit": limit, "sort": "downloads"}, timeout=TIMEOUT).json()
    return [CatalogHit("Hugging Face Datasets", d["id"], "", f"https://huggingface.co/datasets/{d['id']}") for d in data]


def search_data_gov(query: str, limit: int = 5) -> list[CatalogHit]:
    data = httpx2.get("https://catalog.data.gov/search", params={"q": query, "format": "json"}, timeout=TIMEOUT).json()
    return [
        CatalogHit("data.gov (US)", r.get("title", ""), (r.get("description") or "")[:200],
                   f"https://catalog.data.gov/dataset/{r['slug']}")
        for r in (data.get("results") or [])[:limit]
        if r.get("slug")
    ]


def search_data_gov_uk(query: str, limit: int = 5) -> list[CatalogHit]:
    data = httpx2.get("https://ckan.publishing.service.gov.uk/api/action/package_search",
                      params={"q": query, "rows": limit}, timeout=TIMEOUT).json()
    return [
        CatalogHit("data.gov.uk", r.get("title", ""), (r.get("notes") or "")[:200],
                   f"https://www.data.gov.uk/dataset/{r['id']}/{r['name']}")
        for r in data.get("result", {}).get("results", [])
    ]


def search_data_europa(query: str, limit: int = 5) -> list[CatalogHit]:
    data = httpx2.get("https://data.europa.eu/api/hub/search/search",
                      params={"q": query, "limit": limit, "filter": "dataset"}, timeout=TIMEOUT).json()
    hits = []
    for r in data.get("result", {}).get("results", []):
        title = (r.get("title") or {}).get("en") or next(iter((r.get("title") or {}).values()), "")
        desc = (r.get("description") or {}).get("en") or ""
        hits.append(CatalogHit("data.europa.eu", title, desc[:200], f"https://data.europa.eu/data/datasets/{r['id']}"))
    return hits


SEARCHES = {
    "TPC": search_tpc,
    "UCR Time Series Archive": search_ucr,
    "UCI Machine Learning Repository": search_uci,
    "Hugging Face Datasets": search_huggingface,
    "data.gov (US)": search_data_gov,
    "data.gov.uk": search_data_gov_uk,
    "data.europa.eu": search_data_europa,
}


def search_all(queries: list[str], per_query: int = 4) -> tuple[list[CatalogHit], list[str]]:
    """Run every query against every catalog. Returns (unique hits, notes about failed catalogs)."""
    hits: dict[tuple[str, str], CatalogHit] = {}
    failed: set[str] = set()
    for catalog, search in SEARCHES.items():
        for query in queries:
            try:
                for hit in search(query, per_query):
                    hits.setdefault((hit.catalog, hit.name.lower()), hit)
            except Exception:  # one catalog being down must not stop the others
                failed.add(catalog)
    notes = [f"{c} could not be searched this time." for c in sorted(failed)]
    return list(hits.values()), notes
