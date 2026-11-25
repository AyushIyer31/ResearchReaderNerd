"""OpenAlex lookups, with a fake HTTP layer: no network, no key (Harsh Kashyap, PR #1, 21 Nov 2026; adapted to the
hybrid sources on 24 Nov: title search now goes to Semantic Scholar, stubbed to find nothing unless a test says so)."""

import pytest

from researchreadernerd import scholar
from researchreadernerd.scholar import ScholarError

# The real implementations, captured at import time: conftest's no_network fixture stubs
# these on the module, so the fixture below puts the real ones back for these tests.
_REAL = {name: getattr(scholar, name) for name in ("find_paper", "get_works", "find_by_title", "citing_works")}


@pytest.fixture
def fake_http(monkeypatch):
    """Replace scholar's httpx2.get with a scriptable fake. `responses` maps a
    (path-substring) match to a status code and JSON body, or an exception class."""
    for name, fn in _REAL.items():
        monkeypatch.setattr(scholar, name, fn)
    calls = []

    class FakeResponse:
        def __init__(self, status_code, body):
            self.status_code = status_code
            self._body = body
            self.headers = {}
            self.text = ""

        def json(self):
            return self._body

    def fake_get(url, params=None, timeout=None, **kwargs):
        calls.append({"url": url, "params": dict(params or {})})
        for i, (match, outcome) in enumerate(fake_http.responses):
            if match not in url and match not in str(params):
                continue
            if isinstance(outcome, list):  # a scripted sequence: one answer per call
                head, rest = outcome[0], outcome[1:]
                if rest:
                    fake_http.responses[i] = (match, rest)
                else:
                    fake_http.responses.pop(i)
                outcome = head
            if isinstance(outcome, type) and issubclass(outcome, Exception):
                raise outcome("boom")
            status, body = outcome
            return FakeResponse(status, body)
        return FakeResponse(200, {})

    fake_http.responses = []
    fake_http.calls = calls
    monkeypatch.setattr(scholar.httpx2, "get", fake_get)
    monkeypatch.setattr(scholar.time, "sleep", lambda s: None)
    return fake_http


def work_json(i, title, refs=()):
    return {
        "id": f"https://openalex.org/W{i}",
        "title": title,
        "abstract_inverted_index": {"word": [0]},
        "publication_year": 2020,
        "doi": f"https://doi.org/10.0/{i}",
        "cited_by_count": 10,
        "referenced_works": [f"https://openalex.org/{r}" for r in refs],
    }


def test_clean_unquotes_before_stripping_filter_syntax():
    # A percent-encoded comma or colon must not survive into the filter value:
    # OpenAlex would read it as filter syntax and reject the request.
    cleaned = scholar._clean("Attention%3A Is All%2C You Need | Test")
    assert not any(c in cleaned for c in ",:|")
    assert cleaned == "Attention Is All You Need Test"


def test_clean_plain_title_unchanged():
    assert scholar._clean("  ImageNet Classification with Deep CNNs  ") == "ImageNet Classification with Deep CNNs"


def test_abstract_rebuilt_from_inverted_index():
    inverted = {"the": [0, 3], "quick": [1], "fox": [2], "dog": [4]}
    assert scholar._abstract(inverted) == "the quick fox the dog"
    assert scholar._abstract(None) == ""
    assert scholar._abstract({}) == ""


def test_titles_match_tolerates_case_and_punctuation():
    assert scholar.titles_match("Attention Is All You Need!", "attention is all you need")
    assert not scholar.titles_match("Attention Is All You Need", "BERT: Pre-training of Deep Transformers")
    assert not scholar.titles_match("", "")
    # Stop words alone must not count as a match.
    assert not scholar.titles_match("The", "the")


def test_get_retries_then_succeeds(fake_http):
    fake_http.responses = [("/works", [(500, {}), (503, {}), (200, {"ok": 1})])]
    assert scholar._get("/works") == {"ok": 1}
    assert len(fake_http.calls) == 3


def test_get_raises_after_retries_exhausted(fake_http):
    fake_http.responses = [("/works", (503, {}))]
    with pytest.raises(ScholarError, match="HTTP 503"):
        scholar._get("/works", retries=2)
    assert len(fake_http.calls) == 3  # initial try + 2 retries


def test_get_non_retryable_status_fails_fast(fake_http):
    fake_http.responses = [("/works", (400, {}))]
    with pytest.raises(ScholarError, match="HTTP 400"):
        scholar._get("/works")
    assert len(fake_http.calls) == 1


def test_get_404_returns_empty(fake_http):
    fake_http.responses = [("/works/doi:", (404, {}))]
    assert scholar._get("/works/doi:10.48550/arXiv.1706.03762") == {}


def test_get_sends_api_key_and_mailto(fake_http, monkeypatch):
    monkeypatch.setenv("OPENALEX_API_KEY", "key-1")
    monkeypatch.setenv("OPENALEX_MAILTO", "me@example.com")
    fake_http.responses = [("/works", (200, {"results": []}))]
    scholar._get("/works")
    assert fake_http.calls[0]["params"]["api_key"] == "key-1"
    assert fake_http.calls[0]["params"]["mailto"] == "me@example.com"


def test_find_paper_strips_arxiv_version(fake_http):
    fake_http.responses = [("/works/doi:", (200, work_json(7, "A Paper", ["W8"])))]
    found = scholar.find_paper("ignored", "1706.03762v3")
    assert "arXiv.1706.03762" in fake_http.calls[0]["url"]
    assert "v3" not in fake_http.calls[0]["url"]
    work, refs = found
    assert work.id == "W7" and refs == ["W8"]


def test_find_paper_falls_back_to_semantic_scholar_title_match(fake_http, monkeypatch):
    # No OpenAlex record by arXiv DOI: the title is matched in Semantic Scholar and read from OpenAlex by its DOI.
    monkeypatch.setattr(scholar, "_s2_get", lambda path, params=None, **kw: (
        {"data": [{"paperId": "p9", "title": "Attention Is All You Need", "externalIds": {"DOI": "10.0/9"}}]}
        if path == "/paper/search/match" else {}))
    fake_http.responses = [
        ("/works/doi:10.48550", (404, {})),
        ("/works/doi:10.0/9", (200, work_json(9, "Attention Is All You Need", []))),
    ]
    found = scholar.find_paper("attention is all you need", "9999.99999")
    assert found is not None and found[0].id == "W9"
    assert not any("title.search" in str(c["params"]) for c in fake_http.calls), "no OpenAlex search"


def test_find_paper_returns_none_without_match(fake_http):
    fake_http.responses = [("", (200, {"results": [work_json(9, "Something Else Entirely", [])]}))]
    assert scholar.find_paper("attention is all you need") is None


def test_get_works_batches_by_50(fake_http):
    fake_http.responses = [("/works", (200, {"results": [work_json(1, "One")]}))]
    ids = [f"W{i}" for i in range(120)]
    works = scholar.get_works(ids)
    assert len(fake_http.calls) == 3  # 50 + 50 + 20
    assert all(len(w.references) == 0 for w in works)


def test_citing_works_parses_results(fake_http):
    fake_http.responses = [("cites:", (200, {"results": [work_json(1, "Citer", ["W2"]), work_json(2, "Citer2")]}))]
    works = scholar.citing_works("W7", limit=5)
    assert [w.title for w in works] == ["Citer", "Citer2"]
    assert works[0].references == ("W2",)
    assert "cites:W7" in fake_http.calls[0]["params"]["filter"]
