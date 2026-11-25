"""Hybrid sources (24 Nov 2026): searches go to Semantic Scholar, records come from OpenAlex by id."""
from researchreadernerd import scholar

REAL_FIND_BY_TITLE = scholar.find_by_title  # captured at import, before the no_network fixture fakes them
REAL_TOPIC_WORKS = scholar.topic_works
REAL_FIND_PAPER = scholar.find_paper
REAL_S2_GET = scholar._s2_get


def oa_record(i, title, year=2024, refs=()):
    return {"id": f"https://openalex.org/W{i}", "title": title, "publication_year": year, "doi": f"https://doi.org/10.1/{i}",
            "referenced_works": [f"https://openalex.org/{r}" for r in refs]}


def test_find_by_title_reads_the_semantic_scholar_match_from_openalex_by_id(monkeypatch):
    calls = []

    def s2(path, params=None, **kw):
        calls.append(path)
        return {"data": [{"paperId": "p1", "title": "Training Verifiers to Solve Math Word Problems", "externalIds": {"ArXiv": "2110.14168"}}]}

    def oa(path, params=None, **kw):
        calls.append(path)
        return oa_record(7, "Training Verifiers to Solve Math Word Problems") if path == "/works/doi:10.48550/arXiv.2110.14168" else {}

    monkeypatch.setattr(scholar, "_s2_get", s2)
    monkeypatch.setattr(scholar, "_get", oa)
    w = REAL_FIND_BY_TITLE("Training verifiers to solve math word problems")
    assert w is not None and w.id == "W7"
    assert calls == ["/paper/search/match", "/works/doi:10.48550/arXiv.2110.14168"], "no OpenAlex search, only a lookup by id"


def test_find_by_title_rejects_a_poor_match(monkeypatch):
    monkeypatch.setattr(scholar, "_s2_get", lambda path, params=None, **kw: {"data": [{"title": "Something else entirely", "externalIds": {"DOI": "10.1/x"}}]})
    monkeypatch.setattr(scholar, "_get", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no OpenAlex call for a rejected match")))
    assert REAL_FIND_BY_TITLE("GSM8K training verifiers") is None


def test_find_by_title_falls_back_from_doi_to_arxiv_to_mag(monkeypatch):
    monkeypatch.setattr(scholar, "_s2_get", lambda path, params=None, **kw: {"data": [{"title": "A benchmark paper", "externalIds": {"DOI": "10.9/none", "ArXiv": "2401.00001", "MAG": "123"}}]})
    tried = []

    def oa(path, params=None, **kw):
        tried.append(path)
        return oa_record(3, "A benchmark paper") if path == "/works/mag:123" else {}

    monkeypatch.setattr(scholar, "_get", oa)
    assert REAL_FIND_BY_TITLE("A benchmark paper").id == "W3"
    assert tried == ["/works/doi:10.9/none", "/works/doi:10.48550/arXiv.2401.00001", "/works/mag:123"]


def test_topic_works_keeps_the_date_limit_and_skips_unmapped_papers(monkeypatch):
    seen = {}

    def s2(path, params=None, **kw):
        seen.update(params)
        return {"data": [{"title": "no ids", "externalIds": {}},
                         {"title": "old", "externalIds": {"DOI": "10.1/1"}},
                         {"title": "dated later by OpenAlex", "externalIds": {"DOI": "10.1/2"}},
                         {"title": "also old", "externalIds": {"DOI": "10.1/3"}}]}

    records = {"/works/doi:10.1/1": oa_record(1, "old", 2020), "/works/doi:10.1/2": oa_record(2, "later", 2025),
               "/works/doi:10.1/3": oa_record(3, "also old", 2019)}
    monkeypatch.setattr(scholar, "_s2_get", s2)
    monkeypatch.setattr(scholar, "_get", lambda path, params=None, **kw: records.get(path, {}))
    out = REAL_TOPIC_WORKS("face attributes (dataset)", 2, up_to_year=2021)
    assert [w.id for w in out] == ["W1", "W3"]
    assert seen["year"] == "-2021" and seen["sort"] == "citationCount:desc"
    assert "(" not in seen["query"], "search operators are removed from the query"


def test_find_paper_adds_semantic_scholar_references_openalex_did_not_link(monkeypatch):
    def oa(path, params=None, **kw):
        if path == "/works/doi:10.48550/arXiv.2608.23552":
            return oa_record(0, "Oblique retrieval under latent queries", 2026, refs=["W1"])
        if path == "/works/doi:10.1/8":
            return oa_record(8, "A source paper OpenAlex did not link")
        return {}

    def s2(path, params=None, **kw):
        assert path == "/paper/arXiv:2608.23552/references"
        return {"data": [{"citedPaper": {"title": "A referenced paper", "externalIds": {"DOI": "10.1/1"}}},
                         {"citedPaper": {"title": "A source paper OpenAlex did not link", "externalIds": {"DOI": "10.1/8"}}},
                         {"citedPaper": {"title": "Not in OpenAlex at all", "externalIds": {}}}]}

    monkeypatch.setattr(scholar, "_get", oa)
    monkeypatch.setattr(scholar, "_s2_get", s2)
    monkeypatch.setattr(scholar, "get_works", lambda ids: [scholar.Work(id="W1", title="A referenced paper", abstract="", year=2020, url="")])
    work, refs = REAL_FIND_PAPER("Oblique retrieval under latent queries", "2608.23552", arxiv_refs=False)
    assert refs == ["W1", "W8"], "OpenAlex's list first, then the Semantic Scholar entry it lacked; unmapped entries are skipped"
    assert work.references == ("W1",)


def test_a_mislinked_id_is_passed_over(monkeypatch):
    monkeypatch.setattr(scholar, "_s2_get", lambda path, params=None, **kw: {"data": [{"title": "Training Verifiers to Solve Math Word Problems",
                                                                                         "externalIds": {"DOI": "10.1/wrong", "ArXiv": "2110.14168"}}]})
    records = {"/works/doi:10.1/wrong": oa_record(5, "Socratic guiding questions in synthetic arithmetic"),
               "/works/doi:10.48550/arXiv.2110.14168": oa_record(6, "Training Verifiers to Solve Math Word Problems")}
    monkeypatch.setattr(scholar, "_get", lambda path, params=None, **kw: records.get(path, {}))
    assert REAL_FIND_BY_TITLE("Training Verifiers to Solve Math Word Problems").id == "W6"


def test_s2_retries_a_refusal_after_a_short_wait(monkeypatch):
    import types
    statuses = iter([429, 429, 200])
    waits = []

    def fake_get(url, params=None, headers=None, timeout=None):
        return types.SimpleNamespace(status_code=next(statuses), headers={}, json=lambda: {"data": [1]})

    monkeypatch.setattr(scholar.httpx2, "get", fake_get)
    monkeypatch.setattr(scholar.time, "sleep", lambda s: waits.append(s))
    monkeypatch.setattr(scholar, "S2_GAP", 0.0)
    assert REAL_S2_GET("/paper/search/match", {"query": "x"}) == {"data": [1]}
    assert len([w for w in waits if w >= 2.0]) == 2 and max(waits) <= 4.0, "two short waits of 2-4 s, no long backoff"
