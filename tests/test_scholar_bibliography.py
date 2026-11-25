"""The paper's own arXiv bibliography fills in references OpenAlex has not linked (11 Nov 2026)."""
import dataclasses

from researchreadernerd import scholar

REAL_FIND_PAPER = scholar.find_paper  # captured at import, before the no_network fixture fakes it

HTML = """
<ul class="ltx_biblist">
<li class="ltx_bibitem" id="bib.bib1"><span class="ltx_tag ltx_role_refnum ltx_tag_bibitem">[1]</span>
<span class="ltx_bibblock">A. Author, B. Writer.</span>
<span class="ltx_bibblock">OBLIQ-Bench: ranking under oblique queries.</span>
<span class="ltx_bibblock">arXiv preprint arXiv:2602.00001, 2026.</span></li>
<li class="ltx_bibitem" id="bib.bib2"><span class="ltx_tag ltx_role_refnum ltx_tag_bibitem">[2]</span>
<span class="ltx_bibblock">C. Coder.</span>
<span class="ltx_bibblock">A referenced paper.</span>
<span class="ltx_bibblock">In Proc. Conf., 2020.</span></li>
<li class="ltx_bibitem" id="bib.bib3"><span class="ltx_bibblock">Short.</span></li>
</ul>
"""


def test_bibliography_titles_are_read_from_the_arxiv_html(monkeypatch):
    monkeypatch.setattr(scholar, "_fetch_arxiv_html", lambda arxiv_id: HTML)
    assert scholar.arxiv_bibliography("2608.23552v2") == ["OBLIQ-Bench: ranking under oblique queries", "A referenced paper"]


def test_no_html_means_no_titles(monkeypatch):
    monkeypatch.setattr(scholar, "_fetch_arxiv_html", lambda arxiv_id: "")
    assert scholar.arxiv_bibliography("1801.00001") == []


def test_find_paper_adds_bibliography_entries_openalex_did_not_link(monkeypatch):
    def work(i, title, refs=()):
        return scholar.Work(id=f"W{i}", title=title, abstract="", year=2026, url=f"https://doi.org/10.0/{i}", references=tuple(refs))

    def get(path, params=None, **kw):
        if path.startswith("/works/doi:"):
            return {"id": "https://openalex.org/W0", "title": "The paper", "publication_year": 2026, "referenced_works": ["https://openalex.org/W1"]}
        return {}

    monkeypatch.setattr(scholar, "_get", get)
    monkeypatch.setattr(scholar, "get_works", lambda ids: [work(1, "A referenced paper")])
    looked = []

    def find_by_title(title):
        looked.append(title)
        return work(7, "OBLIQ-Bench: ranking under oblique queries") if "OBLIQ" in title else None

    monkeypatch.setattr(scholar, "find_by_title", find_by_title)
    monkeypatch.setattr(scholar, "_fetch_arxiv_html", lambda arxiv_id: HTML)
    found = REAL_FIND_PAPER("The paper", "2608.23552")
    assert found is not None
    work0, refs = found
    assert refs == ["W1", "W7"], "the OpenAlex list first, then the arXiv-only entry that OpenAlex has a record for"
    assert looked == ["OBLIQ-Bench: ranking under oblique queries"], "entries already in the OpenAlex list are not looked up"
    assert work0.references == ("W1",), "the work's own OpenAlex record is unchanged, so callers can tell the two apart"


def test_bibliography_lookups_can_be_switched_off(monkeypatch):
    monkeypatch.setattr(scholar, "_get", lambda path, params=None, **kw: {"id": "https://openalex.org/W0", "title": "The paper", "referenced_works": []} if path.startswith("/works/doi:") else {})
    monkeypatch.setattr(scholar, "_fetch_arxiv_html", lambda arxiv_id: (_ for _ in ()).throw(AssertionError("must not fetch")))
    assert REAL_FIND_PAPER("The paper", "2608.23552", arxiv_refs=False) == (scholar._work({"id": "https://openalex.org/W0", "title": "The paper", "referenced_works": []}), [])
