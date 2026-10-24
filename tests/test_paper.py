import pytest

from researchreadernerd.paper import PaperError, load_paper, paper_from_bytes, parse_arxiv_id, safe_name

from .conftest import MINIMAL_PDF


@pytest.mark.parametrize(
    "text, expected",
    [
        ("2401.01234", "2401.01234"),
        ("2401.01234v2", "2401.01234v2"),
        ("arXiv:1706.03762", "1706.03762"),
        ("https://arxiv.org/abs/1706.03762", "1706.03762"),
        ("https://arxiv.org/pdf/1706.03762v7", "1706.03762v7"),
        ("https://arxiv.org/pdf/1706.03762.pdf", "1706.03762"),
        ("hep-th/9901001", "hep-th/9901001"),
        ("math.GT/0309136", "math.GT/0309136"),
        ("not-a-paper", None),
        ("https://example.com/abs/1706.03762", None),
    ],
)
def test_parse_arxiv_id(text, expected):
    assert parse_arxiv_id(text) == expected


def test_load_local_pdf(tmp_path):
    path = tmp_path / "my-paper.pdf"
    path.write_bytes(MINIMAL_PDF)
    paper = load_paper(str(path))
    assert paper.name == "my-paper"
    assert paper.pdf == MINIMAL_PDF


def test_rejects_non_pdf_file(tmp_path):
    path = tmp_path / "notes.pdf"
    path.write_text("just text")
    with pytest.raises(PaperError, match="not a PDF"):
        load_paper(str(path))


def test_rejects_unknown_source():
    with pytest.raises(PaperError, match="not a PDF file or an arXiv ID"):
        load_paper("definitely-not-a-file-or-id")


def test_arxiv_download(monkeypatch):
    seen = {}

    def fake_download(url):
        seen["url"] = url
        return MINIMAL_PDF

    monkeypatch.setattr("researchreadernerd.paper._download", fake_download)
    paper = load_paper("hep-th/9901001")
    assert seen["url"] == "https://arxiv.org/pdf/hep-th/9901001"
    assert paper.name == "hep-th_9901001"


@pytest.mark.parametrize(
    "filename, expected",
    [("My Paper (v2).pdf", "My-Paper-v2"), ("../../evil.pdf", "evil"), ("....pdf", "uploaded-paper"), ("résumé.pdf", "résumé")],
)
def test_safe_name(filename, expected):
    assert safe_name(filename) == expected


def test_paper_from_bytes():
    paper = paper_from_bytes(MINIMAL_PDF, "draft.pdf")
    assert (paper.name, paper.source) == ("draft", "Uploaded file: draft.pdf")
    with pytest.raises(PaperError, match="not a PDF"):
        paper_from_bytes(b"text", "draft.pdf")
