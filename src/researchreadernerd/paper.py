"""Load a paper as PDF bytes from a local file or an arXiv ID/URL."""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path

import httpx2

# The Claude API accepts requests up to 32 MB; base64 adds about a third. Local models get the
# extracted text, so the same cap keeps uploads bounded either way.
MAX_PDF_BYTES = 24 * 1024 * 1024

# New-style IDs (2401.01234, optional version) and old-style IDs (hep-th/9901001).
_NEW_ID = r"\d{4}\.\d{4,5}(?:v\d+)?"
_OLD_ID = r"[a-z][a-z\-]*(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?"
_ARXIV_ID = re.compile(rf"^(?:arxiv:)?({_NEW_ID}|{_OLD_ID})$", re.IGNORECASE)
_ARXIV_URL = re.compile(
    rf"^https?://(?:www\.)?arxiv\.org/(?:abs|pdf)/({_NEW_ID}|{_OLD_ID})(?:\.pdf)?/?$",
    re.IGNORECASE,
)


class PaperError(Exception):
    """The paper could not be loaded."""


@dataclass(frozen=True)
class Paper:
    name: str  # short name used for report filenames
    pdf: bytes
    source: str  # path or URL the PDF came from


def parse_arxiv_id(text: str) -> str | None:
    """Return the arXiv ID in `text` (a bare ID or an arxiv.org URL), else None."""
    text = text.strip()
    for pattern in (_ARXIV_ID, _ARXIV_URL):
        match = pattern.match(text)
        if match:
            return match.group(1)
    return None


def load_paper(source: str) -> Paper:
    """Load a paper from a local PDF path, an arXiv ID, or an arXiv URL."""
    path = Path(source).expanduser()
    if path.is_file():
        return Paper(name=path.stem, pdf=_check_pdf(path.read_bytes(), source), source=str(path))

    arxiv_id = parse_arxiv_id(source)
    if arxiv_id is None:
        raise PaperError(f"{source!r} is not a PDF file or an arXiv ID/URL.")

    url = f"https://arxiv.org/pdf/{arxiv_id}"
    return Paper(name=arxiv_id.replace("/", "_"), pdf=_check_pdf(_download(url), url), source=url)


def safe_name(filename: str) -> str:
    """A report-friendly name from an uploaded file's name: 'My Paper (v2).pdf' -> 'My-Paper-v2'."""
    stem = Path(filename).stem
    cleaned = re.sub(r"[^\w.\-]+", "-", stem).strip("-.")
    return cleaned[:80] or "uploaded-paper"


def paper_from_bytes(data: bytes, filename: str) -> Paper:
    """A paper from an uploaded PDF's bytes."""
    return Paper(name=safe_name(filename), pdf=_check_pdf(data, filename), source=f"Uploaded file: {filename}")


def _download(url: str) -> bytes:
    # httpx2 (installed with the anthropic SDK) carries its own CA bundle, so this
    # works on Python installs without system certificates (e.g. python.org on macOS).
    headers = {"User-Agent": "ResearchReaderNerd/0.1"}
    try:
        with httpx2.stream("GET", url, headers=headers, follow_redirects=True, timeout=60) as response:
            response.raise_for_status()
            data = bytearray()
            for chunk in response.iter_bytes():
                data += chunk
                if len(data) > MAX_PDF_BYTES:
                    break
            return bytes(data)
    except httpx2.HTTPError as e:
        raise PaperError(f"Could not download {url}: {e}") from e


def _check_pdf(data: bytes, source: str) -> bytes:
    # Viewers accept the %PDF magic anywhere in the first 1024 bytes: the spec wants it at byte 0, but many producers
    # (scans, office exports) prepend junk bytes (Harsh Kashyap, PR #1).
    if b"%PDF" not in data[:1024]:
        raise PaperError(f"{source} is not a PDF.")
    if len(data) > MAX_PDF_BYTES:
        raise PaperError(f"{source} is larger than {MAX_PDF_BYTES // (1024 * 1024)} MB.")
    return data


def pdf_text(data: bytes) -> str:
    """The text of a PDF, page by page, for models that cannot read a PDF themselves."""
    from pypdf import PdfReader  # imported here: only local models need it

    try:
        reader = PdfReader(io.BytesIO(data))
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception as e:  # pypdf raises several types on a damaged file
        raise PaperError(f"The PDF's text could not be extracted: {e}") from e
    text = "\n\n".join(f"[page {i}]\n{page.strip()}" for i, page in enumerate(pages, 1) if page.strip())
    text = re.sub(r"[ \t]+", " ", text)
    if not text.strip():
        raise PaperError("The PDF has no extractable text (a scanned paper?); a local model needs the text.")
    return text
