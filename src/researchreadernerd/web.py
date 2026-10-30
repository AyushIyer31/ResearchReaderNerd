"""Local web app: enter an arXiv ID, edit the prompts for each stage, read the report."""

from __future__ import annotations

import argparse
import base64
import hmac
import json
import os
import re
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path
from typing import Callable, Literal

import anthropic
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel, Field

from .analyze import DEFAULT_EFFORT, DEFAULT_MODEL, OUTPUT_FORMAT, Prompts
from .cli import load_dotenv
from .paper import MAX_PDF_BYTES, Paper, PaperError, load_paper, paper_from_bytes, parse_arxiv_id
from .run import describe_error, run_pipeline, save_run
from .storage import LocalStorage, S3Storage, Storage

Effort = Literal["low", "medium", "high", "xhigh", "max"]
MAX_PROMPT_CHARS = 20_000
_REPORT_NAME = re.compile(r"^[\w.\-]+$")
_UPLOAD_ID = re.compile(r"^[0-9a-f]{32}$")
# A run whose saved status hasn't changed for this long was cut off by a stopped server.
# Runs take a few minutes; during a canary deploy another server may still be finishing one.
STALE_RUN_SECONDS = 15 * 60
_JOB_ID = re.compile(r"^\d{8}-[0-9a-f]{12}$")

# The pipeline as the prompt panel shows it. Stages without prompts run locally.
STAGES = [
    {
        "id": "fetch",
        "title": "Fetch the paper",
        "detail": "Downloads the PDF from arXiv, or uses the PDF you uploaded. Runs locally, no prompt.",
        "prompts": [],
    },
    {
        "id": "analyze",
        "title": "Read the paper (R1 + R2)",
        "detail": "One Claude call with the PDF attached. The three system boxes are joined into one system prompt.",
        "prompts": [
            {"key": "instructions", "label": "Shared instructions", "role": "system"},
            {"key": "assumptions", "label": "R1 · Assumptions", "role": "system"},
            {"key": "benchmarks", "label": "R2 · Benchmarks", "role": "system"},
            {"key": "user", "label": "Message sent with the PDF", "role": "user"},
        ],
    },
    {
        "id": "plan",
        "title": "Plan the benchmark search",
        "detail": "From R1/R2: the paper's field, themes, catalog search terms, and the paper each benchmark is cited to.",
        "prompts": [{"key": "plan", "label": "Planning instructions", "role": "system"}],
    },
    {
        "id": "related",
        "title": "Read related papers",
        "detail": "OpenAlex: the paper's references, and the most-cited papers that use each of its benchmarks. "
                  "Claude reads their titles and abstracts and lists the benchmarks each one uses. "
                  "We also count the papers most often cited in their bibliographies, and Claude picks out "
                  "the ones that introduce a dataset or benchmark.",
        "prompts": [{"key": "extract", "label": "Benchmark extraction", "role": "system"},
                    {"key": "label", "label": "Dataset papers in the bibliographies", "role": "system"}],
    },
    {
        "id": "catalogs",
        "title": "Search benchmark catalogs",
        "detail": "Keyword search of TPC, the UCR time series archive, UCI, Hugging Face, data.gov, data.gov.uk "
                  "and data.europa.eu. Runs locally, no prompt.",
        "prompts": [],
    },
    {
        "id": "rank",
        "title": "Choose 10 to consider",
        "detail": "Claude ranks the candidates. Each suggestion keeps the evidence our code found for it.",
        "prompts": [{"key": "rank", "label": "Ranking instructions", "role": "system"}],
    },
    {
        "id": "write",
        "title": "Write the report",
        "detail": "Turns Claude's structured answer into the report. Runs locally, no prompt.",
        "prompts": [],
    },
]


class PromptsIn(BaseModel):
    instructions: str = Field(max_length=MAX_PROMPT_CHARS)
    assumptions: str = Field(max_length=MAX_PROMPT_CHARS)
    benchmarks: str = Field(max_length=MAX_PROMPT_CHARS)
    user: str = Field(max_length=MAX_PROMPT_CHARS)
    plan: str = Field(max_length=MAX_PROMPT_CHARS)
    extract: str = Field(max_length=MAX_PROMPT_CHARS)
    label: str = Field(max_length=MAX_PROMPT_CHARS)
    rank: str = Field(max_length=MAX_PROMPT_CHARS)


class RunIn(BaseModel):
    paper: str | None = Field(default=None, max_length=300)  # arXiv ID or link
    upload_id: str | None = None  # from POST /api/uploads
    effort: Effort = DEFAULT_EFFORT
    prompts: PromptsIn


@dataclass
class Job:
    id: str
    label: str  # what the progress screen calls the paper
    load: Callable[[], Paper]
    arxiv_id: str | None = None
    stage: str = "fetch"
    status: Literal["running", "done", "error"] = "running"
    error: str | None = None
    report: str | None = None
    started: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)

    def public(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "stage": self.stage,
            "status": self.status,
            "error": self.error,
            "report": self.report,
            "started": self.started,
            "updated": self.updated,
            "elapsed": round(time.time() - self.started),
        }


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _job_key(job_id: str) -> str:
    # Run ids start with their UTC date, so a day's runs share a folder and can be counted.
    return f"jobs/{job_id[:8]}/{job_id}.json"


def create_app(
    storage: Storage | None = None,
    client_factory: Callable[[], anthropic.Anthropic] = anthropic.Anthropic,
    *,
    daily_run_limit: int | None = None,
    origin_secret: str | None = None,
    basic_auth_password: str | None = None,
) -> FastAPI:
    storage = storage or LocalStorage(Path("reports"))
    app = FastAPI(title="ResearchReaderNerd", docs_url=None, redoc_url=None)
    # When this server process started. On AWS a deploy starts a new task, so this is the deploy time.
    build = {
        "sha": os.environ.get("BUILD_SHA") or "local",
        "built_at": os.environ.get("BUILD_DATE") or None,
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    jobs: dict[str, Job] = {}  # runs started by this server process
    lock = threading.Lock()

    if origin_secret or basic_auth_password:
        # Production access control. A request passes with either:
        #  - the origin secret header, which the Cloudflare Worker adds after Cloudflare Access login, or
        #  - the password, for direct use of the AWS hostname (the browser asks for it; any username).
        # /health stays open for the load balancer and the deploy smoke test.
        @app.middleware("http")
        async def require_access(request: Request, call_next):
            if request.url.path == "/health":
                return await call_next(request)
            if origin_secret and hmac.compare_digest(request.headers.get("x-origin-secret", ""), origin_secret):
                return await call_next(request)
            if basic_auth_password and _basic_auth_ok(request.headers.get("authorization", ""), basic_auth_password):
                return await call_next(request)
            if basic_auth_password:
                return PlainTextResponse(
                    "Sign in to use ResearchReaderNerd.",
                    status_code=401,
                    headers={"WWW-Authenticate": 'Basic realm="ResearchReaderNerd", charset="UTF-8"'},
                )
            return PlainTextResponse("Forbidden", status_code=403)

    def save_job(job: Job) -> None:
        job.updated = time.time()
        storage.write_text(_job_key(job.id), json.dumps(job.public()), "application/json")

    def run_job(job: Job, request: RunIn) -> None:
        prompts = Prompts(**request.prompts.model_dump())
        try:
            paper = job.load()

            def progress(stage: str) -> None:
                job.stage = stage
                save_job(job)

            result, consideration, spend = run_pipeline(
                paper, client_factory(), prompts, model=DEFAULT_MODEL, effort=request.effort,
                arxiv_id=job.arxiv_id, progress=progress,
            )
            job.stage = "write"
            job.report = save_run(
                paper, result, prompts, storage=storage, requested_model=DEFAULT_MODEL, effort=request.effort,
                consideration=consideration, spend=spend,
            )
            job.status = "done"
        except Exception as e:
            message = describe_error(e)
            if message is None:
                traceback.print_exc()
                message = f"Unexpected error: {e}"
            job.error = message
            job.status = "error"
        try:
            save_job(job)
        except Exception:
            traceback.print_exc()

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "build": {"sha": build["sha"], "date": build["built_at"] or "unknown", "started_at": build["started_at"]},
            "storage": type(storage).__name__,
            "model": DEFAULT_MODEL,
        }

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return resources.files("researchreadernerd").joinpath("static/index.html").read_text(encoding="utf-8")

    @app.get("/api/config")
    def config() -> dict:
        return {
            "stages": STAGES,
            "defaults": Prompts().to_dict(),
            "effort": DEFAULT_EFFORT,
            "efforts": ["low", "medium", "high", "xhigh", "max"],
            "model": DEFAULT_MODEL,
            "build": build,
            "output_schema": OUTPUT_FORMAT["schema"],
        }

    @app.get("/api/reports")
    def list_reports() -> list[dict]:
        keys = dict(storage.list(""))
        items = []
        for key, modified in keys.items():
            if "/" in key or not key.endswith(".json") or key.endswith(".run.json"):
                continue
            name = key[: -len(".json")]
            if f"{name}.html" not in keys:
                continue
            try:
                title = json.loads(storage.read_text(key) or "{}").get("title", name)
            except ValueError:
                continue
            run = _read_json(storage, f"{name}.run.json")
            items.append(
                {
                    "name": name,
                    "title": title,
                    "modified": modified,
                    "custom_prompts": bool(run) and not run.get("prompts_are_default", True),
                }
            )
        return sorted(items, key=lambda r: r["modified"], reverse=True)

    @app.get("/api/reports/{name}")
    def get_report(name: str) -> dict:
        html = storage.read_text(f"{name}.html") if _REPORT_NAME.match(name) else None
        if html is None:
            raise HTTPException(404, "No such report.")
        return {"name": name, "html": html, "run": _read_json(storage, f"{name}.run.json")}

    @app.post("/api/uploads")
    async def upload(request: Request, filename: str = "paper.pdf") -> dict:
        """Store a PDF sent as the raw request body. Returns an id to start a run with."""
        if int(request.headers.get("content-length") or 0) > MAX_PDF_BYTES:
            raise HTTPException(413, f"The PDF is larger than {MAX_PDF_BYTES // (1024 * 1024)} MB.")
        data = await request.body()
        try:
            paper = paper_from_bytes(data, filename)
        except PaperError as e:
            raise HTTPException(400, str(e)) from e
        upload_id = uuid.uuid4().hex
        storage.write_bytes(f"uploads/{upload_id}.pdf", paper.pdf, "application/pdf")
        storage.write_text(f"uploads/{upload_id}.json", json.dumps({"filename": filename}), "application/json")
        return {"upload_id": upload_id, "filename": filename, "name": paper.name, "bytes": len(paper.pdf)}

    def upload_loader(upload_id: str) -> tuple[str, Callable[[], Paper]]:
        meta = _read_json(storage, f"uploads/{upload_id}.json") if _UPLOAD_ID.match(upload_id) else None
        if meta is None:
            raise HTTPException(400, "That upload was not found. Upload the PDF again.")
        filename = meta["filename"]

        def load() -> Paper:
            data = storage.read_bytes(f"uploads/{upload_id}.pdf")
            if data is None:
                raise PaperError("The uploaded PDF is missing. Upload it again.")
            return paper_from_bytes(data, filename)

        return filename, load

    @app.post("/api/runs")
    def start_run(request: RunIn) -> dict:
        if bool(request.paper) == bool(request.upload_id):
            raise HTTPException(400, "Give an arXiv ID or upload a PDF, not both.")
        arxiv_id = None
        if request.upload_id:
            label, load = upload_loader(request.upload_id)
        else:
            arxiv_id = parse_arxiv_id(request.paper)
            if arxiv_id is None:
                raise HTTPException(400, f"{request.paper!r} is not an arXiv ID or arxiv.org link.")
            label, load = f"arXiv {arxiv_id}", lambda: load_paper(arxiv_id)

        with lock:
            today = _today()
            if daily_run_limit is not None and len(storage.list(f"jobs/{today}/")) >= daily_run_limit:
                raise HTTPException(
                    429, f"The daily limit of {daily_run_limit} analyses has been reached. It resets at midnight UTC."
                )
            job = Job(id=f"{today}-{uuid.uuid4().hex[:12]}", label=label, load=load, arxiv_id=arxiv_id)
            jobs[job.id] = job
            save_job(job)
        threading.Thread(target=run_job, args=(job, request), daemon=True).start()
        return job.public()

    @app.get("/api/runs/{job_id}")
    def get_run(job_id: str) -> dict:
        if job_id in jobs:
            return jobs[job_id].public()
        record = _read_json(storage, _job_key(job_id)) if _JOB_ID.match(job_id) else None
        if record is None:
            raise HTTPException(404, "No such run.")
        stale = time.time() - record.get("updated", 0) > STALE_RUN_SECONDS
        if record["status"] == "running" and stale:
            # Started by another server process that stopped (a restart or deploy) before finishing.
            record.update(status="error", error="The server restarted during this analysis. Start it again.")
        record["elapsed"] = round(time.time() - record.get("started", time.time()))
        return record

    return app


def _basic_auth_ok(header: str, password: str) -> bool:
    scheme, _, encoded = header.partition(" ")
    if scheme.lower() != "basic":
        return False
    try:
        _, _, given = base64.b64decode(encoded).decode("utf-8").partition(":")
    except (ValueError, UnicodeDecodeError):
        return False
    return hmac.compare_digest(given.encode(), password.encode())


def _read_json(storage: Storage, key: str) -> dict | None:
    try:
        text = storage.read_text(key)
        return None if text is None else json.loads(text)
    except ValueError:
        return None


def storage_from_env() -> Storage:
    bucket = os.environ.get("RRN_S3_BUCKET")
    if bucket:
        return S3Storage(bucket, os.environ.get("RRN_S3_PREFIX", ""))
    return LocalStorage(Path(os.environ.get("RRN_REPORTS_DIR", "reports")))


def main(argv: list[str] | None = None) -> None:
    import uvicorn

    parser = argparse.ArgumentParser(prog="rrn-web", description="Run the ResearchReaderNerd web app.")
    parser.add_argument("--host", default="127.0.0.1", help="Use 0.0.0.0 only inside a container.")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    args = parser.parse_args(argv)
    load_dotenv()

    limit = os.environ.get("RRN_DAILY_RUN_LIMIT")
    app = create_app(
        storage_from_env(),
        daily_run_limit=int(limit) if limit else None,
        origin_secret=os.environ.get("RRN_ORIGIN_SECRET") or None,
        basic_auth_password=os.environ.get("RRN_BASIC_AUTH_PASSWORD") or None,
    )
    # By default this listens on localhost only: every run spends API credit on your key.
    print(f"ResearchReaderNerd is running at http://{args.host}:{args.port}  (Ctrl+C to stop)")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
