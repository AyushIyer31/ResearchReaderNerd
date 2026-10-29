"""Saving a run's outputs and describing failures. Shared by the CLI and the web app."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Callable

import anthropic

from .analyze import AnalysisError, AnalysisResult, Prompts, analyze_paper
from .consider import Consideration, consider_benchmarks
from .paper import Paper, PaperError
from .pricing import Spend
from .report import render_report
from .report_html import render_html
from .scholar import ScholarError
from .storage import Storage

BUDGET_USD = 1.00  # per paper; later paid steps are skipped once a run reaches it


def run_pipeline(
    paper: Paper,
    client,
    prompts: Prompts,
    *,
    model: str,
    effort: str,
    consider: bool = True,
    arxiv_id: str | None = None,
    progress: Callable[[str], None] = lambda stage: None,
) -> tuple[AnalysisResult, Consideration | None, Spend]:
    """R1/R2 from the PDF, then (optionally) the benchmarks the paper might consider."""
    spend = Spend()
    progress("analyze")
    result = analyze_paper(paper, client, prompts=prompts, model=model, effort=effort)
    spend.add("analyze", result.model, result.input_tokens, result.output_tokens)
    consideration = None
    if consider:
        consideration = consider_benchmarks(
            result.analysis, client=client, model=model, effort=effort, prompts=prompts, spend=spend,
            budget_usd=BUDGET_USD, arxiv_id=arxiv_id, progress=progress,
        )
    return result, consideration, spend


def save_run(
    paper: Paper,
    result: AnalysisResult,
    prompts: Prompts,
    *,
    storage: Storage,
    requested_model: str,
    effort: str,
    consideration: Consideration | None = None,
    spend: Spend | None = None,
) -> str:
    """Write <name>.md, .html, .json (the analysis), .consider.json and .run.json. Returns <name>."""
    name = paper.name
    if spend is None:
        spend = Spend()
        spend.add("analyze", result.model, result.input_tokens, result.output_tokens)
    storage.write_text(
        f"{name}.md",
        render_report(result.analysis, source=paper.source, model=result.model, consideration=consideration),
        "text/markdown; charset=utf-8",
    )
    storage.write_text(
        f"{name}.html",
        render_html(result.analysis, source=paper.source, model=result.model, consideration=consideration),
        "text/html; charset=utf-8",
    )
    # The analysis JSON keeps the structured data for later steps.
    storage.write_text(f"{name}.json", result.analysis.model_dump_json(indent=2), "application/json")
    if consideration is not None:
        storage.write_text(f"{name}.consider.json", consideration.model_dump_json(indent=2), "application/json")
    run_info = {
        "source": paper.source,
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "requested_model": requested_model,
        "model": result.model,
        "effort": effort,
        "input_tokens": spend.input_tokens,
        "output_tokens": spend.output_tokens,
        "estimated_cost_usd": spend.cost,
        "calls": spend.calls,
        "prompts": prompts.to_dict(),
        "prompts_are_default": prompts == Prompts(),
    }
    storage.write_text(f"{name}.run.json", json.dumps(run_info, indent=2), "application/json")
    return name


def describe_error(e: Exception) -> str | None:
    """A plain message for failures a user can act on, or None for unexpected errors."""
    if isinstance(e, (PaperError, AnalysisError, ScholarError)):
        return str(e)
    if isinstance(e, anthropic.AuthenticationError):
        return "The Anthropic API key was rejected. Check ANTHROPIC_API_KEY."
    if isinstance(e, anthropic.BadRequestError):
        return f"The API rejected the request: {e.message}"
    if isinstance(e, anthropic.RateLimitError):
        return "Rate limited by the API. Wait a minute and try again."
    if isinstance(e, anthropic.APIStatusError):
        return f"API error {e.status_code}: {e.message}"
    if isinstance(e, anthropic.APIConnectionError):
        return "Could not reach the Anthropic API. Check your connection."
    # The SDK raises TypeError (not an API error) when no credentials are configured at all.
    if isinstance(e, TypeError) and "authentication" in str(e):
        return "No Anthropic credentials found. Set ANTHROPIC_API_KEY (or add it to .env) and try again."
    return None
