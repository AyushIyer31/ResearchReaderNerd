"""Saving a run's outputs and describing failures. Shared by the CLI and the web app."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import anthropic

from .analyze import AnalysisError, AnalysisResult, Prompts
from .paper import Paper, PaperError
from .report import render_report
from .report_html import render_html
from .storage import Storage

# USD per million tokens (input, output), for the cost estimate shown after a run.
PRICES = {"claude-sonnet-5-5": (2.0, 10.0), "claude-opus-5-5": (4.0, 20.0)}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    price = PRICES.get(model)
    if price is None:
        return None
    return (input_tokens * price[0] + output_tokens * price[1]) / 1_000_000


def save_run(
    paper: Paper, result: AnalysisResult, prompts: Prompts, *, storage: Storage, requested_model: str, effort: str
) -> str:
    """Write <name>.md, .html, .json (the analysis) and .run.json (prompts, model, tokens). Returns <name>."""
    name = paper.name
    storage.write_text(
        f"{name}.md", render_report(result.analysis, source=paper.source, model=result.model), "text/markdown; charset=utf-8"
    )
    storage.write_text(
        f"{name}.html", render_html(result.analysis, source=paper.source, model=result.model), "text/html; charset=utf-8"
    )
    # The analysis JSON keeps the structured data for later steps (R3 to R6 build on the benchmark list).
    storage.write_text(f"{name}.json", result.analysis.model_dump_json(indent=2), "application/json")
    run_info = {
        "source": paper.source,
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "requested_model": requested_model,
        "model": result.model,
        "effort": effort,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "estimated_cost_usd": estimate_cost(result.model, result.input_tokens, result.output_tokens),
        "prompts": prompts.to_dict(),
        "prompts_are_default": prompts == Prompts(),
    }
    storage.write_text(f"{name}.run.json", json.dumps(run_info, indent=2), "application/json")
    return name


def describe_error(e: Exception) -> str | None:
    """A plain message for failures a user can act on, or None for unexpected errors."""
    if isinstance(e, (PaperError, AnalysisError)):
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
