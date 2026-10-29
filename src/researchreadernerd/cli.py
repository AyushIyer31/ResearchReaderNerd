"""Command-line entry point: rrn <pdf path | arXiv ID | arXiv URL>."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import anthropic

from .analyze import DEFAULT_EFFORT, DEFAULT_MODEL, Prompts
from .paper import PaperError, load_paper, parse_arxiv_id
from .run import BUDGET_USD, describe_error, run_pipeline, save_run
from .storage import LocalStorage


def load_dotenv(path: Path = Path(".env")) -> None:
    """Set KEY=value lines from a .env file, without overriding variables already set."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.removeprefix("export ").split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rrn",
        description="Report a paper's assumptions (R1), benchmarks (R2), and benchmarks it might consider (R3).",
    )
    parser.add_argument("paper", help="Path to a PDF, an arXiv ID (2401.01234), or an arXiv URL.")
    parser.add_argument("-o", "--out-dir", type=Path, default=Path("reports"), help="Where to write the report (default: reports/).")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--effort", default=DEFAULT_EFFORT, choices=["low", "medium", "high", "xhigh", "max"])
    parser.add_argument("--skip-consider", action="store_true",
                        help="Only R1 and R2: skip the search for benchmarks the paper might consider.")
    args = parser.parse_args(argv)
    load_dotenv()

    try:
        paper = load_paper(args.paper)
    except PaperError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    print(f"Analyzing {paper.source} with {args.model}...", file=sys.stderr)
    prompts = Prompts()
    try:
        result, consideration, spend = run_pipeline(
            paper, anthropic.Anthropic(), prompts, model=args.model, effort=args.effort,
            consider=not args.skip_consider, arxiv_id=parse_arxiv_id(args.paper),
            progress=lambda stage: print(f"  {stage}...", file=sys.stderr),
        )
    except Exception as e:
        message = describe_error(e)
        if message is None:
            raise
        print(f"error: {message}", file=sys.stderr)
        return 1

    name = save_run(
        paper, result, prompts, storage=LocalStorage(args.out_dir), requested_model=args.model, effort=args.effort,
        consideration=consideration, spend=spend,
    )
    out = args.out_dir
    found = f"Found {len(result.analysis.assumptions)} assumptions and {len(result.analysis.benchmarks)} benchmarks"
    if consideration is not None:
        found += f", and {len(consideration.suggestions)} benchmarks to consider"
    print(
        f"{found}.\n"
        f"Tokens: {spend.input_tokens:,} in, {spend.output_tokens:,} out (about ${spend.cost:.3f}; budget ${BUDGET_USD:.2f})\n"
        f"Report: {out / (name + '.md')}\nWeb:    {out / (name + '.html')}\nData:   {out / (name + '.json')}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
