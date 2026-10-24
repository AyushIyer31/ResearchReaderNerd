"""Render a PaperAnalysis as a Markdown report."""

from __future__ import annotations

from datetime import date

from .schema import Assumption, Benchmark, PaperAnalysis


def _cell(text: str) -> str:
    """Make text safe inside a Markdown table cell."""
    return " ".join(text.split()).replace("|", "\\|")


def render_report(analysis: PaperAnalysis, *, source: str, model: str, today: date | None = None) -> str:
    today = today or date.today()
    lines = [
        f"# {analysis.title}",
        "",
        f"**Authors:** {', '.join(analysis.authors) or 'not listed'}  ",
        f"**Source:** {source}  ",
        f"**Generated:** {today.isoformat()} by ResearchReaderNerd with {model}",
        "",
        "## Summary",
        "",
        analysis.summary,
        "",
        f"**Proposed method:** {analysis.proposed_method}",
        "",
    ]
    lines += _assumptions_section(analysis.assumptions)
    lines += _benchmarks_section(analysis.benchmarks, analysis.proposed_method)

    if analysis.caveats:
        lines += ["## Caveats", ""]
        lines += [f"- {c}" for c in analysis.caveats]
        lines.append("")

    lines += [
        "---",
        "",
        "_Everything above was extracted from the paper alone. Implicit assumptions are the model's "
        "inference; check each quote against its location before relying on it._",
        "",
    ]
    return "\n".join(lines)


def _assumptions_section(assumptions: list[Assumption]) -> list[str]:
    lines = ["## R1: Assumptions", ""]
    if not assumptions:
        return lines + ["No assumptions were identified.", ""]

    groups = [
        ("Stated in the paper", [a for a in assumptions if a.kind == "explicit"]),
        ("Implicit (inferred, not stated)", [a for a in assumptions if a.kind == "implicit"]),
    ]
    for heading, group in groups:
        if not group:
            continue
        lines += [f"### {heading}", ""]
        for i, a in enumerate(group, 1):
            lines += [
                f"{i}. **{a.statement}** ({a.category.replace('_', ' ')}, {a.location})",
                f"    - Evidence: \"{a.evidence}\"",
                f"    - Why it matters: {a.why_it_matters}",
            ]
        lines.append("")
    return lines


def _benchmarks_section(benchmarks: list[Benchmark], proposed_method: str) -> list[str]:
    lines = ["## R2: Benchmarks", ""]
    if not benchmarks:
        return lines + ["No benchmarks were identified.", ""]

    lines += [
        "| Benchmark | Task | Splits | Metrics |",
        "| --- | --- | --- | --- |",
    ]
    for b in benchmarks:
        lines.append(
            f"| {_cell(b.name)} | {_cell(b.task)} | {_cell(', '.join(b.splits) or 'not stated')} "
            f"| {_cell(', '.join(b.metrics))} |"
        )
    lines.append("")

    for b in benchmarks:
        lines += [f"### {b.name}", ""]
        if b.cited_as:
            lines += [f"**Cited as:** {b.cited_as}", ""]
        if b.notes:
            lines += [f"**Comparison notes:** {b.notes}", ""]
        if b.results:
            lines += [
                f"Rows in bold are {proposed_method} (the paper's method).",
                "",
                "| Method | Metric | Value | Where |",
                "| --- | --- | --- | --- |",
            ]
            for r in b.results:
                method = f"**{_cell(r.method)}**" if r.is_proposed_method else _cell(r.method)
                lines.append(f"| {method} | {_cell(r.metric)} | {_cell(r.value)} | {_cell(r.location)} |")
            lines.append("")
        else:
            lines += ["No numbers reported on this benchmark.", ""]
    return lines
