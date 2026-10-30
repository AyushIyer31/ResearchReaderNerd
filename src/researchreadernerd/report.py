"""Render a PaperAnalysis as a Markdown report."""

from __future__ import annotations

from datetime import date

from .schema import Assumption, Benchmark, PaperAnalysis


def _cell(text: str) -> str:
    """Make text safe inside a Markdown table cell."""
    return " ".join(text.split()).replace("|", "\\|")


def render_report(
    analysis: PaperAnalysis, *, source: str, model: str, today: date | None = None, consideration=None
) -> str:
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
    if consideration is not None:
        lines += _consider_section(consideration)

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

    lines += [f"Top {len(assumptions)}, most important first.", ""]
    for i, a in enumerate(assumptions, 1):
        kind = "stated" if a.kind == "explicit" else "inferred, not stated in the paper"
        lines += [
            f"{i}. **{a.statement}** ({kind}; {a.category.replace('_', ' ')}, {a.location})",
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


def _consider_section(c) -> list[str]:
    lines = ["## R3: Benchmarks to consider", ""]
    if c.suggestions:
        lines += [f"Benchmarks this paper did not use that fit its task, most useful first ({c.field}).", ""]
        for i, sug in enumerate(c.suggestions, 1):
            lines.append(f"{i}. **{sug.name}** ({', '.join(sug.found_by)}): {sug.why}")
            for e in sug.evidence:
                lines.append(f"    - {e}")
            for link in sug.links:
                lines.append(f"    - [{link.title}]({link.url})")
        lines.append("")
    else:
        lines += ["No suggestions this time.", ""]

    lines += ["### Who else uses this paper's benchmarks", ""]
    for u in c.used_by:
        if u.source is None:
            lines.append(f"- **{u.benchmark}**: source paper not found in OpenAlex.")
            continue
        lines.append(f"- **{u.benchmark}**, cited to [{u.source.title}]({u.source.url}). Most-cited papers citing it:")
        lines += [f"    - [{p.title}]({p.url})" for p in u.papers[:10]]
    lines.append("")

    lines += [
        "### How these were found",
        "",
        f"- Themes: {', '.join(c.themes)}",
        f"- Catalog searches: {', '.join(c.catalog_queries)}",
        f"- Read {c.stats.get('papers_read', 0)} related papers' titles and abstracts "
        f"({c.stats.get('references', 0)} references); {c.stats.get('catalog_hits', 0)} catalog hits.",
        f"- Bibliographies: checked the {c.stats.get('cited_works_checked', 0)} papers the related papers cite most; "
        f"{c.stats.get('dataset_papers', 0)} introduce a benchmark not used here.",
    ]
    lines += [f"- Note: {n}" for n in c.notes]
    lines.append("")
    return lines
