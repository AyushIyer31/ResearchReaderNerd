"""Render a PaperAnalysis as a self-contained HTML report."""

from __future__ import annotations

from datetime import date
from html import escape

from .schema import Assumption, Benchmark, PaperAnalysis

STYLE = """
/* Layout: one reading column; results tables scroll sideways inside their own box. */
:root {
  --paper: #fafbfc; --ink: #1c2330; --muted: #5b6577; --rule: #dde2ea; --panel: #f1f4f8;
  --accent: #1f5c8a; --accent-tint: #e4eef7; --inferred: #9a5b00; --inferred-tint: #fbf0dc;
  --display: "Newsreader", "Iowan Old Style", Georgia, serif;
  --body: "Source Sans 3", "Segoe UI", system-ui, sans-serif;
  --data: "IBM Plex Mono", ui-monospace, "SF Mono", Menlo, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --paper: #11151c; --ink: #e2e7ef; --muted: #9aa5b6; --rule: #2a323f; --panel: #181e27;
  --accent: #82b6de; --accent-tint: #1b2a3a; --inferred: #e3ad55; --inferred-tint: #2d2414; color-scheme: dark } }
:root[data-theme="dark"] {
  --paper: #11151c; --ink: #e2e7ef; --muted: #9aa5b6; --rule: #2a323f; --panel: #181e27;
  --accent: #82b6de; --accent-tint: #1b2a3a; --inferred: #e3ad55; --inferred-tint: #2d2414; color-scheme: dark }
body { background: var(--paper); color: var(--ink); font: 17px/1.6 var(--body);
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }
main { max-width: 52rem; margin: 0 auto; padding-block: 2.5rem 4rem; display: grid; gap: 2.75rem; }
h1, h2, h3 { font-family: var(--display); font-weight: 600; line-height: 1.2; text-wrap: balance; margin: 0; }
h1 { font-size: 2.3rem; }
h2 { font-size: 1.6rem; display: flex; gap: .6rem; align-items: baseline; }
h3 { font-size: 1.2rem; }
p { margin: 0; max-width: 65ch; }
a { color: var(--accent); }
.eyebrow { font: 600 .78rem var(--body); letter-spacing: .08em; text-transform: uppercase; color: var(--muted); }
.qid { font: 500 .85rem var(--data); color: var(--accent); }
header { display: grid; gap: .75rem; }
.authors { color: var(--muted); }
.meta { font: .82rem/1.5 var(--data); color: var(--muted); overflow-wrap: anywhere; }
.counts { font: .9rem var(--data); color: var(--muted); }
details.sec { border-top: 1px solid var(--rule); padding-top: 1.25rem; min-width: 0; }
details.sub { border: 1px solid var(--rule); border-radius: 4px; padding: .7rem .9rem; min-width: 0; }
details > summary { list-style: none; cursor: pointer; display: flex; flex-wrap: wrap; align-items: baseline; gap: .3rem .75rem; }
details > summary::-webkit-details-marker { display: none; }
.twisty { flex: none; align-self: center; display: grid; place-items: center; width: 1.45rem; height: 1.45rem;
  border-radius: 50%; border: 1.5px solid var(--accent); background: var(--accent-tint); transition: transform .15s; }
.twisty::before { content: ""; width: .36rem; height: .36rem; margin-left: -.14rem;
  border-right: 2px solid var(--accent); border-bottom: 2px solid var(--accent); transform: rotate(-45deg); }
details[open] > summary > .twisty { transform: rotate(90deg); }
@media (prefers-reduced-motion: reduce) { .twisty { transition: none; } }
details > summary { border-radius: 6px; padding: .25rem .35rem; margin: -.25rem -.35rem; }
details > summary:hover { background: var(--accent-tint); }
details > summary:hover h2, details > summary:hover h3 { color: var(--accent); }
.sum-meta { font: .8rem var(--data); color: var(--muted); }
.sec-body { display: grid; gap: 1.25rem; padding-top: 1.1rem; min-width: 0; }
.sub-body { display: grid; gap: .9rem; padding-top: .9rem; min-width: 0; }
.sec-body > details.sub + details.sub { margin-top: -.5rem; }
.toggles { display: flex; gap: .5rem; }
.toggles button { font: .8rem var(--body); color: var(--muted); background: none; border: 1px solid var(--rule); border-radius: 3px; padding: .15rem .55rem; cursor: pointer; }
.toggles button:hover { color: var(--accent); border-color: var(--accent); }
:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.lede { font-size: 1.1rem; }
.method { color: var(--muted); }
.method strong { color: var(--ink); }
ol.assumptions { list-style: none; margin: 0; padding: 0; display: grid; gap: 1.1rem; }
ol.assumptions li { display: grid; gap: .35rem; padding-left: 1rem; border-left: 2px solid var(--rule); }
ol.assumptions li.implicit { border-left-color: var(--inferred); }
.statement { font-weight: 600; }
.rank { font: 500 .85rem var(--data); color: var(--accent); margin-right: .6rem; }
.tags { display: flex; flex-wrap: wrap; gap: .4rem; font: .78rem var(--data); }
.tag { padding: .1rem .45rem; border-radius: 3px; background: var(--panel); color: var(--muted); }
.tag.inferred { background: var(--inferred-tint); color: var(--inferred); font-weight: 600; }
blockquote { margin: 0; padding-left: .8rem; border-left: 1px dotted var(--muted); color: var(--muted); font-style: italic; max-width: 65ch; }
.why { max-width: 65ch; }
.why b { font-weight: 600; }
.scroll { overflow-x: auto; border: 1px solid var(--rule); border-radius: 4px; }
table { border-collapse: collapse; width: 100%; font-size: .92rem; }
th, td { padding: .45rem .75rem; text-align: left; border-bottom: 1px solid var(--rule); vertical-align: top; }
th { font: 600 .74rem var(--body); letter-spacing: .06em; text-transform: uppercase; color: var(--muted); background: var(--panel); white-space: nowrap; }
tr:last-child td { border-bottom: 0; }
td.num { font: .88rem var(--data); font-variant-numeric: tabular-nums; white-space: nowrap; }
td.where { font: .8rem var(--data); color: var(--muted); white-space: nowrap; }
tr.proposed td { background: var(--accent-tint); }
tr.proposed td:first-child { font-weight: 600; box-shadow: inset 3px 0 var(--accent); }
.sub-body dl { margin: 0; display: grid; grid-template-columns: max-content 1fr; gap: .3rem 1rem; font-size: .95rem; }
.sub-body dt { color: var(--muted); }
.sub-body dd { margin: 0; min-width: 0; max-width: 65ch; }
.legend { font-size: .85rem; color: var(--muted); }
.legend i { display: inline-block; width: .8rem; height: .8rem; vertical-align: -.1rem; margin-right: .3rem; background: var(--accent-tint); box-shadow: inset 3px 0 var(--accent); }
ol.suggestions { list-style: none; margin: 0; padding: 0; display: grid; gap: 1rem; }
ol.suggestions li { display: grid; gap: .3rem; padding-left: 1rem; border-left: 2px solid var(--accent); }
.found { display: flex; flex-wrap: wrap; gap: .4rem; font: .78rem var(--data); }
ul.evidence, ul.papers { margin: 0; padding-left: 1.1rem; font-size: .92rem; color: var(--muted); display: grid; gap: .2rem; }
ul.papers { color: var(--ink); }
ul.caveats { margin: 0; padding-left: 1.2rem; display: grid; gap: .5rem; max-width: 65ch; }
footer { font-size: .88rem; color: var(--muted); border-top: 1px solid var(--rule); padding-top: 1.25rem; }
@media (max-width: 560px) {
  body { font-size: 16px; }
  h1 { font-size: 1.8rem; }
  .sub-body dl { grid-template-columns: 1fr; }
  .sub-body dt { margin-top: .4rem; }
}
"""

SCRIPT = """
document.querySelectorAll(".toggles button").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll("details").forEach((d) => { d.open = button.dataset.open === "1"; });
  });
});
"""

FONTS = (
    '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500'
    '&family=Newsreader:opsz,wght@6..72,600&family=Source+Sans+3:ital,wght@0,400;0,600;1,400&display=swap">'
)


def render_html(
    analysis: PaperAnalysis,
    *,
    source: str,
    model: str,
    today: date | None = None,
    standalone: bool = True,
    consideration=None,
) -> str:
    """Return the report as HTML. standalone=False omits the <!doctype>/<html> wrapper."""
    today = today or date.today()
    stated = [a for a in analysis.assumptions if a.kind == "explicit"]
    inferred = [a for a in analysis.assumptions if a.kind == "implicit"]
    e = escape

    parts = [
        f"<title>{e(analysis.title)}</title>",
        '<meta name="claude:viewport-fit-sides" content="cover">',
        FONTS,
        f"<style>{STYLE}</style>",
        "<main>",
        "<header>",
        '<div class="eyebrow">ResearchReaderNerd report</div>',
        f"<h1>{e(analysis.title)}</h1>",
        f'<p class="authors">{e(", ".join(analysis.authors) or "Authors not listed")}</p>',
        f'<div class="meta">{_link(source)} · generated {today.isoformat()} with {e(model)}</div>',
        f'<div class="counts">{len(analysis.assumptions)} assumptions ({len(stated)} stated, '
        f"{len(inferred)} inferred) · {len(analysis.benchmarks)} benchmarks · {len(analysis.caveats)} caveats</div>",
        '<div class="toggles"><button type="button" data-open="1">Expand all</button>'
        '<button type="button" data-open="0">Collapse all</button></div>',
        "</header>",
        _section(
            "Summary",
            f'<p class="lede">{e(analysis.summary)}</p>'
            f'<p class="method">Proposed method: <strong>{e(analysis.proposed_method)}</strong></p>',
            meta=e(analysis.proposed_method),
        ),
        _assumptions_html(analysis.assumptions),
        _benchmarks_html(analysis.benchmarks, analysis.proposed_method),
    ]
    if consideration is not None:
        parts.append(_consider_html(consideration))
    if analysis.caveats:
        parts.append(
            _section(
                "Caveats",
                '<ul class="caveats">' + "".join(f"<li>{e(c)}</li>" for c in analysis.caveats) + "</ul>",
                meta=_count(len(analysis.caveats), "caveat"),
            )
        )
    parts += [
        "<footer>Everything in this report was extracted from the paper alone. Inferred assumptions are the "
        "model's reading of the paper; check each quote against its location before relying on it.</footer>",
        "</main>",
        f"<script>{SCRIPT}</script>",
    ]
    body = "\n".join(parts)
    if not standalone:
        return body + "\n"
    return (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        f"</head>\n<body>\n{body}\n</body>\n</html>\n"
    )


def _count(n: int, noun: str) -> str:
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def _section(title: str, body: str, *, meta: str = "", qid: str = "", open: bool = False) -> str:
    """A top-level report section, collapsed to its heading and a one-line summary until clicked."""
    qid_html = f'<span class="qid">{qid}</span> ' if qid else ""
    return (
        f'<details class="sec"{" open" if open else ""}>'
        f'<summary><span class="twisty" aria-hidden="true"></span><h2>{qid_html}{title}</h2>'
        f'<span class="sum-meta">{meta}</span></summary>'
        f'<div class="sec-body">{body}</div></details>'
    )


def _link(source: str) -> str:
    if source.startswith(("http://", "https://")):
        return f'<a href="{escape(source)}">{escape(source)}</a>'
    return escape(source)


def _assumption_li(a: Assumption, rank: int) -> str:
    e = escape
    tags = [f'<span class="tag">{e(a.location)}</span>', f'<span class="tag">{e(a.category.replace("_", " "))}</span>']
    if a.kind == "implicit":
        tags.insert(0, '<span class="tag inferred">Inferred</span>')
    return (
        f'<li class="{a.kind}">'
        f'<div class="statement"><span class="rank">{rank}</span>{e(a.statement)}</div>'
        f'<div class="tags">{"".join(tags)}</div>'
        f"<blockquote>“{e(a.evidence)}”</blockquote>"
        f'<p class="why"><b>Why it matters:</b> {e(a.why_it_matters)}</p>'
        "</li>"
    )


def _sub(title: str, body: str, *, meta: str = "", open: bool = False) -> str:
    """A collapsible group inside a section."""
    return (
        f'<details class="sub"{" open" if open else ""}>'
        f'<summary><span class="twisty" aria-hidden="true"></span><h3>{title}</h3>'
        f'<span class="sum-meta">{meta}</span></summary>'
        f'<div class="sub-body">{body}</div></details>'
    )


def _assumptions_html(assumptions: list[Assumption]) -> str:
    if not assumptions:
        return _section("Assumptions", "<p>No assumptions were identified.</p>", qid="R1", meta="none found")
    inferred = sum(a.kind == "implicit" for a in assumptions)
    items = '<ol class="assumptions">' + "".join(_assumption_li(a, i) for i, a in enumerate(assumptions, 1)) + "</ol>"
    meta = f"top {len(assumptions)}, most important first · {len(assumptions) - inferred} stated, {inferred} inferred"
    return _section("Assumptions", items, qid="R1", meta=meta)


def _results_table(b: Benchmark) -> str:
    """One row per (method, source location), one column per metric."""
    e = escape
    metrics: list[str] = []
    rows: dict[tuple[str, str], dict] = {}
    for r in b.results:
        if r.metric not in metrics:
            metrics.append(r.metric)
        row = rows.setdefault((r.method, r.location), {"proposed": r.is_proposed_method, "values": {}})
        row["values"][r.metric] = r.value

    head = "".join(f"<th>{e(m)}</th>" for m in metrics)
    body = []
    for (method, location), row in rows.items():
        cells = "".join(f'<td class="num">{e(row["values"].get(m, "–"))}</td>' for m in metrics)
        cls = ' class="proposed"' if row["proposed"] else ""
        body.append(f"<tr{cls}><td>{e(method)}</td>{cells}<td class=\"where\">{e(location)}</td></tr>")
    return (
        f'<div class="scroll"><table><thead><tr><th>Method</th>{head}<th>Where</th></tr></thead>'
        f"<tbody>{''.join(body)}</tbody></table></div>"
    )


def _benchmarks_html(benchmarks: list[Benchmark], proposed_method: str) -> str:
    e = escape
    if not benchmarks:
        return _section("Benchmarks", "<p>No benchmarks were identified.</p>", qid="R2", meta="none found")

    rows = "".join(
        f"<tr><td>{e(b.name)}</td><td>{e(b.task)}</td><td>{e(', '.join(b.metrics))}</td></tr>" for b in benchmarks
    )
    out = [
        '<div class="scroll"><table><thead><tr><th>Benchmark</th><th>Task</th><th>Metrics</th></tr></thead>'
        f"<tbody>{rows}</tbody></table></div>",
        f'<p class="legend"><i></i>Highlighted rows are {e(proposed_method)}, the paper\'s own method. '
        "Open a benchmark to see its results.</p>",
    ]
    for b in benchmarks:
        details = [("Splits", ", ".join(b.splits) or "Not stated")]
        if b.cited_as:
            details.append(("Cited as", b.cited_as))
        if b.notes:
            details.append(("Comparison notes", b.notes))
        dl = "".join(f"<dt>{e(k)}</dt><dd>{e(v)}</dd>" for k, v in details)
        table = _results_table(b) if b.results else "<p>No numbers reported on this benchmark.</p>"
        methods = len({(r.method, r.location) for r in b.results})
        out.append(_sub(e(b.name), f"<dl>{dl}</dl>{table}", meta=f"{e(b.task)} · {_count(methods, 'row')}", open=False))
    return _section("Benchmarks", "".join(out), qid="R2", meta=_count(len(benchmarks), "benchmark"))


def _consider_html(c) -> str:
    e = escape
    if c.suggestions:
        items = []
        for i, sug in enumerate(c.suggestions, 1):
            chips = "".join(f'<span class="tag">{e(f)}</span>' for f in sug.found_by)
            evidence = "".join(f"<li>{e(x)}</li>" for x in sug.evidence)
            links = "".join(f'<li><a href="{e(l.url)}">{e(l.title)}</a></li>' for l in sug.links)
            items.append(
                f'<li><div class="statement"><span class="rank">{i}</span>{e(sug.name)}</div>'
                f'<div class="found">{chips}</div><p class="why">{e(sug.why)}</p>'
                f'<ul class="evidence">{evidence}{links}</ul></li>'
            )
        body = f'<p class="legend">Benchmarks this paper did not use that fit its task ({e(c.field)}), most useful first.</p>'
        body += '<ol class="suggestions">' + "".join(items) + "</ol>"
    else:
        body = "<p>No suggestions this time.</p>"

    users = []
    for u in c.used_by:
        if u.source is None:
            users.append(_sub(e(u.benchmark), "<p>Source paper not found in OpenAlex.</p>", meta="not found", open=False))
            continue
        papers = "".join(f'<li><a href="{e(p.url)}">{e(p.title)}</a></li>' for p in u.papers)
        inner = (f'<p>Cited to <a href="{e(u.source.url)}">{e(u.source.title)}</a>. Most-cited papers citing it:</p>'
                 f'<ul class="papers">{papers}</ul>')
        users.append(_sub(e(u.benchmark), inner, meta=f"{len(u.papers)} papers", open=False))
    body += '<h3 style="margin-top:.5rem">Who else uses this paper\'s benchmarks</h3>' + "".join(users)

    how = (f"<dl><dt>Themes</dt><dd>{e(', '.join(c.themes))}</dd>"
           f"<dt>Catalog searches</dt><dd>{e(', '.join(c.catalog_queries))}</dd>"
           f"<dt>Read</dt><dd>{c.stats.get('papers_read', 0)} related papers' titles and abstracts "
           f"({c.stats.get('references', 0)} references), {c.stats.get('catalog_hits', 0)} catalog hits</dd>"
           f"<dt>Bibliographies</dt><dd>Checked the {c.stats.get('cited_works_checked', 0)} papers the related papers "
           f"cite most; {c.stats.get('dataset_papers', 0)} introduce a benchmark not used here</dd></dl>")
    if c.notes:
        how += '<ul class="caveats">' + "".join(f"<li>{e(n)}</li>" for n in c.notes) + "</ul>"
    body += _sub("How these were found", how, meta="sources and limits", open=False)
    return _section("Benchmarks to consider", body, qid="R3", meta=_count(len(c.suggestions), "suggestion"))
