# ResearchReaderNerd

Answers a reader's questions about a scientific paper, with every claim tied to where it came from. See the [proposal](docs/PROPOSAL.md) for the full plan.

**Status: Phase 1.** It answers the two questions that the paper itself can answer:

- **R1: What are the assumptions of the model?** Stated and implicit assumptions, each with its location in the paper, a verbatim quote, and why it matters.
- **R2: What are the benchmarks used?** Each dataset or benchmark, its splits and metrics, how the paper cites it, and every number reported on it for the proposed method and the baselines.

R3 to R6 (other work on the same benchmarks, comparison with competitors, missing related work, code availability) come in later phases.

## Setup

Requires Python 3.10 or later and an [Anthropic API key](https://console.anthropic.com/).

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
export ANTHROPIC_API_KEY=...   # your key
```

Instead of exporting the key, you can put `ANTHROPIC_API_KEY=...` in a `.env` file in the project folder. `rrn` reads it at startup, and `.gitignore` keeps it out of git.

## Web app

```bash
.venv/bin/rrn-web            # then open http://127.0.0.1:8000
```

Enter an arXiv ID or link, or upload a PDF (the Upload PDF button, or drop the file anywhere on the page), and press Analyze. Uploaded PDFs are kept in `reports/uploads/`, and their report is named after the file. The panel on the right lists each stage of the pipeline with the prompts it sends to Claude. Edit any prompt before a run; edits are kept in your browser, and "Reset" restores the default. Each report saves the prompts, model, and token counts it was made with (`reports/<paper>.run.json`), and "Load this run's prompts" copies them back into the panel.

The server listens on localhost only, because every run spends API credit on your key.

## Command line

```bash
.venv/bin/rrn 1706.03762                        # arXiv ID
.venv/bin/rrn https://arxiv.org/abs/1706.03762  # arXiv URL
.venv/bin/rrn path/to/paper.pdf                 # local PDF
```

Each run writes two files to `reports/` (change it with `-o`):

- `<paper>.md`: the readable report.
- `<paper>.json`: the same analysis as structured data. Later phases build on its benchmark list.

Options: `--model` (default `claude-haiku-5-5`, the cheapest; `claude-sonnet-5-5` and `claude-opus-5-5` read more carefully at roughly 20x and 40x the cost), `--effort` (`low` to `max`, default `high`), and `--skip-consider` (R1 and R2 only, without the search for benchmarks to consider).

## How it works

The PDF goes to Claude as a document, so tables and figures are read along with the text. Claude must answer in a fixed JSON schema ([schema.py](src/researchreadernerd/schema.py)), which the report writer turns into Markdown. The prompt limits Claude to the paper itself: no numbers or citations from its own memory.

With Sonnet or Opus, a request Claude's safety classifiers decline is re-run automatically on Anthropic's recommended fallback model (the API's `fallbacks: "default"` option). Haiku has no such fallback; a declined paper reports the refusal.

## Tests

```bash
.venv/bin/pytest
```

The tests use a fake HTTP transport, so they need no API key and make no network calls.
