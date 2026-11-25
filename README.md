# ResearchReaderNerd

Answers a reader's questions about a scientific paper, with every claim tied to where it came from. See the [proposal](docs/PROPOSAL.md) for the full plan.

**Status: Phase 1.** It answers the question that the paper itself can answer:

- **R2: What are the benchmarks used?** Each dataset or benchmark, its splits and metrics, how the paper cites it, and every number reported on it for the proposed method and the baselines.

R3 to R6 (other work on the same benchmarks, comparison with competitors, missing related work, code availability) come in later phases. The benchmarks-to-consider step gathers candidates from the paper's references, from papers that use the same benchmarks, from those papers' bibliographies, from the model's own knowledge of the topic (each name kept only when OpenAlex confirms its source paper or papers naming it), and from public catalogs; the ten suggestions are ordered by that evidence. A published paper runs in a second mode: the most-cited papers citing it are read too, a benchmark found only through them counts if it existed six months before the paper, and evidence from cited and from citing papers is always reported separately. An unpublished manuscript has no citing papers, so only the first mode applies; a paper not yet written, given as a title and abstract, has no reference list either, and the most-cited papers on its topics stand in for it.

## Setup

Requires Python 3.10 or later and [Ollama](https://ollama.com/) with the Qwen2.5 14B model, which runs on your own machine (about 9 GB to download; 16 GB of memory or more recommended at the default 32K context).

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
ollama pull qwen2.5:14b
```

Ollama must be running (`ollama serve`, or the desktop app). If it listens somewhere other than `http://127.0.0.1:11434`, set `OLLAMA_HOST`.

To use Claude instead, pass a Claude model name (`--model claude-haiku-5-5`, or `RRN_MODEL=claude-haiku-5-5` for the web app) and provide an [Anthropic API key](https://console.anthropic.com/): `export ANTHROPIC_API_KEY=...`, or put `ANTHROPIC_API_KEY=...` in a `.env` file in the project folder. `rrn` reads it at startup, and `.gitignore` keeps it out of git.

The benchmarks-to-consider step looks papers up in [OpenAlex](https://openalex.org). Without a key, OpenAlex allows roughly 100 requests a day per network address, which a handful of reports uses up. A free key (sign up at openalex.org, then Settings → API key) allows about ten times that: put `OPENALEX_API_KEY=...` in `.env` or export it.

## Web app

```bash
.venv/bin/rrn-web            # then open http://127.0.0.1:8000
```

Enter an arXiv ID or link, or upload a PDF (the Upload PDF button, or drop the file anywhere on the page), and press Analyze. Uploaded PDFs are kept in `reports/uploads/`, and their report is named after the file. The panel on the right lists each stage of the pipeline with the prompts it sends to Claude. Edit any prompt before a run; edits are kept in your browser, and "Reset" restores the default. Each report saves the prompts, model, and token counts it was made with (`reports/<paper>.run.json`), and "Load this run's prompts" copies them back into the panel.

The server listens on localhost only. The model it uses is `RRN_MODEL`, else the default local model.

## Command line

```bash
.venv/bin/rrn 1706.03762                        # arXiv ID
.venv/bin/rrn https://arxiv.org/abs/1706.03762  # arXiv URL
.venv/bin/rrn path/to/paper.pdf                 # local PDF
```

Each run writes two files to `reports/` (change it with `-o`):

- `<paper>.md`: the readable report.
- `<paper>.json`: the same analysis as structured data. Later phases build on its benchmark list.

Options: `--model` (default `qwen2.5:14b` through Ollama; any other pulled Ollama model works, and `claude-haiku-5-5`, `claude-sonnet-5-5` or `claude-opus-5-5` use the Anthropic API), `--effort` (`low` to `max`, default `medium`; Claude only), and `--skip-consider` (R1 and R2 only, without the search for benchmarks to consider).

A local model has a fixed context window (`RRN_NUM_CTX`, default 32768 tokens). The paper's text is cut to fit, and the report's caveats say so when that happens. On an Apple M3 Max, reading a paper with Qwen2.5 14B takes a few minutes.

## How it works

With a local model, the paper's text is extracted with pypdf, page by page, and sent to Ollama's chat API with the JSON schema as its `format`, so the model's answer is constrained to the schema ([schema.py](src/researchreadernerd/schema.py)). With Claude, the PDF goes to the API as a document, so tables and figures are read along with the text. Either way the report writer turns the structured answer into Markdown, and the prompt limits the model to the paper itself: no numbers or citations from its own memory.

With Sonnet or Opus, a request Claude's safety classifiers decline is re-run automatically on Anthropic's recommended fallback model (the API's `fallbacks: "default"` option). Haiku has no such fallback; a declined paper reports the refusal.

## Tests

```bash
.venv/bin/pytest
```

The tests use a fake HTTP transport, so they need no API key and make no network calls.
