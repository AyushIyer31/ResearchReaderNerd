"""Ask the model to read the paper and return its benchmarks (R2)."""

from __future__ import annotations

import base64
from dataclasses import asdict, dataclass

import anthropic

from .llm import FALLBACK_BETA, FALLBACK_MODELS, AnalysisError, OllamaError, ask, output_format
from .paper import Paper
from .schema import PaperAnalysis

DEFAULT_MODEL = "qwen2.5:14b"  # local, through Ollama; any "claude-*" name uses the Anthropic API
DEFAULT_EFFORT = "medium"
# Room for the model's reasoning plus long results tables. Above ~16K the response
# must be streamed, or the HTTP request can time out. A local model is capped lower
# (llm.MAX_LOCAL_OUTPUT) so the paper fits in its context window.
DEFAULT_MAX_TOKENS = 32000

OUTPUT_FORMAT = output_format(PaperAnalysis)

__all__ = ["AnalysisError", "OllamaError", "FALLBACK_BETA", "FALLBACK_MODELS"]  # re-exported for callers and tests

INSTRUCTIONS_PROMPT = """\
You are a careful reviewer reading a scientific paper for a researcher who wants to \
understand it and check it. Work only from the attached paper. Do not add facts, \
numbers, or citations from your own knowledge: if something is not in the paper, \
leave it out or mention it under caveats.

Title and authors: copy the paper's own title and author list as printed on its first page. \
The title is the paper's title, not the file name and not a description of this report.

Summary: keep it under 150 words. Cover the problem, the approach, and the main claim."""

BENCHMARKS_PROMPT = """\
Benchmarks (R2):
- List every dataset, benchmark, or environment used for evaluation, including ones \
used only in ablations or the appendix.
- For each, record every number reported on it, for the proposed method and for every \
baseline, copied exactly as printed. Note which table or figure each number is from.
- Record how the paper cites the benchmark, copied from its bibliography.
- Under notes, flag anything that limits comparison with other papers, such as custom \
splits or baselines rerun by the authors instead of copied from the original papers."""

USER_PROMPT = "Read the attached paper and extract its benchmarks."

PLAN_PROMPT = """\
You are planning a search for benchmarks a computer-science paper could have used. From \
the paper's title, summary and benchmarks below:
- Name its computer-science subfield.
- List up to 4 themes (tasks, data types, application areas).
- Write 3 to 5 short keyword queries (2 to 4 words each) for searching dataset catalogs \
(UCI, the UCR time series archive, TPC, Hugging Face, government open-data portals). \
Describe the task and data, not the paper's own benchmarks: to find alternatives to \
"WMT 2014 English-German", search "machine translation" or "parallel corpus".
- For each benchmark, give the title of the paper it is cited to: from its bibliography \
entry when there is one, otherwise the paper usually cited for it, if you know it. Each \
title is checked against OpenAlex and dropped if no paper matches."""

EXTRACT_PROMPT = """\
For each numbered paper below (title and abstract), list the benchmarks, datasets or \
evaluation suites it evaluates on, by their standard names (for example "ImageNet", \
"TPC-H", "GLUE"). List only names that appear in the text. Do not list methods, models \
or metrics. Return an empty list for a paper that names none."""

LABEL_PROMPT = """\
The numbered papers below are the ones most often cited by papers related to a \
computer-science paper. Pick out the papers that introduce a dataset, benchmark or \
evaluation suite (for example "ImageNet: A large-scale hierarchical image database" \
introduces ImageNet), and give each one's standard short name. Skip papers that \
introduce a method, model, tool or survey, even when they also release some data as a \
side product. Return an empty list if none do."""

PROPOSE_PROMPT = """\
You advise the authors of the computer-science paper below on benchmarks to evaluate on. \
From its title, summary and method, name up to 15 public benchmarks, datasets or evaluation \
suites that papers on this problem are expected to use, by their standard names, most \
important first. Choose only established, obtainable benchmarks; do not invent names. Skip \
the benchmarks the paper already uses. For each, give the title of the paper that introduced \
it (the one usually cited for it); leave it empty if you do not know it. Each name is checked \
against OpenAlex and dropped when no paper confirms it."""

PROPOSE_DOMAIN_PROMPT = """\
You advise the authors of the computer-science paper below on benchmarks to evaluate on. \
Name up to 15 datasets, benchmarks or evaluation suites that are specific to this paper's \
subfield and task: the field's own collections (medical, educational, security, robotics, \
networking, forensics, scientific and similar domain data), competition and challenge data, \
and the test sets the specialist papers on this exact problem report on. Leave out the \
general-purpose benchmarks everyone in the field would name first; those are asked for \
separately. Choose only established, obtainable benchmarks under their standard names; do \
not invent names. Skip the benchmarks the paper already uses. For each, give the title of \
the paper that introduced it (the one usually cited for it); leave it empty if you do not \
know it. Each name is checked against OpenAlex and dropped when no paper confirms it."""

DATE_PROMPT = """\
For each numbered benchmark or dataset name, give the title of the paper or report that introduced it \
and the year it became available. If you do not know it, give an empty title and the year 0. Each \
title is checked against OpenAlex; the year counts only when that record confirms it."""

RANK_PROMPT = """\
You advise the authors of the paper below on benchmarks they should consider adding. \
The numbered candidates were found in the paper's references, in the most-cited papers \
citing it (for a published paper; only benchmarks that existed before it), in papers that \
use the same benchmarks, in the bibliographies of those related papers (a benchmark's \
source paper cited by many of them), among benchmarks named from the paper's topic and \
confirmed in OpenAlex, and in public dataset catalogs; each shows the evidence for it, with \
cited and citing papers counted separately, and the list is in order of evidence, strongest first.
- Choose 10 that would most strengthen this paper's evaluation. The report orders them \
by evidence, and by your order among equals, so list the best first; when a pick rests on \
weaker evidence, say so in its sentence. Return fewer than 10 only if fewer than 10 \
candidates are datasets or benchmarks related to the paper's task.
- Prefer established benchmarks that reviewers in this field would expect, and those \
with stronger evidence (used by more related papers). Catalog hits are keyword matches: \
prefer recognized benchmarks, and never pick an entry that is a paper title or a \
collection unrelated to the task.
- Skip benchmarks the paper already uses, near-duplicates of one already chosen, and \
candidates that do not fit its task.
- For each, write one sentence on why it fits this paper.
- Choose only from the numbered candidates."""

# The recency rule (version 2, change 2): appended to the ranking prompt only when candidates are dated, so that
# with dating off the ranker sees exactly the as-run prompt and listing (13 Nov 2026).
RECENCY_RULE = ("Each candidate also shows the year it was first seen where that is known; a benchmark introduced in "
                "the last year or two has had little time to be used, so do not hold thin evidence against it: judge a "
                "recent benchmark by its fit to the task and its source, and include the recent ones that fit.")
BASE_RANK_PROMPT = RANK_PROMPT  # the as-run prompt (sha256 c6d4434f..., sonnet_rerank_manifest.json): dating off, and stage A's baseline
RANK_PROMPT = BASE_RANK_PROMPT + " " + RECENCY_RULE  # dating on


@dataclass(frozen=True)
class Prompts:
    """The editable prompts. system() joins the first two for the reading call; the others
    drive the "benchmarks to consider" steps."""

    instructions: str = INSTRUCTIONS_PROMPT
    benchmarks: str = BENCHMARKS_PROMPT
    user: str = USER_PROMPT
    plan: str = PLAN_PROMPT
    extract: str = EXTRACT_PROMPT
    label: str = LABEL_PROMPT
    propose: str = PROPOSE_PROMPT
    propose_domain: str = ""  # second proposal call for the subfield's own datasets (PROPOSE_DOMAIN_PROMPT); off by default, 12 Nov 2026: not in version 2
    date: str = DATE_PROMPT  # dating the candidates before the ranking (11 Nov 2026)
    rank: str = RANK_PROMPT

    def system(self) -> str:
        return "\n\n".join(p.strip() for p in (self.instructions, self.benchmarks) if p.strip())

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class AnalysisResult:
    analysis: PaperAnalysis
    model: str  # the model that answered; differs from the requested one if a fallback ran
    input_tokens: int
    output_tokens: int


def analyze_paper(
    paper: Paper,
    client,  # OllamaClient or anthropic.Anthropic (see llm.make_client)
    *,
    prompts: Prompts | None = None,
    model: str = DEFAULT_MODEL,
    effort: str = DEFAULT_EFFORT,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> AnalysisResult:
    prompts = prompts or Prompts()
    if not prompts.user.strip():
        raise AnalysisError("The user message is empty. Add an instruction for Claude.")
    document = {
        "type": "document",
        "source": {"type": "base64", "media_type": "application/pdf",
                   "data": base64.standard_b64encode(paper.pdf).decode("ascii")},
        "title": paper.name,
    }
    answer = ask(
        client, model=model, effort=effort, system=prompts.system(),
        content=[document, {"type": "text", "text": prompts.user}],
        schema=PaperAnalysis, max_tokens=max_tokens,
    )
    return AnalysisResult(answer.value, answer.model, answer.input_tokens, answer.output_tokens)
