"""Ask Claude to read the paper and return its assumptions (R1) and benchmarks (R2)."""

from __future__ import annotations

import base64
from dataclasses import asdict, dataclass

import anthropic

from .llm import FALLBACK_BETA, FALLBACK_MODELS, AnalysisError, ask, output_format
from .paper import Paper
from .schema import PaperAnalysis

DEFAULT_MODEL = "claude-haiku-5-5"
DEFAULT_EFFORT = "medium"
# Room for the model's reasoning plus long results tables. Above ~16K the response
# must be streamed, or the HTTP request can time out.
DEFAULT_MAX_TOKENS = 32000

OUTPUT_FORMAT = output_format(PaperAnalysis)

__all__ = ["AnalysisError", "FALLBACK_BETA", "FALLBACK_MODELS"]  # re-exported for callers and tests

INSTRUCTIONS_PROMPT = """\
You are a careful reviewer reading a scientific paper for a researcher who wants to \
understand it and check it. Work only from the attached paper. Do not add facts, \
numbers, or citations from your own knowledge: if something is not in the paper, \
leave it out or mention it under caveats.

Summary: keep it under 150 words. Cover the problem, the approach, and the main claim."""

ASSUMPTIONS_PROMPT = """\
Assumptions (R1):
- List only the top 5 assumptions the method, theory, and experiments rely on: the ones \
whose failure would most change the results or claims. Order them most important first.
- Explicit assumptions are ones the paper states. Implicit assumptions are ones the \
results depend on but the paper never states; include them only when you can point \
to the passage that relies on them.
- Give each one a precise location (section, equation, table, or page) and a short \
verbatim quote from that location."""

BENCHMARKS_PROMPT = """\
Benchmarks (R2):
- List every dataset, benchmark, or environment used for evaluation, including ones \
used only in ablations or the appendix.
- For each, record every number reported on it, for the proposed method and for every \
baseline, copied exactly as printed. Note which table or figure each number is from.
- Record how the paper cites the benchmark, copied from its bibliography.
- Under notes, flag anything that limits comparison with other papers, such as custom \
splits or baselines rerun by the authors instead of copied from the original papers."""

USER_PROMPT = "Read the attached paper and extract its assumptions and benchmarks."

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

RANK_PROMPT = """\
You advise the authors of the paper below on benchmarks they should consider adding. \
The numbered candidates were found in the paper's references, in papers that use the \
same benchmarks, in the bibliographies of those related papers (a benchmark's source \
paper cited by many of them), and in public dataset catalogs; each shows the evidence \
for it.
- Choose 10 that would most strengthen this paper's evaluation, most useful first. \
Put the best-supported first; when a pick rests on weaker evidence, say so in its \
sentence. Return fewer than 10 only if fewer than 10 candidates are datasets or \
benchmarks related to the paper's task.
- Prefer established benchmarks that reviewers in this field would expect, and those \
with stronger evidence (used by more related papers). Catalog hits are keyword matches: \
prefer recognized benchmarks, and never pick an entry that is a paper title or a \
collection unrelated to the task.
- Skip benchmarks the paper already uses, near-duplicates of one already chosen, and \
candidates that do not fit its task.
- For each, write one sentence on why it fits this paper.
- Choose only from the numbered candidates."""


@dataclass(frozen=True)
class Prompts:
    """The editable prompts. system() joins the first three for the R1/R2 call; the others
    drive the "benchmarks to consider" steps."""

    instructions: str = INSTRUCTIONS_PROMPT
    assumptions: str = ASSUMPTIONS_PROMPT
    benchmarks: str = BENCHMARKS_PROMPT
    user: str = USER_PROMPT
    plan: str = PLAN_PROMPT
    extract: str = EXTRACT_PROMPT
    label: str = LABEL_PROMPT
    rank: str = RANK_PROMPT

    def system(self) -> str:
        return "\n\n".join(p.strip() for p in (self.instructions, self.assumptions, self.benchmarks) if p.strip())

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
    client: anthropic.Anthropic,
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
