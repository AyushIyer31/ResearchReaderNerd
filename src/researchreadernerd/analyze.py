"""Ask Claude to read the paper and return its assumptions (R1) and benchmarks (R2)."""

from __future__ import annotations

import base64
from dataclasses import asdict, dataclass

import anthropic
from pydantic import ValidationError

from .paper import Paper
from .schema import PaperAnalysis

DEFAULT_MODEL = "claude-sonnet-5-5"
DEFAULT_EFFORT = "high"
DEFAULT_MAX_TOKENS = 16000

OUTPUT_FORMAT = {"type": "json_schema", "schema": anthropic.transform_schema(PaperAnalysis.model_json_schema())}

# On a safety-classifier decline, the API re-runs the request on Anthropic's
# recommended fallback model instead of returning the refusal.
FALLBACK_BETA = "server-side-fallback-2026-07-01"

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


@dataclass(frozen=True)
class Prompts:
    """The editable prompt text for the analysis step. The system prompt joins the first three."""

    instructions: str = INSTRUCTIONS_PROMPT
    assumptions: str = ASSUMPTIONS_PROMPT
    benchmarks: str = BENCHMARKS_PROMPT
    user: str = USER_PROMPT

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


class AnalysisError(Exception):
    """Claude did not return a usable analysis."""


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

    # create() rather than parse(): parse() validates the JSON before we can see
    # the stop reason, so a refusal or truncated output would surface as a
    # confusing validation error instead of the messages below.
    response = client.beta.messages.create(
        model=model,
        max_tokens=max_tokens,
        output_config={"effort": effort, "format": OUTPUT_FORMAT},
        betas=[FALLBACK_BETA],
        fallbacks="default",
        system=prompts.system(),
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": base64.standard_b64encode(paper.pdf).decode("ascii"),
                        },
                        "title": paper.name,
                    },
                    {"type": "text", "text": prompts.user},
                ],
            }
        ],
    )

    if response.stop_reason == "refusal":
        category = response.stop_details.category if response.stop_details else None
        raise AnalysisError(f"Claude declined to analyze this paper (category: {category}).")
    if response.stop_reason == "max_tokens":
        raise AnalysisError(
            f"The analysis hit the {max_tokens}-token limit before finishing. Retry with a larger --max-tokens."
        )

    text = next((block.text for block in response.content if block.type == "text"), None)
    if not text:
        raise AnalysisError(f"Claude returned no analysis (stop reason: {response.stop_reason}).")
    try:
        analysis = PaperAnalysis.model_validate_json(text)
    except ValidationError as e:
        raise AnalysisError(f"Claude's analysis did not match the expected format: {e}") from e

    return AnalysisResult(
        analysis=analysis,
        model=response.model,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
    )
