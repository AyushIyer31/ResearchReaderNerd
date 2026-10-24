"""Structured output Claude must return for R1 (assumptions) and R2 (benchmarks)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Assumption(_Strict):
    statement: str = Field(description="The assumption, in one plain sentence.")
    kind: Literal["explicit", "implicit"] = Field(
        description="explicit: the paper states it. implicit: the results depend on it but the paper never states it."
    )
    category: Literal["modeling", "data", "statistical", "experimental_setup", "scope", "other"]
    location: str = Field(description="Where in the paper, e.g. 'Section 3.2', 'Eq. 4', 'Table 2'.")
    evidence: str = Field(
        description="A short verbatim quote (under 40 words) from that location. For implicit assumptions, the passage that relies on it."
    )
    why_it_matters: str = Field(description="What would change in the results or claims if it did not hold.")


class Result(_Strict):
    method: str = Field(description="Method name exactly as the paper labels it.")
    metric: str
    value: str = Field(description="The number exactly as printed, with any ± or units.")
    is_proposed_method: bool = Field(description="True for the paper's own method or its variants.")
    location: str = Field(description="Table or figure the number comes from, e.g. 'Table 3'.")


class Benchmark(_Strict):
    name: str = Field(description="Dataset, benchmark, or environment name.")
    task: str = Field(description="What is evaluated on it, e.g. 'image classification'.")
    splits: list[str] = Field(description="Splits or subsets used, e.g. ['test'], ['val', 'test-dev']. Empty if not stated.")
    metrics: list[str]
    cited_as: str | None = Field(
        description="The reference the paper gives for this benchmark, as written in its bibliography, or null if none."
    )
    results: list[Result] = Field(description="Every reported number on this benchmark: the proposed method and all baselines.")
    notes: str | None = Field(description="Anything that limits comparison: custom splits, reruns vs copied numbers, preprocessing.")


class PaperAnalysis(_Strict):
    title: str
    authors: list[str]
    proposed_method: str = Field(description="Name of the paper's main method or contribution.")
    summary: str = Field(description="Two or three sentences: the problem, the approach, and the main claim.")
    assumptions: list[Assumption]
    benchmarks: list[Benchmark]
    caveats: list[str] = Field(
        description="Things the paper leaves unclear that affect R1 or R2, e.g. missing split details or unreadable tables."
    )
