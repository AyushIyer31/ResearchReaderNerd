"""Structured output the model must return for R2 (benchmarks)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


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
    title: str = Field(description="The paper's own title, exactly as printed on its first page; "
                                   "not the file name and not a description of this task.")
    authors: list[str] = Field(description="The authors as printed on the first page.")
    proposed_method: str = Field(description="Name of the paper's main method or contribution.")
    summary: str = Field(description="Under 150 words: the problem, the approach, and the main claim.")
    benchmarks: list[Benchmark]
    caveats: list[str] = Field(
        description="Things the paper leaves unclear that affect the benchmarks, e.g. missing split details or unreadable tables."
    )
