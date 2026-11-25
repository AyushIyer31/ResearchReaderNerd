"""The local-model path: Ollama's chat API, PDF text extraction, and context fitting."""

import httpx2 as httpx
import pytest

from researchreadernerd.analyze import AnalysisError, Prompts, analyze_paper
from researchreadernerd.consider import consider_benchmarks
from researchreadernerd.llm import OllamaClient, OllamaError, is_local_model, make_client
from researchreadernerd.paper import Paper, PaperError, pdf_text
from researchreadernerd.pricing import Spend
from researchreadernerd.run import describe_error

from .conftest import TEXT_PDF, make_pdf

PAPER = Paper(name="2401.01234", pdf=TEXT_PDF, source="https://arxiv.org/pdf/2401.01234")


def test_pdf_text_extracts_pages():
    assert pdf_text(TEXT_PDF) == "[page 1]\nSampleNet beats baselines on ToyBench."


def test_pdf_without_text_is_an_error():
    with pytest.raises(PaperError, match="no extractable text"):
        pdf_text(make_pdf(""))
    with pytest.raises(PaperError, match="could not be extracted"):
        pdf_text(b"%PDF-1.4\n%fake pdf for tests\n")


def test_request_shape(fake_ollama):
    result = analyze_paper(PAPER, fake_ollama.client)
    body = fake_ollama.requests[0]

    assert body["model"] == "qwen2.5:14b"
    assert body["stream"] is True
    assert body["options"] == {"num_ctx": 32768, "num_predict": 8192, "temperature": 0}
    assert {"benchmarks", "caveats"} <= set(body["format"]["required"])
    system, user = body["messages"]
    assert system["role"] == "system" and "Benchmarks (R2):" in system["content"]
    assert user["role"] == "user"
    assert "SampleNet beats baselines on ToyBench." in user["content"]  # the PDF, as text
    assert user["content"].endswith(Prompts().user)  # followed by the instruction
    assert "2401.01234" in user["content"]  # the document title

    assert (result.model, result.input_tokens, result.output_tokens) == ("qwen2.5:14b", 10, 10)
    assert result.analysis.proposed_method == "SampleNet"


def test_long_paper_is_cut_to_fit_the_context(fake_ollama):
    fake_ollama.client.num_ctx = 1000  # smaller than the answer budget: the floor of 1000 characters applies
    long_paper = Paper(name="long", pdf=make_pdf("word " * 800), source="x")
    analyze_paper(long_paper, fake_ollama.client)
    user = fake_ollama.requests[0]["messages"][1]["content"]
    assert "cut off here to fit the model's context window" in user
    assert "Say so under caveats" in user
    assert user.count("word") < 800


def test_short_paper_is_not_cut(fake_ollama):
    analyze_paper(PAPER, fake_ollama.client)
    assert "cut off" not in fake_ollama.requests[0]["messages"][1]["content"]


def test_truncated_answer_raises(fake_ollama):
    fake_ollama.done_reason = "length"
    with pytest.raises(AnalysisError, match="token limit"):
        analyze_paper(PAPER, fake_ollama.client)


def test_malformed_answer_raises(fake_ollama):
    fake_ollama.reply = '{"title": "missing everything else"}'
    with pytest.raises(AnalysisError, match="expected format"):
        analyze_paper(PAPER, fake_ollama.client)


def test_model_not_pulled_is_explained(fake_ollama):
    fake_ollama.status = 404
    with pytest.raises(OllamaError, match="ollama pull qwen2.5:14b"):
        analyze_paper(PAPER, fake_ollama.client)


def test_unreachable_server_is_explained():
    def refuse(request):
        raise httpx.ConnectError("connection refused")

    client = OllamaClient(base_url="localhost:11434", http=httpx.Client(transport=httpx.MockTransport(refuse)))
    assert client.base_url == "http://localhost:11434"  # a bare host:port, as OLLAMA_HOST is often written
    with pytest.raises(OllamaError, match="Could not reach Ollama at http://localhost:11434"):
        analyze_paper(PAPER, client)


def test_errors_have_plain_messages():
    assert describe_error(OllamaError("Could not reach Ollama")) == "Could not reach Ollama"


def test_make_client_picks_the_backend():
    assert is_local_model("qwen2.5:14b") and not is_local_model("claude-haiku-5-5")
    assert isinstance(make_client("qwen2.5:14b"), OllamaClient)
    assert isinstance(make_client("llama3.3"), OllamaClient)


def test_benchmark_search_runs_on_the_local_model(fake_ollama):
    analysis = analyze_paper(PAPER, fake_ollama.client).analysis
    spend = Spend()
    c = consider_benchmarks(analysis, client=fake_ollama.client, model="qwen2.5:14b", effort="medium",
                            prompts=Prompts(), spend=spend, budget_usd=1.0)
    assert c.ranker_picks == ["CIFAR-10"] and c.suggestions[-1].name == "CIFAR-10"
    assert spend.cost == 0.0  # no price card for a local model
    assert {r["model"] for r in fake_ollama.requests} == {"qwen2.5:14b"}
