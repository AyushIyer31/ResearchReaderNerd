import base64

import pytest

from researchreadernerd.analyze import FALLBACK_BETA, AnalysisError, Prompts, analyze_paper
from researchreadernerd.paper import Paper

from .conftest import MINIMAL_PDF, fake_message

PAPER = Paper(name="2401.01234", pdf=MINIMAL_PDF, source="https://arxiv.org/pdf/2401.01234")


def test_request_shape(fake_api):
    analyze_paper(PAPER, fake_api.client)
    body = fake_api.requests[0]

    assert body["model"] == "claude-haiku-5-5"
    assert body["output_config"]["effort"] == "high"
    assert "fallbacks" not in body  # Haiku has no server-side fallback; sending it is an error
    assert FALLBACK_BETA not in fake_api.headers[0].get("anthropic-beta", "")

    document, instruction = body["messages"][0]["content"]
    assert document["type"] == "document"
    assert document["source"]["media_type"] == "application/pdf"
    assert base64.b64decode(document["source"]["data"]) == MINIMAL_PDF
    assert instruction["type"] == "text"

    schema = body["output_config"]["format"]["schema"]
    assert {"assumptions", "benchmarks", "caveats"} <= set(schema["required"])


def test_parses_analysis(fake_api):
    result = analyze_paper(PAPER, fake_api.client)
    assert (result.model, result.input_tokens, result.output_tokens) == ("claude-haiku-5-5", 10, 10)
    analysis = result.analysis
    assert analysis.proposed_method == "SampleNet"
    assert [a.kind for a in analysis.assumptions] == ["implicit", "explicit"]
    assert analysis.benchmarks[0].results[0].is_proposed_method


def test_refusal_raises(fake_api):
    fake_api.reply = fake_message(
        "", stop_reason="refusal", stop_details={"type": "refusal", "category": "bio", "explanation": None}
    )
    with pytest.raises(AnalysisError, match="declined.*bio"):
        analyze_paper(PAPER, fake_api.client)


def test_truncation_raises(fake_api):
    fake_api.reply = fake_message('{"title": "cut off', stop_reason="max_tokens")
    with pytest.raises(AnalysisError, match="--max-tokens"):
        analyze_paper(PAPER, fake_api.client, max_tokens=100)


def test_malformed_output_raises(fake_api):
    fake_api.reply = fake_message('{"title": "missing everything else"}')
    with pytest.raises(AnalysisError, match="expected format"):
        analyze_paper(PAPER, fake_api.client)


def test_default_system_prompt_has_all_parts(fake_api):
    analyze_paper(PAPER, fake_api.client)
    system = fake_api.requests[0]["system"]
    assert "careful reviewer" in system and "Assumptions (R1):" in system and "Benchmarks (R2):" in system


def test_custom_prompts_are_sent(fake_api):
    prompts = Prompts(instructions="Be brief.", assumptions="", benchmarks="List datasets.", user="Go.")
    analyze_paper(PAPER, fake_api.client, prompts=prompts)
    body = fake_api.requests[0]
    assert body["system"] == "Be brief.\n\nList datasets."
    assert body["messages"][0]["content"][1]["text"] == "Go."


def test_empty_user_message_rejected(fake_api):
    with pytest.raises(AnalysisError, match="user message is empty"):
        analyze_paper(PAPER, fake_api.client, prompts=Prompts(user="  "))
    assert fake_api.requests == []


def test_sonnet_gets_server_side_fallback(fake_api):
    analyze_paper(PAPER, fake_api.client, model="claude-sonnet-5-5")
    assert fake_api.requests[0]["fallbacks"] == "default"
    assert FALLBACK_BETA in fake_api.headers[0]["anthropic-beta"]
