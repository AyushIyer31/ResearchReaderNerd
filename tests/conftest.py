import json

import anthropic
import httpx2 as httpx
import pytest

SAMPLE_ANALYSIS = {
    "title": "A Sample Paper",
    "authors": ["Ada Lovelace", "Alan Turing"],
    "proposed_method": "SampleNet",
    "summary": "The paper proposes SampleNet. It beats baselines on ToyBench.",
    "assumptions": [
        {
            "statement": "Training and test data come from the same distribution.",
            "kind": "implicit",
            "category": "data",
            "location": "Section 4.1",
            "evidence": "We evaluate on the held-out split.",
            "why_it_matters": "Accuracy may drop under distribution shift.",
        },
        {
            "statement": "Noise is Gaussian.",
            "kind": "explicit",
            "category": "statistical",
            "location": "Eq. 3",
            "evidence": "We assume the noise is Gaussian.",
            "why_it_matters": "The likelihood in Eq. 3 is wrong for heavy-tailed noise.",
        },
    ],
    "benchmarks": [
        {
            "name": "ToyBench",
            "task": "classification",
            "splits": ["test"],
            "metrics": ["accuracy"],
            "cited_as": "Doe et al. ToyBench. 2020.",
            "results": [
                {"method": "SampleNet", "metric": "accuracy", "value": "91.2 ± 0.3", "is_proposed_method": True, "location": "Table 2"},
                {"method": "Baseline | v2", "metric": "accuracy", "value": "88.0", "is_proposed_method": False, "location": "Table 2"},
            ],
            "notes": "Baselines rerun by the authors.",
        }
    ],
    "caveats": ["The validation split size is not stated."],
}

MINIMAL_PDF = b"%PDF-1.4\n%fake pdf for tests\n"


def fake_message(text: str, stop_reason: str = "end_turn", stop_details=None) -> dict:
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-5-5",
        "content": [{"type": "text", "text": text}],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "stop_details": stop_details,
        "usage": {"input_tokens": 10, "output_tokens": 10},
    }


@pytest.fixture
def fake_api():
    """An Anthropic client backed by a fake transport. Set `.reply` to change the response."""

    class FakeAPI:
        def __init__(self):
            self.requests: list[dict] = []
            self.headers: list[httpx.Headers] = []
            self.reply = fake_message(json.dumps(SAMPLE_ANALYSIS))
            transport = httpx.MockTransport(self._handle)
            self.client = anthropic.Anthropic(
                api_key="test-key", http_client=httpx.Client(transport=transport), max_retries=0
            )

        def _handle(self, request: httpx.Request) -> httpx.Response:
            self.requests.append(json.loads(request.content))
            self.headers.append(request.headers)
            return httpx.Response(200, json=self.reply)

    return FakeAPI()
