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

# Answers for the "benchmarks to consider" steps, keyed by a property of each step's schema.
SAMPLE_PLAN = {
    "field": "Machine learning",
    "themes": ["image classification"],
    "catalog_queries": ["image classification"],
    "source_titles": [{"benchmark": "ToyBench", "title": "ToyBench: a toy benchmark"}],
}
SAMPLE_EXTRACTION = {"papers": [{"paper": 0, "benchmarks": ["CIFAR-10"]}, {"paper": 1, "benchmarks": ["CIFAR-10", "ToyBench"]}]}
SAMPLE_DATASET_PAPERS = {"dataset_papers": [{"item": 0, "benchmark": "CelebA"}, {"item": 7, "benchmark": "Not listed"}]}
SAMPLE_RANKING = {"picks": [{"candidate": 0, "why": "The standard small image benchmark."},
                            {"candidate": 99, "why": "Not a real candidate."}]}


def fake_message(text: str, stop_reason: str = "end_turn", stop_details=None) -> dict:
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-haiku-5-5",
        "content": [{"type": "text", "text": text}],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "stop_details": stop_details,
        "usage": {"input_tokens": 10, "output_tokens": 10},
    }


def as_event_stream(message: dict) -> bytes:
    """Replay a complete message as the server-sent events of a streamed response."""
    start = {**message, "content": [], "stop_reason": None, "stop_details": None,
             "usage": {"input_tokens": message["usage"]["input_tokens"], "output_tokens": 0}}
    events = [("message_start", {"type": "message_start", "message": start})]
    for i, block in enumerate(message["content"]):
        events += [
            ("content_block_start", {"type": "content_block_start", "index": i, "content_block": {"type": "text", "text": ""}}),
            ("content_block_delta", {"type": "content_block_delta", "index": i, "delta": {"type": "text_delta", "text": block["text"]}}),
            ("content_block_stop", {"type": "content_block_stop", "index": i}),
        ]
    events += [
        ("message_delta", {"type": "message_delta",
                           "delta": {"stop_reason": message["stop_reason"], "stop_sequence": None, "stop_details": message["stop_details"]},
                           "usage": {"output_tokens": message["usage"]["output_tokens"]}}),
        ("message_stop", {"type": "message_stop"}),
    ]
    return "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events).encode()


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
            body = json.loads(request.content)
            self.requests.append(body)
            self.headers.append(request.headers)
            reply = self._reply_for(body)
            if body.get("stream"):
                return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=as_event_stream(reply))
            return httpx.Response(200, json=reply)

        def _reply_for(self, body: dict) -> dict:
            props = body.get("output_config", {}).get("format", {}).get("schema", {}).get("properties", {})
            for key, sample in (("catalog_queries", SAMPLE_PLAN), ("picks", SAMPLE_RANKING), ("papers", SAMPLE_EXTRACTION),
                                ("dataset_papers", SAMPLE_DATASET_PAPERS)):
                if key in props:
                    return fake_message(json.dumps(sample))
            return self.reply

    return FakeAPI()


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Stub OpenAlex and the dataset catalogs: tests never reach the network."""
    from researchreadernerd import consider, scholar
    from researchreadernerd.catalogs import CatalogHit

    def work(i, title, refs=()):
        return scholar.Work(id=f"W{i}", title=title, abstract=f"We evaluate on things ({title}).", year=2020,
                            url=f"https://doi.org/10.0/{i}", cited_by=10, references=tuple(refs))

    # Both related papers cite W9, a dataset paper, and W2, ToyBench's own source paper.
    library = {"W1": work(1, "A referenced paper", ["W9", "W2", "W8"]),
               "W9": work(9, "CelebA: a large-scale face attributes dataset")}
    monkeypatch.setattr(scholar, "find_paper", lambda title, arxiv_id=None: (work(0, title), ["W1"]))
    monkeypatch.setattr(scholar, "get_works", lambda ids: [library[i] for i in ids if i in library])
    monkeypatch.setattr(scholar, "find_by_title", lambda title: work(2, title))
    monkeypatch.setattr(scholar, "citing_works", lambda work_id, limit=20: [work(3, "A paper using ToyBench", ["W9", "W2"])])
    monkeypatch.setattr(consider, "search_all", lambda queries, per_query=4: (
        [CatalogHit("TPC", "TPC-H", "decision support", "https://www.tpc.org/tpch/")], []))
