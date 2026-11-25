"""One model call that must answer in a given JSON format (a pydantic model).

Two backends share the same `ask()`:
- a local model served by Ollama (the default: Qwen2.5 14B), which cannot read a PDF
  itself, so the paper's text is extracted and fitted into its context window;
- Claude, for any model name starting with "claude-".
"""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass, field
from typing import Generic, TypeVar

import anthropic
import httpx2
from pydantic import BaseModel, ValidationError

from .paper import pdf_text

T = TypeVar("T", bound=BaseModel)

# On a safety-classifier decline, the API re-runs the request on Anthropic's
# recommended fallback model instead of returning the refusal. Haiku has no
# server-side fallback (the parameter must not be sent), so only these models get it.
FALLBACK_BETA = "server-side-fallback-2026-07-01"
FALLBACK_MODELS = {"claude-sonnet-5-5", "claude-opus-5-5"}

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
# Qwen2.5's native context. Raise it with RRN_NUM_CTX when the model and memory allow.
DEFAULT_NUM_CTX = 32768
# Tokens kept free for the answer; the rest of the context holds the prompt and the paper.
MAX_LOCAL_OUTPUT = 8192
# A conservative estimate for English prose with numbers and symbols, used to fit the paper.
CHARS_PER_TOKEN = 3.5


class AnalysisError(Exception):
    """The model did not return a usable answer."""


class OllamaError(Exception):
    """Ollama could not be reached, or the model is not pulled."""


@dataclass(frozen=True)
class Answer(Generic[T]):
    value: T
    model: str  # the model that answered; differs from the requested one if a fallback ran
    input_tokens: int
    output_tokens: int


@dataclass
class OllamaClient:
    """A connection to a local Ollama server. `http` can be replaced in tests."""

    base_url: str = field(default_factory=lambda: os.environ.get("OLLAMA_HOST") or DEFAULT_OLLAMA_URL)
    num_ctx: int = field(default_factory=lambda: int(os.environ.get("RRN_NUM_CTX") or DEFAULT_NUM_CTX))
    http: httpx2.Client | None = None

    def __post_init__(self) -> None:
        if "://" not in self.base_url:  # OLLAMA_HOST is often just host:port
            self.base_url = "http://" + self.base_url
        self.base_url = self.base_url.rstrip("/")
        if self.http is None:
            # A 14B model takes minutes per answer: no read timeout, only a connect timeout.
            self.http = httpx2.Client(timeout=httpx2.Timeout(30.0, read=None))


def is_local_model(model: str) -> bool:
    return not model.startswith("claude-")


def make_client(model: str):
    """The client for a model name: Ollama for local models, the Anthropic SDK for Claude."""
    if is_local_model(model):
        return OllamaClient()
    # Streams occasionally stall on a dead connection; give up after 3 minutes without a chunk
    # so the SDK's retries kick in, instead of its 10-minute default.
    return anthropic.Anthropic(timeout=httpx2.Timeout(600.0, connect=10.0, read=180.0))


def output_format(schema: type[BaseModel]) -> dict:
    return {"type": "json_schema", "schema": anthropic.transform_schema(schema.model_json_schema())}


def ask(
    client,
    *,
    model: str,
    effort: str,
    system: str,
    content: list[dict],
    schema: type[T],
    max_tokens: int,
) -> Answer[T]:
    """`content` is in the Anthropic block format (text and document blocks). `effort` only
    applies to Claude; a local model answers at a fixed temperature of 0."""
    if isinstance(client, OllamaClient):
        return _ask_ollama(client, model=model, system=system, content=content, schema=schema, max_tokens=max_tokens)
    return _ask_claude(client, model=model, effort=effort, system=system, content=content, schema=schema,
                       max_tokens=max_tokens)


# ---------- Ollama ----------

def _ask_ollama(client: OllamaClient, *, model: str, system: str, content: list[dict], schema: type[T],
                max_tokens: int) -> Answer[T]:
    num_predict = min(max_tokens, MAX_LOCAL_OUTPUT)
    budget = int((client.num_ctx - num_predict - 256) * CHARS_PER_TOKEN) - len(system)
    text = flatten_content(content, budget)
    body = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": text}],
        "stream": True,
        "format": schema.model_json_schema(),
        "options": {"num_ctx": client.num_ctx, "num_predict": num_predict, "temperature": 0},
        "keep_alive": "15m",
    }
    parts: list[str] = []
    final: dict | None = None
    try:
        with client.http.stream("POST", f"{client.base_url}/api/chat", json=body) as response:
            if response.status_code != 200:
                response.read()
                raise OllamaError(_describe_status(model, response))
            for line in response.iter_lines():
                if not line.strip():
                    continue
                event = json.loads(line)
                if event.get("error"):
                    raise OllamaError(f"Ollama reported an error: {event['error']}")
                parts.append(event.get("message", {}).get("content", ""))
                if event.get("done"):
                    final = event
    except httpx2.HTTPError as e:
        raise OllamaError(
            f"Could not reach Ollama at {client.base_url} ({e.__class__.__name__}). "
            "Start it with `ollama serve`, or set OLLAMA_HOST."
        ) from e
    if final is None:
        raise OllamaError("Ollama closed the stream before the answer was finished.")
    if final.get("done_reason") == "length":
        raise AnalysisError(
            f"The answer hit the {num_predict}-token limit before finishing. "
            "Raise RRN_NUM_CTX, or try a paper with fewer results tables."
        )
    answer = "".join(parts)
    try:
        value = schema.model_validate_json(answer)
    except ValidationError as e:
        raise AnalysisError(f"The answer from {model} did not match the expected format: {e}") from e
    return Answer(value, final.get("model") or model, int(final.get("prompt_eval_count") or 0),
                  int(final.get("eval_count") or 0))


def _describe_status(model: str, response) -> str:
    try:
        message = response.json().get("error", "")
    except ValueError:
        message = response.text
    if response.status_code == 404 or "not found" in message:
        return f"Ollama does not have the model {model!r}. Pull it with: ollama pull {model}"
    return f"Ollama error {response.status_code}: {message}"


def flatten_content(content: list[dict], budget: int) -> str:
    """Turn Anthropic-style blocks into one text message: text blocks as they are, PDF
    document blocks as extracted text. Documents are cut to fit `budget` characters
    after the other text; a cut is marked so the model can report it under caveats."""
    texts = [block["text"] for block in content if block["type"] == "text"]
    documents = [block for block in content if block["type"] == "document"]
    remaining = budget - sum(len(t) for t in texts)
    per_document = max(remaining // len(documents), 1000) if documents else 0
    pieces: list[str] = []
    for block in content:
        if block["type"] == "text":
            pieces.append(block["text"])
        elif block["type"] == "document":
            text = pdf_text(base64.standard_b64decode(block["source"]["data"]))
            if len(text) > per_document:
                omitted = len(text) - per_document
                text = text[:per_document] + (
                    f"\n\n[The paper is cut off here to fit the model's context window; {omitted:,} characters "
                    "of the end of the paper were omitted. Say so under caveats.]"
                )
            name = block.get("title")
            label = f"The paper (from the file “{name}”)" if name else "The paper"
            pieces.append(f"{label}, as text extracted from its PDF:\n\n{text}\n\n(end of paper)")
    return "\n\n".join(pieces)


# ---------- Claude ----------

def _ask_claude(client: anthropic.Anthropic, *, model: str, effort: str, system: str, content: list[dict],
                schema: type[T], max_tokens: int) -> Answer[T]:
    # RRN_SERVER_FALLBACK=0 (launch config, 13 Nov 2026): no server-side fallback, so every answer comes from the
    # model named; a decline is an error, recorded and retried, never an answer from another model.
    use_fallback = model in FALLBACK_MODELS and os.environ.get("RRN_SERVER_FALLBACK", "1") != "0"
    fallback = {"betas": [FALLBACK_BETA], "fallbacks": "default"} if use_fallback else {}
    # Streamed (a long answer over a plain request can time out) and read as plain text rather
    # than with parse(): parse() validates the JSON before we can see the stop reason, so a
    # refusal or a cut-off answer would surface as a confusing validation error.
    with client.beta.messages.stream(
        model=model,
        max_tokens=max_tokens,
        output_config={"effort": effort, "format": output_format(schema)},
        system=system,
        messages=[{"role": "user", "content": content}],
        **fallback,
    ) as stream:
        response = stream.get_final_message()
    # Ledger first: a refused, cut-off or malformed answer is billed too, so it is counted before any check raises.
    _ledger(model, response.model, response.usage.input_tokens, response.usage.output_tokens)

    if response.stop_reason == "refusal":
        category = response.stop_details.category if response.stop_details else None
        raise AnalysisError(f"Claude declined this request (category: {category}).")
    if response.stop_reason == "max_tokens":
        raise AnalysisError(
            f"The answer hit the {max_tokens}-token limit before finishing. Try a lower effort level."
        )
    text = next((block.text for block in response.content if block.type == "text"), None)
    if not text:
        raise AnalysisError(f"Claude returned no answer (stop reason: {response.stop_reason}).")
    try:
        value = schema.model_validate_json(text)
    except ValidationError as e:
        raise AnalysisError(f"Claude's answer did not match the expected format: {e}") from e
    return Answer(value, response.model, response.usage.input_tokens, response.usage.output_tokens)


def _ledger(requested: str, answered: str, input_tokens: int, output_tokens: int) -> None:
    """RRN_SPEND_LEDGER=<file> (launch runner): one JSON line per Claude response that arrives (refused, cut-off and
    malformed answers included), the judge's too, so a stage stops at its cap. Unset, nothing is written."""
    path = os.environ.get("RRN_SPEND_LEDGER")
    if not path:
        return
    from datetime import datetime, timezone
    from .pricing import estimate_cost
    line = json.dumps({"t": datetime.now(timezone.utc).isoformat(timespec="seconds"), "requested": requested, "model": answered,
                       "input_tokens": input_tokens, "output_tokens": output_tokens,
                       "usd": estimate_cost(answered, input_tokens, output_tokens)})
    with open(path, "a") as f:
        f.write(line + "\n")
