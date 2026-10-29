"""One Claude call that must answer in a given JSON format (a pydantic model)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

import anthropic
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

# On a safety-classifier decline, the API re-runs the request on Anthropic's
# recommended fallback model instead of returning the refusal. Haiku has no
# server-side fallback (the parameter must not be sent), so only these models get it.
FALLBACK_BETA = "server-side-fallback-2026-07-01"
FALLBACK_MODELS = {"claude-sonnet-5-5", "claude-opus-5-5"}


class AnalysisError(Exception):
    """Claude did not return a usable answer."""


@dataclass(frozen=True)
class Answer(Generic[T]):
    value: T
    model: str  # the model that answered; differs from the requested one if a fallback ran
    input_tokens: int
    output_tokens: int


def output_format(schema: type[BaseModel]) -> dict:
    return {"type": "json_schema", "schema": anthropic.transform_schema(schema.model_json_schema())}


def ask(
    client: anthropic.Anthropic,
    *,
    model: str,
    effort: str,
    system: str,
    content: list[dict],
    schema: type[T],
    max_tokens: int,
) -> Answer[T]:
    fallback = {"betas": [FALLBACK_BETA], "fallbacks": "default"} if model in FALLBACK_MODELS else {}
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
