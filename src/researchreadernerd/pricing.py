"""Token prices and cost estimates."""

from __future__ import annotations

from dataclasses import dataclass, field

# USD per million tokens: (input, output), and for Haiku a higher card once the
# prompt is over 100K tokens. Used for the cost estimate shown after a run.
PRICES = {
    "claude-haiku-5-5": {"small": (0.10, 0.50), "large": (0.50, 2.50), "threshold": 100_000},
    "claude-sonnet-5-5": {"small": (2.0, 10.0)},
    "claude-opus-5-5": {"small": (4.0, 20.0)},
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    price = PRICES.get(model)
    if price is None:
        return None
    card = price["large"] if "threshold" in price and input_tokens > price["threshold"] else price["small"]
    return (input_tokens * card[0] + output_tokens * card[1]) / 1_000_000


@dataclass
class Spend:
    """Running total of tokens and estimated cost across several Claude calls."""

    calls: list[dict] = field(default_factory=list)

    def add(self, step: str, model: str, input_tokens: int, output_tokens: int) -> None:
        self.calls.append({"step": step, "model": model, "input_tokens": input_tokens, "output_tokens": output_tokens,
                           "estimated_cost_usd": estimate_cost(model, input_tokens, output_tokens)})

    @property
    def cost(self) -> float:
        return sum(c["estimated_cost_usd"] or 0.0 for c in self.calls)

    @property
    def input_tokens(self) -> int:
        return sum(c["input_tokens"] for c in self.calls)

    @property
    def output_tokens(self) -> int:
        return sum(c["output_tokens"] for c in self.calls)
