from pytest import approx

from researchreadernerd.pricing import estimate_cost


def test_haiku_uses_higher_card_over_100k_input_tokens():
    assert estimate_cost("claude-haiku-5-5", 100_000, 10_000) == approx(0.010 + 0.005)
    assert estimate_cost("claude-haiku-5-5", 100_001, 10_000) == approx((100_001 * 0.50 + 10_000 * 2.50) / 1_000_000)


def test_flat_pricing_and_unknown_models():
    assert estimate_cost("claude-sonnet-5-5", 1_000_000, 0) == approx(2.0)
    assert estimate_cost("some-other-model", 1, 1) is None
