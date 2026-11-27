"""Launch guards (Harsh Kashyap's review of the v2 package, 24 Nov 2026): OpenAlex free budget guarded per request,
Claude spend reserved before each call, and stage A's ranking prompts kept apart with recorded provenance."""
import hashlib
import json
import sys
import types
from pathlib import Path

import pytest

from researchreadernerd import llm, scholar

REAL_GET = scholar._get  # captured before the no_network fixture runs (it does not fake _get, but keep it explicit)


# ---------- OpenAlex: request-level free-budget guard ----------

class FakeHTTP:
    def __init__(self, remaining=1.0, prepaid=0.0):
        self.remaining, self.prepaid, self.calls = remaining, prepaid, []

    def get(self, url, params=None, timeout=None, **kw):
        self.calls.append(url)
        if url.endswith("/rate-limit"):
            body = {"rate_limit": {"daily_remaining_usd": self.remaining, "prepaid_remaining_usd": self.prepaid}}
        else:
            body = {"results": []}
        return types.SimpleNamespace(status_code=200, headers={}, json=lambda: body)


@pytest.fixture
def guarded(monkeypatch):
    monkeypatch.setenv("RRN_OPENALEX_FREE_ONLY", "1")
    monkeypatch.setattr(scholar, "_FREE", {"read_at": 0.0, "remaining": None, "since": 0.0})
    monkeypatch.setattr(scholar.time, "sleep", lambda s: None)

    def install(**kw):
        h = FakeHTTP(**kw)
        monkeypatch.setattr(scholar.httpx2, "get", h.get)
        return h
    return install


def test_a_priced_request_the_free_budget_cannot_cover_is_not_sent(guarded):
    h = guarded(remaining=0.0205)
    with pytest.raises(scholar.ScholarError, match="free daily budget spent"):
        REAL_GET("/works", {"filter": "title.search:x"})  # $0.001 + $0.02 margin > $0.0205
    assert not any("/works" in u and "rate-limit" not in u for u in h.calls), "the request itself was never sent"


def test_requests_within_the_budget_go_through_and_are_counted(guarded):
    h = guarded(remaining=0.5)
    REAL_GET("/works", {"filter": "cites:W1"})
    REAL_GET("/works", {"filter": "cites:W2"})
    assert sum(1 for u in h.calls if u.endswith("/works")) == 2
    assert scholar._FREE["since"] == pytest.approx(0.0002)


def test_lookups_by_id_are_free_and_never_refused(guarded):
    h = guarded(remaining=0.0)
    REAL_GET("/works/doi:10.1/x", {"select": "id"})
    assert h.calls == ["https://api.openalex.org/works/doi:10.1/x"], "no budget read, no refusal for a free lookup"


def test_a_key_with_prepaid_credit_is_refused(guarded):
    guarded(remaining=1.0, prepaid=3.18)
    with pytest.raises(scholar.ScholarError, match="prepaid credit"):
        REAL_GET("/works", {"filter": "cites:W1"})


def test_an_unreadable_budget_refuses_the_request(guarded, monkeypatch):
    guarded()
    monkeypatch.setattr(scholar.httpx2, "get", lambda url, **kw: (_ for _ in ()).throw(scholar.httpx2.ConnectError("down")))
    with pytest.raises(scholar.ScholarError, match="could not be read"):
        REAL_GET("/works", {"filter": "cites:W1"})


def test_the_guard_is_off_outside_the_launch(monkeypatch):
    monkeypatch.delenv("RRN_OPENALEX_FREE_ONLY", raising=False)
    h = FakeHTTP(remaining=0.0)
    monkeypatch.setattr(scholar.httpx2, "get", h.get)
    REAL_GET("/works", {"filter": "cites:W1"})
    assert h.calls == ["https://api.openalex.org/works"]


# ---------- Claude: reserve the worst case before each call ----------

class FakeClaude:
    def __init__(self, tokens=1000):
        self.seen = []

        def count(**kw):
            self.seen.append(kw)
            return types.SimpleNamespace(input_tokens=tokens)
        self.beta = types.SimpleNamespace(messages=types.SimpleNamespace(count_tokens=count))


class Schema(llm.BaseModel):
    x: int


def test_a_call_whose_worst_case_passes_the_cap_is_not_sent(tmp_path, monkeypatch):
    ledger = tmp_path / "spend.jsonl"
    monkeypatch.setenv("RRN_SPEND_LEDGER", str(ledger))
    monkeypatch.setenv("RRN_SPEND_CAP_USD", "0.10")
    # Sonnet: 8,000 output tokens alone cost $0.08; with the input allowance the worst case passes $0.10 after one call.
    rid = llm._reserve(FakeClaude(1000), model="claude-sonnet-5-5", system="s", content=[{"type": "text", "text": "t"}], schema=Schema, max_tokens=8000)
    assert rid and llm.ledger_spend(str(ledger)) > 0.08
    with pytest.raises(llm.SpendCapError, match="not sent"):
        llm._reserve(FakeClaude(1000), model="claude-sonnet-5-5", system="s", content=[{"type": "text", "text": "t"}], schema=Schema, max_tokens=8000)


def test_a_lost_response_stays_counted_at_its_worst_case_and_a_settled_one_at_its_price(tmp_path, monkeypatch):
    ledger = tmp_path / "spend.jsonl"
    monkeypatch.setenv("RRN_SPEND_LEDGER", str(ledger))
    monkeypatch.setenv("RRN_SPEND_CAP_USD", "5")
    lost = llm._reserve(FakeClaude(500), model="claude-haiku-5-5", system="s", content=[{"type": "text", "text": "t"}], schema=Schema, max_tokens=4000)
    done = llm._reserve(FakeClaude(500), model="claude-haiku-5-5", system="s", content=[{"type": "text", "text": "t"}], schema=Schema, max_tokens=4000)
    worst = [json.loads(x)["usd"] for x in ledger.read_text().splitlines()]
    llm._ledger("claude-haiku-5-5", "claude-haiku-5-5", 400, 100, settles=done)
    actual = llm.estimate_cost("claude-haiku-5-5", 400, 100) if hasattr(llm, "estimate_cost") else (400 * 0.10 + 100 * 0.50) / 1e6
    assert lost != done
    assert llm.ledger_spend(str(ledger)) == pytest.approx(worst[0] + actual), "the lost call at its worst case, the settled one at its price"


def test_the_worst_case_bounds_the_actual_cost():
    for model in ("claude-haiku-5-5", "claude-sonnet-5-5"):
        from researchreadernerd.pricing import estimate_cost
        assert llm.worst_case_usd(model, 5000, 8000) >= estimate_cost(model, 5000, 7999)
    assert llm.worst_case_usd("some-new-model", 1000, 1000) >= llm.worst_case_usd("claude-sonnet-5-5", 1000, 1000)


def test_uncountable_input_means_the_call_is_not_sent(tmp_path, monkeypatch):
    monkeypatch.setenv("RRN_SPEND_LEDGER", str(tmp_path / "spend.jsonl"))
    monkeypatch.setenv("RRN_SPEND_CAP_USD", "5")

    class Broken:
        beta = types.SimpleNamespace(messages=types.SimpleNamespace(count_tokens=lambda **kw: (_ for _ in ()).throw(llm.anthropic.APIConnectionError(request=None))))
    with pytest.raises(llm.SpendCapError, match="Could not count"):
        llm._reserve(Broken(), model="claude-haiku-5-5", system="s", content=[{"type": "text", "text": "t"}], schema=Schema, max_tokens=100)


def test_no_reservation_outside_the_launch(monkeypatch):
    monkeypatch.delenv("RRN_SPEND_CAP_USD", raising=False)
    assert llm._reserve(None, model="claude-haiku-5-5", system="s", content=[], schema=Schema, max_tokens=100) is None


# ---------- Stage A: ranking prompts kept apart, provenance recorded ----------

@pytest.fixture
def v2(monkeypatch, tmp_path):
    exp = Path(__file__).resolve().parents[1] / "experiments"
    monkeypatch.setenv("RRN_OUT", str(tmp_path))
    monkeypatch.syspath_prepend(str(exp))
    sys.modules.pop("v2_offline", None)
    import v2_offline
    return v2_offline


def test_the_baseline_prompt_has_no_recency_rule_and_matches_the_cached_baseline(v2):
    from researchreadernerd.analyze import BASE_RANK_PROMPT, RANK_PROMPT, RECENCY_RULE
    assert RECENCY_RULE not in BASE_RANK_PROMPT and RANK_PROMPT == BASE_RANK_PROMPT + " " + RECENCY_RULE
    assert v2.PROMPT_FOR == {"plain": BASE_RANK_PROMPT, "v2": RANK_PROMPT}
    v2.check_baseline_provenance()  # the 342 cached baseline reranks were made with this exact prompt


def test_a_fresh_rerank_uses_its_design_prompt_and_records_it(v2, monkeypatch, tmp_path):
    seen = []

    def fake_ask(client, *, system, **kw):
        seen.append(system)
        return types.SimpleNamespace(value=types.SimpleNamespace(picks=[types.SimpleNamespace(candidate=0)]), input_tokens=1, output_tokens=1, model="claude-sonnet-5-5")

    monkeypatch.setattr(v2, "ask", fake_ask)
    c = {"candidates": ["A"]}
    v2.rank("t", None, tmp_path / "plain.json", c, "plain")
    v2.rank("t", None, tmp_path / "v2.json", c, "v2")
    assert seen == [v2.PROMPT_FOR["plain"], v2.PROMPT_FOR["v2"]]
    for f, d in (("plain.json", "plain"), ("v2.json", "v2")):
        assert json.loads((tmp_path / f).read_text())["prompt_sha256"] == hashlib.sha256(v2.PROMPT_FOR[d].encode()).hexdigest()


def test_a_cached_rerank_from_the_other_prompt_is_refused(v2, tmp_path):
    f = tmp_path / "x.json"
    f.write_text(json.dumps({"picks": ["A"], "prompt_sha256": hashlib.sha256(v2.PROMPT_FOR["v2"].encode()).hexdigest()}))
    with pytest.raises(RuntimeError, match="not the plain prompt"):
        v2.rank("t", None, f, {"candidates": ["A"]}, "plain")
    f.write_text(json.dumps({"picks": ["A"]}))  # no recorded prompt: accepted only as a sonnet_rerank.py baseline
    with pytest.raises(RuntimeError):
        v2.rank("t", None, f, {"candidates": ["A"]}, "v2")
    assert v2.rank("t", None, f, {"candidates": ["A"]}, "plain") == ["A"]


def test_the_count_includes_the_output_schema_exactly(tmp_path, monkeypatch):
    monkeypatch.setenv("RRN_SPEND_LEDGER", str(tmp_path / "spend.jsonl"))
    monkeypatch.setenv("RRN_SPEND_CAP_USD", "5")
    fake = FakeClaude(436)
    llm._reserve(fake, model="claude-haiku-5-5", system="s", content=[{"type": "text", "text": "t"}], schema=Schema, max_tokens=100)
    assert fake.seen[0]["output_config"] == {"format": llm.output_format(Schema)}, "counted with the same format the call sends"
    reserved = json.loads((tmp_path / "spend.jsonl").read_text().splitlines()[0])["usd"]
    worst = llm.worst_case_usd("claude-haiku-5-5", int(436 * 1.02) + 50, 100)
    assert worst <= reserved < worst + 1e-6, "reserved at the worst case, rounded up"


# ---------- OpenAlex balance in the documented response shape (Harsh, 25 Nov 2026) ----------
# help.openalex.org/api/authentication documents prepaid_balance_usd; a missing or malformed balance must fail closed.

DOCUMENTED = {"daily_budget_usd": 1, "daily_used_usd": 0.0, "daily_remaining_usd": 1.0, "prepaid_balance_usd": 3.18,
              "resets_at": "2026-11-26T00:00:00.000Z"}


class ShapeHTTP(FakeHTTP):
    def __init__(self, body):
        super().__init__()
        self.body = body

    def get(self, url, params=None, timeout=None, **kw):
        self.calls.append(url)
        body = {"rate_limit": self.body} if url.endswith("/rate-limit") else {"results": []}
        return types.SimpleNamespace(status_code=200, headers={}, json=lambda: body)


@pytest.mark.parametrize("balance, refused", [
    ({"prepaid_balance_usd": 3.18}, "prepaid credit"),             # the documented field alone
    ({"prepaid_remaining_usd": 3.18}, "prepaid credit"),           # the older field alone
    ({"prepaid_balance_usd": 0, "prepaid_remaining_usd": 2.0}, "prepaid credit"),  # the larger counts
    ({}, "no prepaid balance"),                                    # missing: fail closed
    ({"prepaid_balance_usd": None}, "malformed"),
    ({"prepaid_balance_usd": "3.18"}, "malformed"),
    ({"prepaid_balance_usd": -1}, "malformed"),
])
def test_the_request_guard_reads_the_documented_balance_and_fails_closed(guarded, monkeypatch, balance, refused):
    guarded()
    body = {k: v for k, v in DOCUMENTED.items() if not k.startswith("prepaid")} | balance
    h = ShapeHTTP(body)
    monkeypatch.setattr(scholar.httpx2, "get", h.get)
    with pytest.raises(scholar.ScholarError, match=refused):
        REAL_GET("/works", {"filter": "cites:W1"})
    assert not any(u.endswith("/works") for u in h.calls), "the priced request was never sent"


def test_a_zero_documented_balance_lets_requests_through(guarded, monkeypatch):
    guarded()
    h = ShapeHTTP(DOCUMENTED | {"prepaid_balance_usd": 0})
    monkeypatch.setattr(scholar.httpx2, "get", h.get)
    REAL_GET("/works", {"filter": "cites:W1"})
    assert h.calls[-1].endswith("/works")


@pytest.fixture
def launch_mod(monkeypatch, tmp_path):
    exp = Path(__file__).resolve().parents[1] / "experiments"
    monkeypatch.syspath_prepend(str(exp))
    sys.modules.pop("launch", None)
    import launch
    monkeypatch.setattr(launch, "LAUNCH", tmp_path)
    return launch


def test_startup_refuses_a_documented_balance_and_a_missing_or_malformed_one(launch_mod):
    assert "holds $3.18 of prepaid credit" in launch_mod.startup_refusal(DOCUMENTED)
    assert "no prepaid balance" in launch_mod.startup_refusal({k: v for k, v in DOCUMENTED.items() if k != "prepaid_balance_usd"})
    assert "malformed" in launch_mod.startup_refusal(DOCUMENTED | {"prepaid_balance_usd": "x"})
    assert launch_mod.startup_refusal(DOCUMENTED | {"prepaid_balance_usd": 0}) is None


def test_the_meter_records_a_decrease_of_the_documented_balance(launch_mod, monkeypatch):
    reads = iter([DOCUMENTED, DOCUMENTED | {"prepaid_balance_usd": 3.17}])
    monkeypatch.setattr(launch_mod, "openalex_state", lambda: next(reads))
    m = launch_mod.Meter("B")
    m.poll()
    start = m.s["prepaid_spent"]
    m.poll()
    assert m.s["prepaid_spent"] == pytest.approx(0.01), "3.18 -> 3.17 recorded as a prepaid charge"
    assert launch_mod.prepaid_charged(m, start), "the post-step and final checks see it"


def test_the_meter_fails_closed_on_a_missing_or_malformed_balance(launch_mod, monkeypatch):
    for bad in ({k: v for k, v in DOCUMENTED.items() if k != "prepaid_balance_usd"}, DOCUMENTED | {"prepaid_balance_usd": None}):
        monkeypatch.setattr(launch_mod, "openalex_state", lambda bad=bad: bad)
        with pytest.raises(RuntimeError, match="prepaid"):
            launch_mod.Meter("B").poll()


# ---------- spend watcher (27 Nov 2026: stage B stop at 13:03) ----------

class FakeProc:
    def __init__(self, done=False):
        self.done, self.terminated = done, False

    def poll(self):
        return 0 if self.done else None

    def terminate(self):
        self.terminated = True; self.done = True


class CountingMeter:
    def __init__(self):
        self.polls, self.s = 0, {"prepaid_spent": 0.0, "daily_left": 1.0}

    def poll(self):
        self.polls += 1


def test_a_watcher_whose_step_ended_does_not_watch_the_next_step(launch_mod, monkeypatch):
    """The watcher of a short step wakes after its step has ended and the next step has started: it must return without
    reading spend, because it is bound to its own step's process (it used to read the loop's rebound 'proc')."""
    short, nxt = FakeProc(), FakeProc()
    meter = CountingMeter()

    def sleep(s):  # the short step ends while its watcher sleeps; the next step is then running
        short.done = True
    monkeypatch.setattr(launch_mod.time, "sleep", sleep)
    monkeypatch.setattr(launch_mod, "over_cap", lambda *a: None)
    launch_mod.watch_step(short, meter, "B", [], 0.0, [], [])
    assert meter.polls == 0 and not nxt.terminated


def test_a_watcher_stops_its_own_step_when_spend_cannot_be_read(launch_mod, monkeypatch):
    proc, reason = FakeProc(), []

    class Broken(CountingMeter):
        def poll(self):
            raise RuntimeError("OpenAlex spend could not be read from the rate-limit endpoint")
    monkeypatch.setattr(launch_mod.time, "sleep", lambda s: None)
    launch_mod.watch_step(proc, Broken(), "B", [], 0.0, reason, [])
    assert proc.terminated and reason and "could not be read" in reason[0]


def test_the_meter_file_is_never_seen_truncated(launch_mod, monkeypatch):
    """Meter writes replace the file whole, so a reader (another stage part's Meter) never parses an empty file."""
    import threading
    monkeypatch.setattr(launch_mod, "openalex_state", lambda: DOCUMENTED | {"prepaid_balance_usd": 0.0})
    m = launch_mod.Meter("B")
    m.poll()
    errors, stop = [], threading.Event()

    def read():
        while not stop.is_set():
            try:
                launch_mod.Meter("B")
            except json.JSONDecodeError as e:
                errors.append(e)
    t = threading.Thread(target=read); t.start()
    for _ in range(300):
        m.poll()
    stop.set(); t.join()
    assert errors == []
