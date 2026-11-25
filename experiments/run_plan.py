"""Itemized run plan with per-step calls, tokens and cost, a retry allowance and a hard cap (12 Nov 2026).

Token counts per call come from the saved runs (means over the recorded input/output tokens per run type and the
Sonnet manifest); prices from the tool's price table (src/researchreadernerd/pricing.py). OpenAlex cost per call is
derived from the observed $1.60 for the 272 runs of 10-11 Nov at about 30 calls per run.

  python run_plan.py -> out/report_2026-11-08/run_plan_2026-11-12.md
"""
import glob, json, statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = HERE / "out" / "report_2026-11-08"
PRICE = {"claude-haiku-5-5": (0.10, 0.50), "claude-sonnet-5-5": (2.0, 10.0)}  # USD per M tokens, in / out (Haiku small card, inputs under 100K)
OA_PER_CALL = 1.60 / (272 * 30)  # observed spend / (runs x calls per run)
RETRY = 0.15  # allowance on every line for retried calls (rate limits, timeouts, re-judged chunks)
CAP_FACTOR = 1.5  # hard cap over the estimate including retries, rounded up to the dollar


def mean_tokens(pattern):
    tin, tout = [], []
    for f in glob.glob(str(HERE / pattern)):
        d = json.loads(Path(f).read_text())
        if "input_tokens" in d and "error" not in d:
            tin.append(d["input_tokens"]); tout.append(d["output_tokens"])
    return (st.mean(tin), st.mean(tout)) if tin else (0, 0)


def plan_lines():
    """The per-step lines of the approved plan and the token means behind them (used by launch_worksheet.py too)."""
    blind_in, blind_out = mean_tokens("out_2026*/blind/*.refs+model.json")
    loo_in, loo_out = mean_tokens("out_2026*/loo/*.json")
    man = json.loads((R / "sonnet_rerank_manifest.json").read_text())
    rank_in, rank_out = man["input_tokens"] / man["runs"], man["output_tokens"] / man["runs"]
    # the Haiku pipeline without its ranking call (Haiku's rank call has about the same tokens as Sonnet's on the same input)
    pipe_blind = (blind_in - rank_in, blind_out - rank_out); pipe_loo = (loo_in - rank_in, loo_out - rank_out)
    read_in, read_out = 55_869, 10_451  # reading a PDF, from the old-set summary (mean tokens in / out per paper)

    def line(stage, step, model, calls, tin, tout, oa_calls=0, note=""):
        price = PRICE.get(model, (0, 0))
        model_cost = calls * (tin * price[0] + tout * price[1]) / 1e6
        oa_cost = oa_calls * OA_PER_CALL
        return {"stage": stage, "step": step, "model": model or "-", "calls": calls, "tin": tin, "tout": tout, "oa_calls": oa_calls,
                "model_cost": model_cost, "oa_cost": oa_cost, "note": note}

    L = []
    # ---- Stage A: cached replay (342 saved runs: 41 user-1 refs+model + 159 leave-one-out post-cutoff; 25 + 117 old set)
    A = 342
    L += [line("A", "A1 date undated candidates (one call per run)", "claude-haiku-5-5", A, 3_000, 800),
          line("A", "A2 confirm source titles in OpenAlex (<= 60 lookups per run, cached per title)", "", 0, 0, 0, A * 60),
          line("A", "A3 rank with Sonnet, year-aware prompt (one call per run)", "claude-sonnet-5-5", A, rank_in + 1_000, rank_out, 0, "+1K tokens for the year suffix per candidate"),
          line("A", "A4 score against the saved judge matches", "", 0, 0, 0, 0, "no provider calls")]
    # ---- Stage B: live version-2 development run on the existing sets
    U1, LOO = 180, 276
    L += [line("B", "B1 user-1 runs: plan, read related abstracts, label bibliographies, propose (Haiku)", "claude-haiku-5-5", U1, *pipe_blind),
          line("B", "B2 leave-one-out runs: same pipeline on the scrubbed paper (Haiku)", "claude-haiku-5-5", LOO, *pipe_loo),
          line("B", "B3 dating call per run (Haiku)", "claude-haiku-5-5", U1 + LOO, 3_000, 800),
          line("B", "B4 ranking call per run (Sonnet, year-aware)", "claude-sonnet-5-5", U1 + LOO, rank_in + 1_000, rank_out),
          line("B", "B5 OpenAlex per run: ~30 as before + <= 40 bibliography lookups + <= 60 dating lookups", "", 0, 0, 0, (U1 + LOO) * 130),
          line("B", "B6 judge re-matching where the pool changed (Haiku, one call per run)", "claude-haiku-5-5", U1 + LOO, 2_500, 400),
          line("B", "B7 arXiv HTML bibliography fetch (one per paper)", "", 0, 0, 0, 0, "90 fetches from arxiv.org, free"),
          line("B", "B8 Haiku-alone baselines, probes, Haiku-alone leave-one-out", "", 0, 0, 0, 0, "cached from the current runs; re-used")]
    # ---- Stage C: held-out sample, 30 papers, about 25 scored, about 90 public benchmarks
    P, S, NB = 30, 25, 90
    L += [line("C", "C1 draw the sample (OpenAlex sample query + abstracts)", "", 0, 0, 0, 10),
          line("C", "C2 read 30 PDFs (Haiku)", "claude-haiku-5-5", P, read_in, read_out),
          line("C", "C3 truth-list grouping by the judge (Haiku, one call per paper)", "claude-haiku-5-5", P, 3_000, 1_000),
          line("C", "C4 user-1 runs, two conditions (Haiku pipeline)", "claude-haiku-5-5", 2 * S, *pipe_blind),
          line("C", "C5 leave-one-out runs (Haiku pipeline)", "claude-haiku-5-5", NB, *pipe_loo),
          line("C", "C6 dating calls (Haiku)", "claude-haiku-5-5", 2 * S + NB, 3_000, 800),
          line("C", "C7 ranking calls (Sonnet)", "claude-sonnet-5-5", 2 * S + NB, rank_in + 1_000, rank_out),
          line("C", "C8 OpenAlex for the runs (130 per run) and reference resolution", "", 0, 0, 0, (2 * S + NB) * 130 + P * 5),
          line("C", "C9 Haiku-alone baselines: user 1 (30) and leave-one-out (90)", "claude-haiku-5-5", P + NB, 2_500, 600),
          line("C", "C10 memory probes, declining and forced (Haiku)", "claude-haiku-5-5", 2 * P, 1_000, 500),
          line("C", "C11 judge matching: user 1 x2, leave-one-out, Haiku-alone x2, probes x2 (Haiku)", "claude-haiku-5-5", 2 * S + NB + P + NB + 2 * P, 2_500, 400)]
    return L, {"blind": (blind_in, blind_out), "loo": (loo_in, loo_out), "rank": (rank_in, rank_out), "read": (read_in, read_out)}


def main():
    L, m = plan_lines()
    (blind_in, blind_out), (loo_in, loo_out), (rank_in, rank_out), (read_in, read_out) = m["blind"], m["loo"], m["rank"], m["read"]
    out = ["# Itemized run plan (12 November 2026)", "",
           f"Token counts per call are means from the saved runs (user-1 run {blind_in:,.0f} in / {blind_out:,.0f} out; leave-one-out run {loo_in:,.0f} / {loo_out:,.0f}; Sonnet ranking {rank_in:,.0f} / {rank_out:,.0f} from the 342 cached rankings; PDF reading {read_in:,} / {read_out:,}). "
           f"Prices: Haiku 5.5 ${PRICE['claude-haiku-5-5'][0]:.2f} / ${PRICE['claude-haiku-5-5'][1]:.2f} per M tokens in / out (inputs under 100K), Sonnet 5.5 ${PRICE['claude-sonnet-5-5'][0]:.2f} / ${PRICE['claude-sonnet-5-5'][1]:.2f}. "
           f"OpenAlex ${OA_PER_CALL:.5f} per call, from the observed $1.60 for 272 runs at about 30 calls each. Retry allowance {int(RETRY * 100)}% on every line. Hard cap = {CAP_FACTOR}x the estimate including retries, rounded up; enforced by the runner, which reads the OpenAlex balance (quota.py) and the summed per-run model spend before every step and halts at the cap; partial results are kept and reported as partial. "
           "Model ids exactly: claude-haiku-5-5 for every call except ranking; claude-sonnet-5-5 for the ranking call. No other provider.", "",
           "| Stage | Step | Model | Calls | Tokens in / out per call | OpenAlex calls | Model cost | OpenAlex cost |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    totals = {}
    for l in L:
        out.append(f"| {l['stage']} | {l['step']}{' (' + l['note'] + ')' if l['note'] else ''} | {l['model']} | {l['calls'] or '-'} | {f'{l[chr(116)+chr(105)+chr(110)]:,.0f} / {l[chr(116)+chr(111)+chr(117)+chr(116)]:,.0f}' if l['calls'] else '-'} | {l['oa_calls'] or '-'} | {f'${l[chr(109)+chr(111)+chr(100)+chr(101)+chr(108)+chr(95)+chr(99)+chr(111)+chr(115)+chr(116)]:.2f}' if l['model_cost'] else '-'} | {f'${l[chr(111)+chr(97)+chr(95)+chr(99)+chr(111)+chr(115)+chr(116)]:.2f}' if l['oa_cost'] else '-'} |")
        t = totals.setdefault(l["stage"], {"model": 0.0, "oa": 0.0, "calls": 0, "oa_calls": 0}); t["model"] += l["model_cost"]; t["oa"] += l["oa_cost"]; t["calls"] += l["calls"]; t["oa_calls"] += l["oa_calls"]
    out += ["", "## Totals, retry allowance and hard caps", "", "| Stage | What | Model calls | OpenAlex calls | Model cost | OpenAlex cost | Estimate | With 15% retries | Hard cap |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    import math
    grand = {"est": 0.0, "retry": 0.0, "cap": 0}
    names = {"A": "Cached replay: dating + Sonnet year-aware ranking on the 342 saved pools; no new pool", "B": "Recall of the version-2 tool (Sonnet ranking, dating, arXiv bibliography) on the 90 existing papers, user 1 and leave-one-out, by benchmark age, against the cached Haiku-alone baselines; the only stage that can measure the bibliography route; development data for the version decision, not a result", "C": "Held-out sample: 30 new papers under the frozen protocol, full test set"}
    for s, t in totals.items():
        est = t["model"] + t["oa"]; ret = est * (1 + RETRY); cap = math.ceil(ret * CAP_FACTOR)
        grand["est"] += est; grand["retry"] += ret; grand["cap"] += cap
        out.append(f"| {s} | {names[s]} | {t['calls']:,} | {t['oa_calls']:,} | ${t['model']:.2f} | ${t['oa']:.2f} | ${est:.2f} | ${ret:.2f} | ${cap} |")
    out.append(f"| All | A + B + C | {sum(t['calls'] for t in totals.values()):,} | {sum(t['oa_calls'] for t in totals.values()):,} | ${sum(t['model'] for t in totals.values()):.2f} | ${sum(t['oa'] for t in totals.values()):.2f} | ${grand['est']:.2f} | ${grand['retry']:.2f} | ${grand['cap']} |")
    out += ["", "Scope notes. A makes no new pool, so it tests dating and Sonnet only. B is development data under the protocol; the second proposal call stays off. C starts only after the manifest is frozen for the chosen version, the window and seed are stated and the section-6 predictions are recorded. "
            "The decision rule between A and B is the protocol's section 8. Each stage is approved separately; a stage that reaches its cap stops and is reported as partial. Current OpenAlex balance $3.18. OpenAlex meters a $1 daily budget plus prepaid credit; without prepaid credit stage B alone would take about twelve quota days, so the plan assumes a prepaid key covering the OpenAlex column.", ""]
    (R / "run_plan_2026-11-12.md").write_text("\n".join(out)); print("\n".join(out[-8:]))


if __name__ == "__main__":
    main()
