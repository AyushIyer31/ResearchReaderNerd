"""Per-step token and endpoint cost worksheet for the approved launch (14 Nov 2026). No provider calls.

Takes the call counts and token means of the approved plan (run_plan.py, $79 total ceiling) and itemizes each step by
endpoint: the Anthropic Messages API per model id, and OpenAlex by endpoint class at the card OpenAlex publishes on
its rate-limit endpoint (singleton $0, list $0.0001, search $0.001 per call). Our title lookups are filters on
title.search / title_and_abstract.search; the card does not say whether these bill as list or as search, so both are
priced. The one direct measurement we have (8 Nov 2026, 30 user-1 runs, quota read before and after) says list.

  python launch_worksheet.py -> out/report_2026-11-08/launch_cost_worksheet_2026-11-14.md and .csv
"""
from __future__ import annotations

import csv
import glob
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "src"))
from run_plan import CAP_FACTOR, OA_PER_CALL, PRICE, RETRY, plan_lines  # noqa: E402

R = HERE / "out" / "report_2026-11-08"
CARD = {"singleton": 0.0, "list": 0.0001, "search": 0.001}  # OpenAlex rate-limit endpoint, endpoint_costs_usd, read 13 Nov 2026
APPROVED = {"A": 21, "B": 43, "C": 15, "All": 79}
# 8 Nov 2026, out/logs/tonight_2026-11-08.log: 'benchmark_eval.py run ... refs+model' on the 30 old-set papers (no OpenAlex
# cache in that harness), quota.py read 0.3331 before and 0.3917 after, with no prepaid credit on the key that day.
OBSERVED = {"runs": 30, "usd": 0.3917 - 0.3331, "log": "out/logs/tonight_2026-11-08.log lines 45 and 138"}


def cache_split() -> dict[str, float]:
    """Share of each endpoint class among the cached OpenAlex responses of the as-run leave-one-out harness."""
    c = Counter()
    for f in [f for d in ("out_2026", "out_2026b", "out_old_v2") for f in glob.glob(str(HERE / d / "oa_cache" / "*.json"))]:
        d = json.loads(Path(f).read_text())
        q = ((d.get("meta") or {}).get("x_query") or {}).get("oql", "")
        c["singleton" if not d.get("meta") else "search_filter" if "in the title" in q else "list"] += 1
    n = sum(c.values())
    return {k: v / n for k, v in c.items()} | {"_n": n}


def stage_a_expected() -> dict:
    """Dating lookups stage A will actually make: undated literature candidates per saved run, at most 60 per run,
    and at most one lookup per distinct source title per set (title cache), approximated by distinct names."""
    import importlib
    out = {}
    for d, papers in [("out_2026", "papers_2026.json"), ("out_2026b", "papers_2026b.json"), ("out_old_v2", "papers.json")]:
        os.environ["RRN_OUT"], os.environ["RRN_PAPERS"] = d, papers
        import sonnet_rerank, v2_offline  # noqa: E401
        importlib.reload(sonnet_rerank); importlib.reload(v2_offline)
        P = {p["id"]: p for p in json.loads((HERE / papers).read_text())}
        runs = {j[5]: j[4] for j in sonnet_rerank.jobs_for(P)}
        per_run, names = 0, set()
        for c in runs.values():
            und = [n for n in c["candidates"] if not (c.get("first_seen") or {}).get(n) and n not in v2_offline.catalog_only(c)]
            per_run += min(60, len(und)); names |= set(und)
        out[d] = {"runs": len(runs), "capped_per_run_sum": per_run, "distinct_names": len(names)}
    return out


def main():
    L, m = plan_lines()
    split = cache_split()
    base_sf, base_list, base_single = 30 * split["search_filter"], 30 * split["list"], 30 * split["singleton"]
    A = stage_a_expected()
    a_expected = sum(v["distinct_names"] for v in A.values())

    def oa_classes(step: str, oa_calls: int) -> dict[str, float]:
        """Split a line's OpenAlex calls into endpoint classes, from what the step calls."""
        if not oa_calls:
            return {}
        if step.startswith("A2"):
            return {"search_filter": oa_calls}  # dating: one title.search filter per source title
        if step.startswith("B5"):
            runs = oa_calls / 130
            return {"singleton": runs * base_single, "list": runs * base_list, "search_filter": runs * (base_sf + 40 + 60)}
        if step.startswith("C1"):
            return {"list": oa_calls}
        if step.startswith("C8"):
            runs = (oa_calls - 150) / 130
            return {"singleton": runs * base_single + 30, "list": runs * base_list + 120, "search_filter": runs * (base_sf + 40 + 60)}
        return {"list": oa_calls}

    rows = []
    for l in L:
        cls = oa_classes(l["step"], l["oa_calls"])
        si, li = round(cls.get("singleton", 0)), round(cls.get("list", 0))
        sf = l["oa_calls"] - si - li if cls else 0  # the split adds up to the plan's call count exactly
        p_in, p_out = PRICE.get(l["model"], (0, 0))
        rows.append({
            "stage": l["stage"], "step": l["step"], "note": l["note"], "endpoint": f"Anthropic Messages API, {l['model']}" if l["calls"] else ("OpenAlex /works" if l["oa_calls"] else "-"),
            "model": l["model"], "calls": l["calls"], "tokens_in_per_call": round(l["tin"]), "tokens_out_per_call": round(l["tout"]),
            "tokens_in_total": round(l["calls"] * l["tin"]), "tokens_out_total": round(l["calls"] * l["tout"]),
            "price_in_per_M": p_in, "price_out_per_M": p_out, "model_usd": round(l["model_cost"], 4),
            "oa_calls": l["oa_calls"], "oa_singleton": round(si), "oa_list": round(li), "oa_search_filter": round(sf),
            "oa_usd_plan_rate": round(l["oa_cost"], 4),
            "oa_usd_card_filters_as_list": round((li + sf) * CARD["list"], 4),
            "oa_usd_card_filters_as_search": round(li * CARD["list"] + sf * CARD["search"], 4)})
    with open(R / "launch_cost_worksheet_2026-11-14.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    def money(x):
        return f"${x:,.2f}" if x else "-"

    out = ["# Launch cost worksheet, per step (14 November 2026)", "",
           "Approved: total ceiling $79 including retries (Harsh, 13 Nov). Call counts and token means are the approved plan's (run_plan_2026-11-12.md, generated by run_plan.py); this sheet itemizes each step by endpoint and prices OpenAlex three ways. The same rows are in launch_cost_worksheet_2026-11-14.csv.", "",
           "## Unit prices", "",
           "| Endpoint | Unit | Price | Source |", "| --- | --- | --- | --- |",
           f"| Anthropic Messages API, claude-haiku-5-5 | per M tokens in / out (prompt under 100K tokens) | ${PRICE['claude-haiku-5-5'][0]:.2f} / ${PRICE['claude-haiku-5-5'][1]:.2f} | src/researchreadernerd/pricing.py |",
           f"| Anthropic Messages API, claude-sonnet-5-5 | per M tokens in / out | ${PRICE['claude-sonnet-5-5'][0]:.2f} / ${PRICE['claude-sonnet-5-5'][1]:.2f} | pricing.py |",
           "| OpenAlex singleton (a work by id or DOI) | per call | $0 | OpenAlex rate-limit endpoint, endpoint_costs_usd, read 13 Nov |",
           "| OpenAlex list (a filter query) | per call | $0.0001 | same |",
           "| OpenAlex search (the search parameter) | per call | $0.001 | same |",
           f"| OpenAlex, the plan's flat rate | per call | ${OA_PER_CALL:.5f} | run_plan.py; used for the approved caps |",
           "| arxiv.org PDF and HTML | per fetch | $0 | public |", "",
           "Our OpenAlex lookups by title are filter queries on title.search or title_and_abstract.search; the card does not say whether OpenAlex bills these as list or as search. "
           f"Direct measurement: on 8 Nov the 30 old-set user-1 runs (refs+model, no OpenAlex cache) cost ${OBSERVED['usd']:.4f} of OpenAlex by the quota read before and after ({OBSERVED['log']}), "
           f"${OBSERVED['usd'] / OBSERVED['runs']:.4f} per run. A run makes about {base_sf:.0f} title-filter calls, which billed as search would be about ${base_sf * CARD['search']:.3f} per run, roughly {base_sf * CARD['search'] / (OBSERVED['usd'] / OBSERVED['runs']):.0f}x the measurement; billed as list the run's ~30 calls come to about ${30 * CARD['list']:.4f}, close to it. So the expected column is 'filters as list'; the plan's flat rate is twice that; 'filters as search' is the worst case.", "",
           "Correction to the plan's wording: run_plan_2026-11-12.md calls the $1.60 for the 272 runs of 10-11 Nov 'observed'. It was the estimate written before those runs (status log, 10 Nov). The 8 Nov measurement above is the observed figure, and it is below the plan's rate, so the approved caps are not affected.", "",
           f"Endpoint split of a pipeline run's ~30 base OpenAlex calls, from the {split['_n']:,} cached responses of the as-run leave-one-out harness: {split['search_filter']:.0%} title filters, {split['list']:.0%} list (id batches, cites), {split['singleton']:.0%} singleton. The bibliography route adds up to 40 title filters per run with a reference list; dating adds up to 60 title filters per run.", "",
           "## Per step", "",
           "Rows B1, B2, C4 and C5 are whole pipeline runs (plan, extraction, labelling, proposals; the dating, ranking and judge calls are their own rows): their Calls column counts runs and their tokens are all of a run's Haiku tokens, from the saved runs. Every other row is one call per unit.", "",
           "| Stage | Step | Endpoint | Calls (runs, on pipeline rows) | Tokens in / out per call (per run, on pipeline rows) | Tokens in / out total | Model cost | OpenAlex calls: singleton / list / title filter | OpenAlex, plan rate | OpenAlex, filters as list (expected) | OpenAlex, filters as search (worst) |",
           "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    tot = {}
    for r in rows:
        out.append(f"| {r['stage']} | {r['step']}{' (' + r['note'] + ')' if r['note'] else ''} | {r['endpoint']} | {r['calls'] or '-'} | "
                   + (f"{r['tokens_in_per_call']:,} / {r['tokens_out_per_call']:,}" if r["calls"] else "-") + " | "
                   + (f"{r['tokens_in_total']:,} / {r['tokens_out_total']:,}" if r["calls"] else "-") + f" | {money(r['model_usd'])} | "
                   + (f"{r['oa_singleton']:,} / {r['oa_list']:,} / {r['oa_search_filter']:,}" if r["oa_calls"] else "-")
                   + f" | {money(r['oa_usd_plan_rate'])} | {money(r['oa_usd_card_filters_as_list'])} | {money(r['oa_usd_card_filters_as_search'])} |")
        for k in (r["stage"], "All"):
            t = tot.setdefault(k, Counter())
            for f in ("model_usd", "oa_usd_plan_rate", "oa_usd_card_filters_as_list", "oa_usd_card_filters_as_search", "tokens_in_total", "tokens_out_total", "calls", "oa_calls"):
                t[f] += r[f]
    out += ["", "## Totals against the approved caps", "",
            "Each total adds the model cost to the OpenAlex column named, then the 15% retry allowance. The caps are the approved ones and do not change.", "",
            "| Stage | Model calls (pipeline rows count runs) | Tokens in / out | Model cost | Plan rate, with retries | Filters as list, with retries (expected) | Filters as search, with retries (worst) | Approved cap |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for k in ("A", "B", "C", "All"):
        t = tot[k]
        f = lambda oa: (t["model_usd"] + t[oa]) * (1 + RETRY)  # noqa: E731
        worst = f("oa_usd_card_filters_as_search")
        out.append(f"| {k} | {t['calls']:,} | {t['tokens_in_total'] / 1e6:,.1f} M / {t['tokens_out_total'] / 1e6:,.2f} M | {money(t['model_usd'])} | {money(f('oa_usd_plan_rate'))} | "
                   f"{money(f('oa_usd_card_filters_as_list'))} | {money(worst)}{' (over the cap)' if worst > APPROVED[k] else ''} | ${APPROVED[k]} |")
    a_sf = sum(r["oa_search_filter"] for r in rows if r["step"].startswith("A2"))
    a_worst_counted = (tot["A"]["model_usd"] + a_expected * CARD["search"]) * (1 + RETRY)
    per_run = ", ".join(f"{v['capped_per_run_sum'] / v['runs']:.0f}" for v in A.values())
    out += ["", f"Stage A's row uses the plan's 20,520 lookups. With the {a_expected:,} lookups counted on the saved pools (next section), stage A at search price is ${a_worst_counted:.2f} with retries, under its cap."]
    # Where the runner does more than the plan counted, priced at the plan's rates (per-run means, plan OpenAlex rate, 15% retries)
    from researchreadernerd.pricing import estimate_cost
    H, S5 = PRICE["claude-haiku-5-5"], PRICE["claude-sonnet-5-5"]
    cost = lambda price, tin, tout: (tin * price[0] + tout * price[1]) / 1e6  # noqa: E731
    per_run_extra = lambda pipe: (cost(H, *pipe) + cost(H, 3_000, 800) + cost(S5, m["rank"][0] + 1_000, m["rank"][1]) + cost(H, 2_500, 400)  # noqa: E731
                                  + 130 * OA_PER_CALL)
    import launch  # noqa: PLC0415  the runner's own count of leave-one-out jobs, from the current truth lists
    loo_jobs = sum(launch._expected(f"loo:{d}:{pj}") for d, pj in (("out_2026", "papers_2026.json"), ("out_2026b", "papers_2026b.json"), ("out_old_v2", "papers.json")))
    d_c4 = (60 - 50) * per_run_extra(m["blind"]) * (1 + RETRY)
    d_b2 = (loo_jobs - 276) * per_run_extra(m["loo"]) * (1 + RETRY)
    reads = [json.loads(Path(f).read_text()) for f in glob.glob(str(HERE / "out_2026*" / "extract" / "*.json")) if "out_2026ab" not in f]
    reads = [r for r in reads if "input_tokens" in r]
    read_cost = sum(estimate_cost("claude-haiku-5-5", r["input_tokens"], r["output_tokens"]) for r in reads) / len(reads)
    big = sum(1 for r in reads if r["input_tokens"] > 100_000)
    plan_c2 = next(r["model_usd"] for r in rows if r["step"].startswith("C2"))
    d_c2 = (30 * read_cost - plan_c2) * (1 + RETRY)
    d_tc = 0.0  # held-out truth check: Gemini in its chat window, as for the development sets; no API call
    # smaller differences: C1 and C2 both judge the user-1 and Haiku-alone pools (C2 re-judges on the checked lists); age_split's
    # Haiku dating of undated truth entries (one call per table); topic+model runs are longer than the refs+model mean the plan used
    topic = [json.loads(Path(f).read_text()) for f in glob.glob(str(HERE / "out_2026*" / "blind" / "*.topic+model.json")) if "out_2026ab" not in f]
    topic = [t for t in topic if "input_tokens" in t]
    t_in = sum(t["input_tokens"] for t in topic) / len(topic); t_out = sum(t["output_tokens"] for t in topic) / len(topic)
    d_small = ((60 + 30) * cost(H, 2_500, 400) + 3 * cost(H, 4_000, 2_000)
               + (90 + 30) * max(0.0, cost(H, t_in - m["blind"][0], t_out - m["blind"][1]))) * (1 + RETRY)
    runner_total = (tot["All"]["model_usd"] + tot["All"]["oa_usd_plan_rate"]) * (1 + RETRY) + d_c4 + d_b2 + d_c2 + d_tc + d_small
    oa_needed = tot["All"]["oa_usd_plan_rate"] * (1 + RETRY)
    c_plan = (tot["C"]["model_usd"] + tot["C"]["oa_usd_plan_rate"]) * (1 + RETRY); b_plan = (tot["B"]["model_usd"] + tot["B"]["oa_usd_plan_rate"]) * (1 + RETRY)
    out += ["", "## Where the runner's counts differ from the plan's", "",
            "The rows above are the approved plan's counts. Where the runner differs, each difference is priced below at the plan's rates with retries. The caps do not change.", "",
            f"- **C4 user-1 runs.** The plan counts 50 (2 conditions x 25 scored papers). The runner runs both conditions on all 30 sampled papers, 60 runs, since which papers have a public benchmark is known only after the truth check. Extra ${d_c4:.2f}.",
            f"- **C2 PDF reading.** The plan uses the old set's mean reading. The {len(reads)} post-cutoff readings average ${read_cost:.4f} each at the tool's prices ({big} of them over 100K input tokens, which bill at Haiku's long-prompt price), so 30 readings come to ${30 * read_cost:.2f}, not ${plan_c2:.2f}. Extra ${d_c2:.2f}.",
            "- **Held-out truth check.** Protocol section 3's check, done as for the development sets: Gemini in its chat window reads each PDF against the list and a reviewer applies only verdicts with a quoted location. No API call, so no cost line.",
            f"- **Smaller differences.** C1 and C2 both judge the user-1 and Haiku-alone pools (C2 again on the checked lists); age_split.py's Haiku dating of undated truth entries; and topic+model runs average {t_in:,.0f} input tokens, not the plan's refs+model mean of {m['blind'][0]:,.0f}. Together ${d_small:.2f}.",
            "- **arXiv bibliography fetches (B7).** The plan's '90 fetches' is per paper; the tool fetches a paper's arXiv HTML at most once per process, so a stage B set makes one fetch per paper for the user-1 runs and one per leave-one-out pass. Free; a failed fetch is written to the OpenAlex ledger.",
            f"- **Effect.** Stage B ${b_plan:.2f} -> ${b_plan + d_b2:.2f} against its $43 cap ({loo_jobs} leave-one-out runs from the current truth lists, {276 - loo_jobs} fewer than the plan's 276: EmulatorBench was settled as own on 14 Nov); stage C ${c_plan:.2f} -> ${c_plan + d_c4 + d_c2 + d_tc:.2f} against its $15 cap; smaller differences ${d_small:.2f}; **all stages ${runner_total:.2f} with retries against $79**.",
            f"- **OpenAlex money.** The OpenAlex column at the plan's rate comes to ${oa_needed:.2f} with retries over all stages. OpenAlex meters a $1 daily budget plus prepaid credit; the runner waits when both are spent, so the run needs prepaid credit of about that amount to finish without day-long pauses."]
    out += ["", "## What the saved data say about the largest OpenAlex lines", "",
            f"- **Stage A dating lookups.** The plan budgets {a_sf:,} (60 for each of the 342 runs). Counted on the saved pools, the live rule (literature candidates only, catalog keyword hits are not dated, at most 60 per run) gives "
            + ", ".join(f"{d}: {v['capped_per_run_sum']:,} lookups over {v['runs']} runs, {v['distinct_names']:,} distinct names" for d, v in A.items())
            + f". The replay caches lookups per source title within a set, so the expected count is about {a_expected:,}, about ${a_expected * CARD['list']:.2f} at list price and ${a_expected * CARD['search']:.2f} at search price.",
            "- **Stage B per-run OpenAlex.** The plan budgets 130 calls for all 456 runs. The 90 topic+model runs read no reference list, so the bibliography route does not run in them. Dating looks up only undated literature candidates: on the saved pools that is " + per_run + " per run in the three sets, not 60. Leave-one-out runs reuse the as-run OpenAlex response cache for identical queries. The budget is therefore an upper bound.",
            "- **Worst case.** If title filters bill as search, stages B and C exceed their caps (totals above). The runner reads OpenAlex spend from the rate-limit endpoint before, during and after every step, and stops a stage when its measured spend reaches the cap or when the spend projected from the share of runs done exceeds the cap once 10% of the runs are done. The stage is then reported as partial; nothing goes over its cap.", "",
            "## Retries and caps", "",
            f"Retry allowance {int(RETRY * 100)}% on every line, for calls repeated after rate limits, timeouts or a re-judged chunk. Hard cap per stage = {CAP_FACTOR}x the plan-rate estimate with retries, rounded up. Approved caps: A $21, B $43, C $15, total $79. The runner also stops at $79 summed over all stages.", ""]
    (R / "launch_cost_worksheet_2026-11-14.md").write_text("\n".join(out))
    # the figures the cover email quotes, so that no number in it is typed by hand
    (R / "launch_cost_summary_2026-11-14.json").write_text(json.dumps({
        "plan_total_with_retries": round((tot["All"]["model_usd"] + tot["All"]["oa_usd_plan_rate"]) * (1 + RETRY), 2),
        "runner_total_with_retries": round(runner_total, 2), "stage_B_with_retries": round(b_plan + d_b2, 2),
        "stage_C_with_retries": round(c_plan + d_c4 + d_c2 + d_tc, 2), "openalex_plan_rate_with_retries": round(oa_needed, 2),
        "worst_case_search_price_total": round((tot["All"]["model_usd"] + tot["All"]["oa_usd_card_filters_as_search"]) * (1 + RETRY), 2)}, indent=1))
    print("\n".join(out[-30:]))


if __name__ == "__main__":
    main()
