"""Cost and time worksheet for the version-2 launch, stages A and B (hybrid sources, 24 Nov 2026).

Claude rows are the approved plan's (launch_cost_worksheet_2026-11-14.csv, from run_plan.py), unchanged by the source
switch. Scholarly lookups: every title or topic search the plan counted on OpenAlex is now a Semantic Scholar call (free,
1 a second with a key), plus one reference-list call per paper lookup; each paper found is read from OpenAlex by id
(free). OpenAlex keeps its list queries ($0.0001) and the title-search fallback ($0.001) for papers whose ids do not
resolve to a matching record. FALLBACK_RATE is the share of Semantic Scholar title lookups that need it: measured in
the live check with the key (RRN_FALLBACK_RATE); until then the bound below is used and the sheet says so.

  python launch_worksheet_v2.py -> out/report_2026-11-08/launch_cost_worksheet_2026-11-24.{md,csv} and launch_cost_summary_2026-11-24.json
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = HERE / "out" / "report_2026-11-08"
PLAN_CSV = R / "launch_cost_worksheet_2026-11-14.csv"
RETRY = 0.15
CAPS = {"A": 21.0, "B": 43.0}
LIST, SEARCH = 0.0001, 0.001
MEASURED = os.environ.get("RRN_FALLBACK_RATE")
FALLBACK_RATE = float(MEASURED) if MEASURED else 0.25  # bound until measured: a quarter of title lookups fall back
B_PAPER_LOOKUPS = 456  # one find_paper per run in stage B (cached per paper inside the leave-one-out harness): one reference-list call each
B_EXPECTED_LOOKUPS = (5000, 10000)  # stage B title lookups after the leave-one-out cache (estimate of 16 Nov: 7,000-15,000 billed lookups over A, B and C)
A_EXPECTED_LOOKUPS = 1672  # stage A dating lookups counted on the saved pools (launch_cost_worksheet_2026-11-14.md)
S2_PER_SECOND = float(os.environ.get("RRN_S2_RATE", "1.0"))  # calls answered per second; the live check of 24 Nov measured about 1 per 13 s
FREE_DAILY = 1.0


def main() -> None:
    rows = [r for r in csv.DictReader(open(PLAN_CSV)) if r["stage"] in ("A", "B")]
    out_rows = []
    for r in rows:
        sf, li, si = float(r["oa_search_filter"] or 0), float(r["oa_list"] or 0), float(r["oa_singleton"] or 0)
        s2 = sf + (B_PAPER_LOOKUPS if r["step"].startswith("B5") else 0)
        fallback = sf * FALLBACK_RATE
        out_rows.append({"stage": r["stage"], "step": r["step"], "model_usd": float(r["model_usd"] or 0),
                         "s2_calls": round(s2), "openalex_list": round(li), "openalex_id_lookups_free": round(si + sf),
                         "openalex_fallback_searches": round(fallback), "openalex_usd": round(li * LIST + fallback * SEARCH, 4)})
    tot = {}
    for st in ("A", "B"):
        rs = [x for x in out_rows if x["stage"] == st]
        m = sum(x["model_usd"] for x in rs); oa = sum(x["openalex_usd"] for x in rs); s2 = sum(x["s2_calls"] for x in rs)
        tot[st] = {"model_usd": m, "model_with_retries": m * (1 + RETRY), "openalex_usd": oa, "openalex_with_retries": oa * (1 + RETRY),
                   "s2_calls": s2, "s2_hours": s2 * (1 + RETRY) / S2_PER_SECOND / 3600,
                   "openalex_free_days": oa * (1 + RETRY) / FREE_DAILY}
    a_exp = A_EXPECTED_LOOKUPS
    with open(R / "launch_cost_worksheet_2026-11-24.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0])); w.writeheader(); w.writerows(out_rows)
    src = f"measured in the live check ({FALLBACK_RATE:.0%})" if MEASURED else f"a bound of {FALLBACK_RATE:.0%} until the live check measures it"
    md = ["# Cost and time worksheet, version-2 launch, stages A and B (24 November 2026)", "",
          "Claude rows are the approved plan's (run_plan_2026-11-12.md; launch_cost_worksheet_2026-11-14.csv), unchanged by the change of "
          "sources. Scholarly data follows protocol version 2, section 8, change 4: Semantic Scholar for every search (free; 1 call a second with a key), "
          "OpenAlex records by id (free), OpenAlex list queries ($0.0001) and the OpenAlex title-search fallback ($0.001) for papers whose ids do not "
          f"resolve to a record with the same title. Fallback share: {src}. Retry allowance {RETRY:.0%} on every line. "
          "OpenAlex runs on its free daily budget only ($1 a day; the runner pauses when it is spent and never uses prepaid credit), so OpenAlex "
          "dollars below are days of free budget, not money.", "",
          "| Stage | Step | Model cost | Semantic Scholar calls | OpenAlex list | OpenAlex by id (free) | OpenAlex fallback searches | OpenAlex $ (free budget) |",
          "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for x in out_rows:
        md.append(f"| {x['stage']} | {x['step']} | ${x['model_usd']:.2f} | {x['s2_calls']:,} | {x['openalex_list']:,} | {x['openalex_id_lookups_free']:,} | "
                  f"{x['openalex_fallback_searches']:,} | ${x['openalex_usd']:.2f} |")
    md += ["", f"| Stage | Claude, with retries (paid) | Approved cap | Semantic Scholar calls, with retries | Hours at {S2_PER_SECOND:.3g} answered calls a second | OpenAlex free budget used | Days of free budget |",
           "| --- | --- | --- | --- | --- | --- | --- |"]
    for st in ("A", "B"):
        t = tot[st]
        md.append(f"| {st} | ${t['model_with_retries']:.2f} | ${CAPS[st]:.0f} | {t['s2_calls'] * (1 + RETRY):,.0f} | {t['s2_hours']:.1f} | "
                  f"${t['openalex_with_retries']:.2f} | {max(t['openalex_free_days'], 0):.1f} |")
    def expected(lookups: float, lists: float) -> tuple[float, float, float]:
        s2 = lookups * (1 + RETRY)
        oa = (lists * LIST + lookups * FALLBACK_RATE * SEARCH) * (1 + RETRY)
        return s2, s2 / S2_PER_SECOND / 3600, oa / FREE_DAILY
    b_list = sum(x["openalex_list"] for x in out_rows if x["stage"] == "B")
    ea = expected(a_exp, 0)
    eb_lo, eb_hi = expected(B_EXPECTED_LOOKUPS[0] + B_PAPER_LOOKUPS, b_list), expected(B_EXPECTED_LOOKUPS[1] + B_PAPER_LOOKUPS, b_list)
    md += ["", "**Expected** (what the saved data predicts, rather than the plan's caps):", "",
           f"| Stage | Semantic Scholar calls, with retries | Hours at {S2_PER_SECOND:.3g} answered calls a second | Days of OpenAlex free budget |", "| --- | --- | --- | --- |",
           f"| A | {ea[0]:,.0f} | {ea[1]:.1f} | {ea[2]:.1f} |",
           f"| B | {eb_lo[0]:,.0f}-{eb_hi[0]:,.0f} | {eb_lo[1]:.1f}-{eb_hi[1]:.1f} | {eb_lo[2]:.1f}-{eb_hi[2]:.1f} |"]
    md += ["", "Notes:",
           f"- **Upper bounds.** Lookup counts are the plan's caps (60 dating lookups and 40 bibliography lookups per run). Counted on the saved pools, stage A needs about {a_exp:,} "
           f"lookups, not 20,520: about {a_exp * (1 + RETRY) / 3600:.1f} hours of Semantic Scholar calls. Stage B's dating needs 12-33 lookups per run, not 60, and the "
           "leave-one-out harness caches every response per paper, so its real count is a fraction of the bound.",
           "- **Paid money is Claude only:** "
           f"stage A ${tot['A']['model_with_retries']:.2f} and stage B ${tot['B']['model_with_retries']:.2f} with retries, under the approved caps "
           "(A $21, B $43). The runner stops a stage at its cap, as before.",
           "- **No paid OpenAlex.** The runner reads OpenAlex's rate-limit endpoint every 30 s; below $0.05 of free budget it pauses the step and reruns it "
           "after the daily reset, and it stops the stage if prepaid credit is ever charged.",
           "- **Semantic Scholar key.** Without one, about half the calls were refused in the free-source test; the run starts only with the key.",
           "- **Live check with the key (24 Nov 2026).** 55 title lookups (40 + 15 from the free-source test's bibliography titles) and 3 reference lists: every real paper "
           "matched correctly, none wrongly; the titles not found were model cards, leaderboards and mis-parsed author lists. 6 of 59 Semantic Scholar lookups needed the "
           "OpenAlex title-search fallback (10%, the rate used above). Semantic Scholar refused most calls even with the key (keyed limit 1 a second): one answer every "
           "13 s on average with the tool's retries, the rate used for the hours above. At a full 1 a second the hours fall by 13x."]
    (R / "launch_cost_worksheet_2026-11-24.md").write_text("\n".join(md) + "\n")
    (R / "launch_cost_summary_2026-11-24.json").write_text(json.dumps({
        "fallback_rate": FALLBACK_RATE, "fallback_rate_measured": bool(MEASURED),
        "claude_with_retries": {s: round(tot[s]["model_with_retries"], 2) for s in tot},
        "openalex_free_budget_with_retries": {s: round(tot[s]["openalex_with_retries"], 2) for s in tot},
        "s2_calls_with_retries": {s: round(tot[s]["s2_calls"] * (1 + RETRY)) for s in tot},
        "paid_total_with_retries": round(sum(tot[s]["model_with_retries"] for s in tot), 2)}, indent=1))
    print(json.dumps(tot, indent=1))


if __name__ == "__main__":
    main()
