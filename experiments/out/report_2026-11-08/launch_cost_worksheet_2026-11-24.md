# Cost and time worksheet, version-2 launch, stages A and B (24 November 2026)

Claude rows are the approved plan's (run_plan_2026-11-12.md; launch_cost_worksheet_2026-11-14.csv), unchanged by the change of sources. Scholarly data follows protocol version 2, section 8, change 4: Semantic Scholar for every search (free; 1 call a second with a key), OpenAlex records by id (free), OpenAlex list queries ($0.0001) and the OpenAlex title-search fallback ($0.001) for papers whose ids do not resolve to a record with the same title. Fallback share: measured in the live check (10%). Retry allowance 15% on every line. OpenAlex runs on its free daily budget only ($1 a day; the runner pauses when it is spent and never uses prepaid credit), so OpenAlex dollars below are days of free budget, not money.

| Stage | Step | Model cost | Semantic Scholar calls | OpenAlex list | OpenAlex by id (free) | OpenAlex fallback searches | OpenAlex $ (free budget) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | A1 date undated candidates (one call per run) | $0.24 | 0 | 0 | 0 | 0 | $0.00 |
| A | A2 confirm source titles in OpenAlex (<= 60 lookups per run, cached per title) | $0.00 | 20,520 | 0 | 20,520 | 2,052 | $2.05 |
| A | A3 rank with Sonnet, year-aware prompt (one call per run) | $7.85 | 0 | 0 | 0 | 0 | $0.00 |
| A | A4 score against the saved judge matches | $0.00 | 0 | 0 | 0 | 0 | $0.00 |
| B | B1 user-1 runs: plan, read related abstracts, label bibliographies, propose (Haiku) | $0.53 | 0 | 0 | 0 | 0 | $0.00 |
| B | B2 leave-one-out runs: same pipeline on the scrubbed paper (Haiku) | $1.61 | 0 | 0 | 0 | 0 | $0.00 |
| B | B3 dating call per run (Haiku) | $0.32 | 0 | 0 | 0 | 0 | $0.00 |
| B | B4 ranking call per run (Sonnet, year-aware) | $10.46 | 0 | 0 | 0 | 0 | $0.00 |
| B | B5 OpenAlex per run: ~30 as before + <= 40 bibliography lookups + <= 60 dating lookups | $0.00 | 55,857 | 3,746 | 55,534 | 5,540 | $5.91 |
| B | B6 judge re-matching where the pool changed (Haiku, one call per run) | $0.21 | 0 | 0 | 0 | 0 | $0.00 |
| B | B7 arXiv HTML bibliography fetch (one per paper) | $0.00 | 0 | 0 | 0 | 0 | $0.00 |
| B | B8 Haiku-alone baselines, probes, Haiku-alone leave-one-out | $0.00 | 0 | 0 | 0 | 0 | $0.00 |

| Stage | Claude, with retries (paid) | Approved cap | Semantic Scholar calls, with retries | Hours at 0.077 answered calls a second | OpenAlex free budget used | Days of free budget |
| --- | --- | --- | --- | --- | --- | --- |
| A | $9.30 | $21 | 23,598 | 85.1 | $2.36 | 2.4 |
| B | $15.09 | $43 | 64,236 | 231.7 | $6.80 | 6.8 |

**Expected** (what the saved data predicts, rather than the plan's caps):

| Stage | Semantic Scholar calls, with retries | Hours at 0.077 answered calls a second | Days of OpenAlex free budget |
| --- | --- | --- | --- |
| A | 1,923 | 6.9 | 0.2 |
| B | 6,274-12,024 | 22.6-43.4 | 1.1-1.6 |

Notes:
- **Upper bounds.** Lookup counts are the plan's caps (60 dating lookups and 40 bibliography lookups per run). Counted on the saved pools, stage A needs about 1,672 lookups, not 20,520: about 0.5 hours of Semantic Scholar calls. Stage B's dating needs 12-33 lookups per run, not 60, and the leave-one-out harness caches every response per paper, so its real count is a fraction of the bound.
- **Paid money is Claude only:** stage A $9.30 and stage B $15.09 with retries, under the approved caps (A $21, B $43). The runner stops a stage at its cap, as before.
- **No paid OpenAlex.** The runner reads OpenAlex's rate-limit endpoint every 30 s; below $0.05 of free budget it pauses the step and reruns it after the daily reset, and it stops the stage if prepaid credit is ever charged.
- **Semantic Scholar key.** Without one, about half the calls were refused in the free-source test; the run starts only with the key.
- **Live check with the key (24 Nov 2026).** 55 title lookups (40 + 15 from the free-source test's bibliography titles) and 3 reference lists: every real paper matched correctly, none wrongly; the titles not found were model cards, leaderboards and mis-parsed author lists. 6 of 59 Semantic Scholar lookups needed the OpenAlex title-search fallback (10%, the rate used above). Semantic Scholar refused most calls even with the key (keyed limit 1 a second): one answer every 13 s on average with the tool's retries, the rate used for the hours above. At a full 1 a second the hours fall by 13x.
