# Protocol amendment: stage C success criterion (proposed 16 November 2026)

Adds section 5a to protocol_2026-11-11.md. Proposed, not yet agreed. Once agreed it is merged into the protocol and the launch manifest is re-frozen before the first held-out draw (section 6).

## 5a. Stage C success criterion

- **Primary comparison.** User 1' (leave-one-out), benchmarks introduced in 2025 or 2026, on the held-out papers. Measure: the tool's final 10 against Haiku alone on the same papers, as the paired difference in recall (stated here as tool minus Haiku alone, the reverse of section 5's sign), benchmark-weighted, with the section 5 percentile bootstrap over papers (10,000 rounds, seed 7). Family matching on the truth-checked lists, with unresolved entries kept, is primary. The without-unresolved and exact-match versions are sensitivities.
- **Minimum improvement.** The tool leads by at least 10 points.
- **Interval.** The decision at each look uses a two-sided 95% interval, so 2.5% on each side. With two looks this keeps the overall chance of a false "confirmed" near 5%, the same as a single 90% interval at one look. The 90% intervals of section 5 are still reported.
- **Draw 1:** 30 papers, recipe heldout_recipe_2026-11-14.md (window 4 to 8 November 2026, seed 23).
  - *Confirmed:* the estimate is at least +10 and the 95% lower bound is above 0. Stop.
  - *Not supported:* the estimate is 0 or below. Stop.
  - *Otherwise:* draw 2. This covers an estimate above 0 that does not meet the confirmed rule.
- **Draw 2:** only under the rule above. 30 more papers, same recipe and tool version, window 9 to 13 November 2026, seed 29, excluding every earlier paper including draw 1. Same PDF check, sealing and truth check as draw 1. The maximum total sample is 60 papers. There is no third draw.
- **Final decision after draw 2:** made on the two draws together (60 papers, one paired bootstrap over all 60). Draw 1 is not judged again alone, and draw 2 is not judged alone.
  - *Confirmed:* the estimate is at least +10 and the 95% lower bound is above 0.
  - *Consistent, not confirmed:* the estimate is at least +10 and the interval includes 0. This is reported as inconclusive, and it is final.
  - *Not supported:* the estimate is below +10.
- **Secondary, reported but not part of the decision:** each draw separately, 2025 and 2026 separately, 2024 or earlier, all benchmarks pooled, user 1 (title and abstract), the paper-weighted estimand, and the four predictions in section 6.
- **Cost.** Draw 2 is a second stage C, about $10 by the launch worksheet, with its own $15 cap. It needs the group's approval of that cap before it runs. The rule only decides whether draw 2 is needed.

## Justification

From the development sets (post-cutoff a+b, 41 papers; age_rows_out_2026.json and age_rows_out_2026b.json), user 1', tool minus Haiku alone, with a 90% paired bootstrap over papers:

| Benchmarks introduced | n | Difference | 90% interval |
| --- | --- | --- | --- |
| 2025 | 23 | +17 points | 0 to +37 |
| 2025 or 2026 | 46 | +11 points | 0 to +21 |
| 2024 or earlier | 113 | -15 points | -22 to -8 |

- **Why 2025 and 2026 together.** On 30 papers, 2025 alone gives about 16 benchmarks and a 90% interval about 43 points wide. Pooling the two years gives about 33 benchmarks. It is also the recent group in the stage A rule.
- **Why 10 points.** It is the same bar as the stage A rule, and it is close to the development gap of +11.
- **Why this rule.** I simulated it by redrawing papers from the development set, 400 trials each.

| Outcome | If the development gap is real | If there is no gap |
| --- | --- | --- |
| Confirmed at 30 | 24% | 4% |
| Not supported at 30 (stop) | 7% | 41% |
| Confirmed at 60 | 26% | 2% |
| Consistent at 60 | 10% | 1% |
| Not supported at 60 | 32% | 52% |
| **Confirmed overall** | **about 50%** | **about 6%** |

  The second draw raises the chance of confirming a real gap from about 24% to about 50%, and keeps a false "confirmed" near 5%. The power figures are optimistic, because the development gap was measured on the data used to choose version 2. A stricter stop at 30 (stopping when the estimate is below +10) would end about 45% of real-gap cases early, which is why the stop is set at 0.
