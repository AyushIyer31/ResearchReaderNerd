"""Bundle the three launch artifacts for Harsh (14 Nov 2026), with the hashed code and its diff against package v7.
No provider calls.

  python launch_bundle.py -> out/report_2026-11-08/launch_2026-11-14/ and launch_2026-11-14.zip
"""
from __future__ import annotations

import difflib
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import launch  # noqa: E402

REPORT = HERE / "out" / "report_2026-11-08"
DEST = REPORT / "launch_2026-11-14"
V7 = REPORT / "for_harsh_v7"


def unified(old: Path, new: Path, a: str, b: str) -> list[str]:
    x = old.read_text().splitlines(keepends=True) if old.exists() else []
    return list(difflib.unified_diff(x, new.read_text().splitlines(keepends=True), a, b))


def main() -> None:
    bad = launch.mismatches()
    if bad:
        sys.exit("files differ from the launch manifest; run launch.py freeze first:\n  " + "\n  ".join(bad))
    if DEST.exists():
        shutil.rmtree(DEST)
    (DEST / "code" / "src" / "researchreadernerd").mkdir(parents=True)
    (DEST / "code" / "experiments").mkdir(parents=True)
    for f in ("launch_manifest_2026-11-14.json", "launch_manifest_2026-11-14.md", "heldout_recipe_2026-11-14.md",
              "launch_cost_worksheet_2026-11-14.md", "launch_cost_worksheet_2026-11-14.csv", "launch_cost_summary_2026-11-14.json"):
        shutil.copy(REPORT / f, DEST / f)
    (DEST / "decisions").mkdir()
    shutil.copy(REPORT / "hand_check" / "decisions_batch6.json", DEST / "decisions" / "truth_decisions_2026-11-14.json")
    src = HERE.parent / "src" / "researchreadernerd"
    for f in launch.TOOL_FILES:
        shutil.copy(src / f, DEST / "code" / "src" / "researchreadernerd" / f)
    for f in launch.SCRIPTS + launch.INPUT_FILES:
        shutil.copy(HERE / f, DEST / "code" / "experiments" / f)
    shutil.copy(HERE.parent / "pyproject.toml", DEST / "code" / "pyproject.toml")
    shutil.copy(HERE / "launch_bundle.py", DEST / "code" / "experiments" / "launch_bundle.py")
    # inputs hashed in the manifest that package v7 does not hold as hashed here: the paper readings, their verbatim
    # checks, and the truth lists (changed on 14 Nov by the four availability decisions)
    for d in ("out_2026", "out_2026b", "out_old_v2"):
        for sub in ("extract", "verbatim", "truth"):
            shutil.copytree(HERE / d / sub, DEST / "inputs" / d / sub)
    diff = []
    for f in launch.TOOL_FILES:
        diff += unified(V7 / "snapshot" / "src_v2_untested" / f, src / f, f"v7/snapshot/src_v2_untested/{f}", f"launch/code/src/researchreadernerd/{f}")
    for f in launch.SCRIPTS + ["launch_bundle.py"]:
        if f.endswith(".py"):
            diff += unified(V7 / "scripts" / f, HERE / f, f"v7/scripts/{f}", f"launch/code/experiments/{f}")
    (DEST / "changes_since_v7.diff").write_text("".join(diff))
    man = json.loads((REPORT / "launch_manifest_2026-11-14.json").read_text())
    cost = json.loads((REPORT / "launch_cost_summary_2026-11-14.json").read_text())
    (DEST / "README.md").write_text(f"""# Launch artifacts (14 Nov 2026)

1. **Launch manifest and configuration with hashes:** launch_manifest_2026-11-14.json (binding; frozen {man['frozen']}; sha256 in SHA256SUMS) and launch_manifest_2026-11-14.md (readable).
2. **Stage C held-out recipe, seed and exclusions:** heldout_recipe_2026-11-14.md; the copy the runner executes is code/experiments/heldout_recipe_2026-11-14.json.
3. **Per-step token and endpoint cost worksheet:** launch_cost_worksheet_2026-11-14.md and .csv; the totals are in launch_cost_summary_2026-11-14.json: ${cost['plan_total_with_retries']:.2f} for the approved plan and ${cost['runner_total_with_retries']:.2f} for what the runner does, both with retries, against $79.

Nothing has run. Every stage runs only through code/experiments/launch.py, and only after your written go-ahead.

## Checking the hashes

- code/ holds every hashed tool file, script, paper list and pyproject.toml.
- Input folders are hashed as folder hashes (tree_sha in launch.py). The manifest records {{files, sha256}}, where sha256 is one hash fed, for each file in sorted path order, the UTF-8 bytes of '<path relative to the folder>\\0<file sha256 hex>\\n'; a key with a pattern (such as blind/*.claude-haiku-5-5.*+model.json) hashes only the files matching it.
- inputs/<set>/ holds the paper readings (extract), their verbatim checks and the truth lists. The other hashed inputs (probe, probe_forced, loo_llm, loo_llm_match, abl, match, blind, loo, rerank, results, memory_probe files) are in for_harsh_v7 under each set's runs/ or scoring/ (out_2026 = post_cutoff_a, out_2026b = post_cutoff_b, out_old_v2 = old_2018-2021). The OpenAlex response cache (oa_cache) is not shipped.
- Hashed documents not in this bundle are the copies already sent: protocol_2026-11-11.md, run_plan_2026-11-12.md, run_request_2026-11-12.md, and years_haiku.json in for_harsh_v7/report/.

## How the runner runs and stops

- Order: stage A; `decide`, which needs a complete A and records whether version 2 includes dating and the recency rule, bound to the manifest's hash; B; C1; the truth check by hand; C2. Per the approved run request, B runs for either outcome of A, and the manifest covers both versions through that one switch (protocol section 8 had made B conditional on A).
- Every run and decide refuses if any hashed file, document included, differs from the manifest. freeze never overwrites the manifest (with --replace it archives the old one first), and every status file, log, decision and draw record carries the manifest's sha256, so each result traces to the manifest sent.
- C1 draws only after B is complete (or after the group's recorded decision to go on from a B that stopped early) and after the version-2 predictions are recorded; it pauses after the draw for the recipe's PDF check, records the sample's hashes, runs, and seals the pre-check scoring together with the tool's runs and the Haiku-alone answers, so that the truth-check reviewer sees neither. A complete C1 is never run again.
- C2 starts only when every sampled paper's truth list is marked checked and the lists still hash to what the check recorded; it restores the sealed runs, re-judges on the checked lists, and is never run twice.
- A stage stops at its cap less $0.50, at $79 less $0.50 over all stages, or when its spend projected from the share of its runs done exceeds the cap after 10%; checked before and after every step and every 30 s during it. Stage C's parts share the cap and have separate meters. If OpenAlex spend cannot be read, the stage stops. A stopped stage keeps its results and is reported as partial.
- Spend: every Claude response that arrives is written to a ledger at pricing.py's prices, refused, cut-off and malformed answers included (a model missing from the price table is charged at Sonnet's price); a response lost to a network failure or to the runner stopping a step is not, which the $0.50 margin covers. OpenAlex dollars come from its rate-limit endpoint (fields as in code/experiments/openalex_rate_limit_2026-11-13.json, a real response); the endpoint reports the key's whole use, so any other use of the key that day is counted too.
- Retries: a paid step that stops at a failed call runs up to three times, each pass skipping what is cached. A run that still fails is left out, listed in the age table and match_sensitivity, and its cells are left out of n.

## Changes since package v7

changes_since_v7.diff has every code change since package v7's snapshot/src_v2_untested and scripts/. Five changes close gaps between the approved plan and the code:

1. User-1 runs with a reference list (refs+model) read the paper's own arXiv bibliography, as version 2 says (RRN_USER1_ARXIV_BIB, benchmark_eval.py). topic+model reads no reference list and is unchanged.
2. The recency rule is one switch with dating (RECENCY_RULE, analyze.py and consider.py). With dating off the ranker gets the exact as-run prompt and listing.
3. Stage A's offline dating follows the live rule: literature candidates only, at most 60 OpenAlex lookups per run (v2_offline.py; shipped in package v6, not v7, so the diff shows the whole file). Two differences from the live run remain: the replay caches lookups by title within a set, and it keeps an OpenAlex year that the live run would reject because the match is the source paper of one of the paper's own benchmarks.
4. Sonnet's server-side fallback is off (RRN_SERVER_FALLBACK=0), so every answer comes from the model named.
5. Score replays are off in launch runs (RRN_SKIP_REPLAY=1); they were not in the plan.

Fixes from checking the launch end to end, including an offline run of every stage with the providers stubbed:

- Stage A no longer imports leave_one_out (whose import routed OpenAlex calls through a cache inside the hashed as-run folders), creates its dating folder, skips and retries a run with a failed call, and writes its rows only when every run succeeded; its rows follow the current truth lists.
- benchmark_eval.py redo retries runs that ended in an error or whose dating stopped, and failed readings other than an unreadable PDF; run and score refuse while the held-out runs are sealed.
- The arXiv bibliography is fetched once per paper per process, a second apart; a missing or failed fetch is written to the OpenAlex ledger.
- age_split.py takes a hand-assigned year before the model's memory (protocol section 3); no development year changes, since no benchmark left undated by the stored years and the tool's evidence has a hand year that differs from Haiku's (checked on all three sets). It also tolerates a missing run, and its tables add the user-1 topic+model column, both memory probes by benchmark age (protocol section 4), a single set with 90% Wilson intervals, and the headline with and without unresolved entries.

New for the held-out sample:

- heldout_truth_check.py: protocol section 3's check, by the development sets' procedure (Gemini in its chat window, no API call). packet writes each paper's prompt, the JSON answer format and its pre-check list; review validates the saved answers and lists every non-keep verdict, every entry whose availability the PDF does not show, and every year to be adopted; the reviewer decides each; apply applies only reviewed verdicts with a quoted location, applies the availability decisions (public with evidence, own, or unresolved), takes years from the quoted citation (the benchmark's arXiv id first, else the printed year if it is in the quote), keeps the pre-check lists, and records the lists' hash for C2.
- match_sensitivity.py held-out mode: section 5's exact-match sensitivity and paired bootstrap (benchmark- and paper-weighted, 10,000 rounds, seed 7), with the user-1 topic+model rows. Its development-set output is unchanged byte for byte. combined_report.py is the development-set narrative and is not run on the held-out sample; the held-out tables come from age_split.py and match_sensitivity.py.
- Instrumentation: the two ledgers, outputs under out/launch_2026-11-14/, the saved draw response.

## Truth lists: the four open cases (protocol section 3)

Settled on 14 Nov by the 12 Nov availability rule, with the evidence in decisions/truth_decisions_2026-11-14.json and in each entry's hand_check record: EmulatorBench (Prime Agent) is own (no release before the paper; cited to the authors' unpublished manuscript; a 'preview benchmark'); PMPP-Hard and MazeBench (Prime Agent) and the Mental Health Multimodal tabular set (MERID) stay public (dated releases before each paper). Stage B therefore has 275 leave-one-out runs, one fewer than the plan's 276.

## Departures from the protocol and run request, stated

- Seed: protocol section 2 names seed 7, the seed of the development draws; section 6 has the held-out seed stated before the draw, which is 23.
- Truth check order: section 3 checks every scored paper before scoring. The held-out pre-check scoring in C1 is sealed unread and kept only as the record of what the check changed; every reported held-out number is scored on the checked lists in C2.
- Version-2 predictions: recorded in out/launch_2026-11-14/predictions_v2.json before the draw, as protocol section 8 requires; the runner checks they exist and binds their hash into the draw record. They will be sent to you before the draw.

## Known limits

- The OpenAlex response cache is not shipped, so its three folder hashes can be checked only on this machine.
- The code is git commit {man['git']['commit'][:7]} plus uncommitted changes; experiments/ is untracked and hashed file by file.
- If OpenAlex bills title lookups at its search price, stages B and C would exceed their caps; the runner then stops that stage early (worst case ${cost['worst_case_search_price_total']:.2f} over all stages).
- The run needs about ${cost['openalex_plan_rate_with_retries']:.0f} of OpenAlex credit beyond what the key holds; when the daily budget and prepaid credit are spent, leave-one-out runs stop part-way, the runner waits for funds before the next step, and the runs are retried.
- Stage B's leave-one-out runs reuse the as-run OpenAlex response cache for identical queries.
- A failed arXiv fetch leaves that run without the paper's own bibliography (change 3); failures are counted in the OpenAlex ledger and reported, not retried.
- The held-out window starts after Haiku 5.5's June 2026 training cutoff; Sonnet 5.5, which only ranks candidates the tool has found, is not separately checked against the window.
""")
    import hashlib
    sums = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(DEST)}" for p in sorted(DEST.rglob("*")) if p.is_file()]
    (DEST / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    z = shutil.make_archive(str(DEST), "zip", root_dir=REPORT, base_dir=DEST.name)
    print(z, f"{Path(z).stat().st_size / 1e3:.0f} KB", f"{len(sums)} files")


if __name__ == "__main__":
    main()
