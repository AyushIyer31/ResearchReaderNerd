"""Launch runner for the approved plan (13 Nov 2026; Harsh approved a total ceiling of $79 including retries).

  python launch.py freeze          write the launch manifest (config + sha256 of every file the runs read or execute)
  python launch.py check [STAGE]   verify the files against the manifest and show balances; no paid call
  python launch.py run A           stage A: cached replay (dating + Sonnet year-aware ranking on the 342 saved pools)
  python launch.py decide          apply the predeclared rule to stage A's rows; records the version-2 configuration
  python launch.py run B           stage B: live version-2 development run on the 90 existing papers
  python launch.py run C1          stage C, part 1: draw (then pauses for the PDF check and any swap), read, user-1 runs,
                                   Haiku alone, sealed pre-check scoring (then the truth check by hand, protocol
                                   section 3: heldout_truth_check.py packet, Gemini's answers, review, apply)
  python launch.py run C2          stage C, part 2, after the truth check: scoring on the checked lists, leave-one-out,
                                   probes, leak check, age table, match sensitivity and paired bootstrap

Every stage, and decide, refuses to start if a file differs from the manifest. During a stage every Claude response that
arrives (including a refused, cut-off or malformed answer) is appended to a spend ledger; a response lost to a network
failure or to the runner stopping a step is not, which the $0.50 stop margin covers (llm.py, RRN_SPEND_LEDGER) and every OpenAlex response to an endpoint ledger (scholar.py,
RRN_OPENALEX_LEDGER); OpenAlex dollars are read from its rate-limit endpoint before and after every step and every 30 s
during it. A step is not started, or is stopped, and the stage ends, reported as partial, when the stage's spend reaches its cap, when all stages together reach $79, or when the
spend projected from the share of runs done exceeds the cap once 10% are done. Stages B and C write to new folders;
stage A writes new files (dating/, v2_rerank/, v2_offline rows) inside the as-run folders and changes no hashed file.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PY = sys.executable
LAUNCH = HERE / "out" / "launch_2026-11-14"
REPORT = HERE / "out" / "report_2026-11-08"
MANIFEST = REPORT / "launch_manifest_2026-11-14.json"
RECIPE = HERE / "heldout_recipe_2026-11-14.json"
DECISION = LAUNCH / "decision.json"
TRUTH_CHECKED = LAUNCH / "heldout_truth_checked.json"  # protocol section 3 truth check, every sampled paper
PDFS_CHECKED = LAUNCH / "heldout_pdfs_checked.json"  # recipe: every sampled PDF readable, swaps recorded, before any run
PREDICTIONS = LAUNCH / "predictions_v2.json"  # protocol section 8: version-2 predictions recorded before the draw
SEALED_RUNS = ("blind", "abl", "replay")  # held-out tool runs and Haiku-alone answers, sealed from the end of C1 to the start of C2
B_PARTIAL_OK = LAUNCH / "stage_b_partial_accepted.json"  # the group's recorded decision to go on after a stage B that stopped early

HAIKU, SONNET = "claude-haiku-5-5", "claude-sonnet-5-5"
CAPS = {"A": 21.0, "B": 43.0, "C": 15.0}  # approved 13 Nov 2026 (run_plan_2026-11-12.md)
TOTAL_CAP = 79.0
PROJECT_AFTER = 0.10  # projection check starts once this share of a stage's runs is done
STOP_MARGIN = 0.50  # stop this far below a cap: spend is read every 30 s, and a few calls can land between reads
MIN_OPENALEX_FUNDS = 0.50  # a step starts only with at least this much OpenAlex money left (daily budget + prepaid)

AS_RUN = {"post_cutoff_a": ("out_2026", "papers_2026.json"), "post_cutoff_b": ("out_2026b", "papers_2026b.json"), "old": ("out_old_v2", "papers.json")}
V2 = {"post_cutoff_a": ("out_v2_2026", "papers_2026.json"), "post_cutoff_b": ("out_v2_2026b", "papers_2026b.json"), "old": ("out_v2_old", "papers.json")}
HELDOUT = ("out_heldout", "papers_heldout.json")
# Tool-independent inputs copied from each as-run folder into its stage B folder: the checked truth lists, the PDF
# readings, the Haiku-alone answers and their judge matches, the probes, and the OpenAlex response cache of the
# leave-one-out harness. Tool outputs (blind, loo, their matches, rankings, results) are not copied.
SEED_DIRS = ["truth", "extract", "verbatim", "probe", "probe_forced", "loo_llm", "loo_llm_match", "oa_cache"]
SEED_GLOBS = [("abl", f"*.{HAIKU}.llm.json"), ("match", f"*.{HAIKU}.llm.json"), ("match", f"*.{HAIKU}.probe.json"), ("match", f"*.{HAIKU}.probe_forced.json")]
SEED_FILES = ["memory_probe.md", "memory_probe_forced.json"]

TOOL_FILES = sorted(p.name for p in (ROOT / "src" / "researchreadernerd").glob("*.py"))
SCRIPTS = ["launch.py", "benchmark_eval.py", "leave_one_out.py", "loo_llm.py", "ablation.py", "memory_probe.py", "memory_probe_forced.py",
           "leak_check.py", "age_split.py", "v2_offline.py", "sonnet_rerank.py", "compare_sets.py", "quota.py", "sample_2026.py",
           "match_sensitivity.py", "run_plan.py", "launch_worksheet.py", "heldout_truth_check.py", "openalex_rate_limit_2026-11-13.json", "match_overrides.json", "years_manual.json", "heldout_recipe_2026-11-14.json"]
INPUT_FILES = ["papers.json", "papers_2026.json", "papers_2026b.json", "papers_2026_swaps.json"]
DOCS = ["protocol_2026-11-11.md", "run_plan_2026-11-12.md", "run_request_2026-11-12.md", "launch_cost_worksheet_2026-11-14.md",
        "launch_cost_worksheet_2026-11-14.csv", "launch_cost_summary_2026-11-14.json", "heldout_recipe_2026-11-14.md", "years_haiku.json"]  # years_haiku.json: age_split's seed year cache

# The rule of protocol section 8, as applied to stage A (fixed 13 Nov 2026 before stage A runs). Rows: the pooled
# post-cutoff sets (a + b), leave-one-out (user 1'), final 10. 'v2' = Sonnet ranking on dated candidates with the
# recency rule, reserved tier as live; 'base' = Sonnet ranking without years, reserved tier as live (version 2 without
# the rule). Adopt the rule if v2 - base >= +10 points on the benchmarks introduced in 2025 or 2026 pooled, and
# v2 - base >= -3 points on those introduced in 2024 or earlier; otherwise version 2 is Sonnet + bibliography without it.
DECISION_RULE = {"test": "user 1'", "sets": ["out_2026", "out_2026b"], "measure": "final 10, benchmark-weighted recall",
                 "v2": "v2: Sonnet, dated, + tier", "base": "Sonnet, no years, + tier",
                 "recent_bins": [2025, 2026], "recent_min_gain_points": 10.0, "older_bin": "2024 or earlier", "older_max_loss_points": 3.0}


def env_for(part: str, out: str | None = None, papers: str | None = None, date_candidates: bool | None = None) -> dict:
    """The launch settings of every step: models, switches and the two ledgers of the stage part (A, B, C1, C2). They are
    laid over the shell environment, which supplies only PATH and the like (the API keys come from .env)."""
    e = {"RRN_RANK_MODEL": SONNET, "RRN_SERVER_FALLBACK": "0", "RRN_LAUNCH_REPORT": str(LAUNCH / "report"),
         "RRN_SPEND_LEDGER": str(LAUNCH / f"spend_{part}.jsonl"), "RRN_OPENALEX_LEDGER": str(LAUNCH / f"openalex_{part}.jsonl")}
    if part[0] in ("B", "C"):
        e["RRN_USER1_ARXIV_BIB"] = "1"
        e["RRN_SKIP_REPLAY"] = "1"
        e["RRN_DATE_CANDIDATES"] = "1" if date_candidates else "0"
    if out:
        e["RRN_OUT"], e["RRN_PAPERS"] = out, papers
    return e


def tries(step: dict, n: int = 3) -> list[dict]:
    """A paid step that stops at its first failed call, run up to n times: each pass skips what is cached and retries the
    rest; only the last pass must succeed."""
    return [step | {"ok_codes": (0, 1) if i < n - 1 else (0,)} for i in range(n)]


def steps_for(stage: str, date_candidates: bool | None = None) -> list[dict]:
    """Each step: the command, its environment, and the runs it produces (for the projection check)."""
    S = []
    if stage == "A":
        for out, papers in AS_RUN.values():
            for attempt in range(3):  # a run with a failed call is skipped and retried by the next pass; finished runs are cached
                S.append({"cmd": ["v2_offline.py"], "env": env_for("A", out, papers), "progress": (f"{out}/v2_rerank/*.json", f"{out}/rerank_{SONNET}/*.json"),
                          "ok_codes": (0, 1) if attempt < 2 else (0,)})
        S.append({"cmd": ["v2_offline.py", "summarize", "out_2026", "out_2026b"], "env": env_for("A")})
        S.append({"cmd": ["v2_offline.py", "summarize", "out_old_v2"], "env": env_for("A") | {"RRN_LAUNCH_REPORT": str(LAUNCH / "report" / "old")}})
    elif stage == "B":
        for label, (out, papers) in V2.items():
            e = env_for("B", out, papers, date_candidates)
            as_run = AS_RUN[label][0]
            for _ in range(3):  # failed lookups are deleted by redo and retried, as in the as-run runner
                S.append({"cmd": ["benchmark_eval.py", "run", "--models", HAIKU, "--conditions", "refs+model,topic+model"], "env": e,
                          "progress": (f"{out}/blind/*.json", f"{as_run}/blind/*.{HAIKU}.*+model.json")})
                S.append({"cmd": ["benchmark_eval.py", "redo"], "env": e})
            S += tries({"cmd": ["benchmark_eval.py", "score", "--models", HAIKU], "env": e})
            for _ in range(3):
                S.append({"cmd": ["leave_one_out.py", "run"], "env": e, "progress": (f"{out}/loo/*.json", f"loo:{out}:{papers}"), "ok_codes": (0, 1, 2) if _ < 2 else (0, 1)})
            S += [{"cmd": ["leave_one_out.py", "score"], "env": e}, *tries({"cmd": ["loo_llm.py"], "env": e}), {"cmd": ["leak_check.py"], "env": e}]
        S += tries({"cmd": ["age_split.py", V2["post_cutoff_a"][0], V2["post_cutoff_a"][1], V2["post_cutoff_b"][0], V2["post_cutoff_b"][1]],
                    "env": env_for("B", date_candidates=date_candidates) | {"RRN_AGE_TABLE": "age_table_v2_tool.md"}})
        S += tries({"cmd": ["age_split.py", V2["old"][0], V2["old"][1]], "env": env_for("B", date_candidates=date_candidates) | {"RRN_AGE_TABLE": "age_table_v2_tool_old.md"}})
    elif stage == "C1":
        r = json.loads(RECIPE.read_text())
        e = env_for("C1", *HELDOUT, date_candidates)
        S.append({"cmd": r["command"].split()[1:], "env": env_for("C1", date_candidates=date_candidates), "once": HELDOUT[1]})
        for _ in range(3):
            S.append({"cmd": ["benchmark_eval.py", "run", "--models", HAIKU, "--conditions", "refs+model,topic+model"], "env": e,
                      "progress": (f"{HELDOUT[0]}/blind/*.json", 60)})
            S.append({"cmd": ["benchmark_eval.py", "redo"], "env": e})
        S += tries({"cmd": ["ablation.py", "run", "--models", HAIKU, "--conds", "llm"], "env": e})
        S += tries({"cmd": ["benchmark_eval.py", "score", "--models", HAIKU], "env": e, "seal": True})
    elif stage == "C2":
        e = env_for("C2", *HELDOUT, date_candidates)
        S += tries({"cmd": ["benchmark_eval.py", "score", "--models", HAIKU], "env": e})
        for _ in range(3):
            S.append({"cmd": ["leave_one_out.py", "run"], "env": e, "progress": (f"{HELDOUT[0]}/loo/*.json", f"loo:{HELDOUT[0]}:{HELDOUT[1]}"), "ok_codes": (0, 1, 2) if _ < 2 else (0, 1)})
        S += [{"cmd": ["leave_one_out.py", "score"], "env": e}, *tries({"cmd": ["loo_llm.py"], "env": e}),
              *tries({"cmd": ["memory_probe.py"], "env": e}), *tries({"cmd": ["memory_probe_forced.py"], "env": e}), {"cmd": ["leak_check.py"], "env": e},
              *tries({"cmd": ["age_split.py", *HELDOUT], "env": env_for("C2", date_candidates=date_candidates) | {"RRN_AGE_TABLE": "age_table_heldout.md"}}),
              # protocol section 5: exact-match sensitivity and the paired bootstrap (both estimands) on the held-out sample
              {"cmd": ["match_sensitivity.py", "held-out", *HELDOUT], "env": env_for("C2", date_candidates=date_candidates)}]
    return S


# ---------- hashes and the manifest ----------

def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def tree_sha(d: Path, pattern: str = "*") -> dict:
    """One hash over a folder's files (sorted relative paths and their sha256), with the file count."""
    files = sorted(f for f in d.glob(pattern) if f.is_file()) if d.exists() else []
    h = hashlib.sha256()
    for f in files:
        h.update(f"{f.relative_to(d)}\0{sha(f)}\n".encode())
    return {"files": len(files), "sha256": h.hexdigest()}


def file_hashes() -> dict:
    src = ROOT / "src" / "researchreadernerd"
    out = {"tool_source (src/researchreadernerd)": {f: sha(src / f) for f in TOOL_FILES},
           "scripts (experiments)": {f: sha(HERE / f) for f in SCRIPTS if (HERE / f).exists()},
           "paper lists (experiments)": {f: sha(HERE / f) for f in INPUT_FILES},
           "documents (report folder)": {f: sha(REPORT / f) for f in DOCS if (REPORT / f).exists()},
           "pyproject.toml": sha(ROOT / "pyproject.toml")}
    seeds, pools = {}, {}
    for label, (d, _) in AS_RUN.items():
        base = HERE / d
        seeds[d] = {sub: tree_sha(base / sub) for sub in SEED_DIRS}
        seeds[d] |= {f"{sub}/{pat}": tree_sha(base / sub, pat) for sub, pat in SEED_GLOBS}
        seeds[d] |= {f: sha(base / f) for f in SEED_FILES if (base / f).exists()}
        pools[d] = {f"blind/*.{HAIKU}.*+model.json": tree_sha(base / "blind", f"*.{HAIKU}.*+model.json"), "loo": tree_sha(base / "loo"),
                    "results.json": sha(base / "results.json"), "loo_results.json": sha(base / "loo_results.json"),
                    f"rerank_{SONNET}": tree_sha(base / f"rerank_{SONNET}")}
    out["stage B seed inputs (copied from the as-run folders)"] = seeds
    out["stage A saved pools (read, not changed)"] = pools
    return out


def prompts() -> dict:
    sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(HERE))
    from researchreadernerd import analyze
    import ablation, benchmark_eval, loo_llm, memory_probe, memory_probe_forced  # noqa: E401
    P = {f"analyze.Prompts.{k}": v for k, v in vars(analyze.Prompts()).items()}
    P |= {"analyze.RECENCY_RULE": analyze.RECENCY_RULE, "benchmark_eval.MATCH_PROMPT": benchmark_eval.MATCH_PROMPT,
          "benchmark_eval.DEDUPE_PROMPT": benchmark_eval.DEDUPE_PROMPT, "ablation.LLM_ONLY_PROMPT": ablation.LLM_ONLY_PROMPT,
          "ablation.DATE_PROMPT (age_split year fallback)": ablation.DATE_PROMPT, "loo_llm.PROMPT": loo_llm.PROMPT,
          "memory_probe.PROBE_PROMPT": memory_probe.PROBE_PROMPT, "memory_probe_forced.PROMPT": memory_probe_forced.PROMPT}
    return {k: {"sha256": hashlib.sha256(v.encode()).hexdigest(), "chars": len(v), "in_use": bool(v)} for k, v in P.items()}


def versions() -> dict:
    import importlib.metadata as md
    return {"python": sys.version.split()[0], **{p: md.version(p) for p in ("anthropic", "httpx2", "pydantic", "pypdf") if _has(p)}}


def _has(p: str) -> bool:
    import importlib.metadata as md
    try:
        md.version(p); return True
    except md.PackageNotFoundError:
        return False


def cmd_freeze(replace: bool = False) -> None:
    """Write the manifest. An existing manifest is never overwritten silently: with --replace it is first archived under
    out/launch_2026-11-14/replaced_manifests/ with its sha256, and every later status, log and decision records the new hash."""
    if MANIFEST.exists():
        if not replace:
            sys.exit(f"{MANIFEST.name} exists (sha256 {sha(MANIFEST)}); it is not overwritten. 'launch.py freeze --replace' archives it first.")
        arch = LAUNCH / "replaced_manifests"; arch.mkdir(parents=True, exist_ok=True)
        shutil.copy(MANIFEST, arch / f"{datetime.now():%Y%m%dT%H%M%S}_{sha(MANIFEST)[:16]}.json")
    sys.path.insert(0, str(ROOT / "src"))
    from researchreadernerd import consider, pricing
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout  # noqa: E731
    man = {
        "name": "launch manifest, approved plan (A cached replay, B live version-2 development run, C held-out sample)",
        "frozen": datetime.now().isoformat(timespec="seconds"),
        "approval": {"by": "Harsh", "date": "2026-11-13", "total_ceiling_usd": TOTAL_CAP, "including_retries": True},
        "git": {"commit": git("rev-parse", "HEAD").strip(), "working_tree_uncommitted": True,
                "diff_vs_commit_sha256": hashlib.sha256(git("diff", "HEAD").encode()).hexdigest()},
        "versions": versions(),
        "models": {"every call except ranking": HAIKU, "ranking call (one per run)": SONNET, "judge": HAIKU, "server_side_fallback": "off (RRN_SERVER_FALLBACK=0): every answer is from the model named; a decline is an error, not another model's answer",
                   "efforts": {"tool: plan, propose, rank": "medium", "tool: extract, label, date": "low", "judge (matching, grouping)": "low",
                               "PDF reading": "medium, low on an output-cap overrun", "Haiku alone (user 1 and leave-one-out)": "medium",
                               "memory probes": "medium", "age_split dating of undated truth entries": "medium"},
                   "max_tokens": {"rank": 8000, "date": 8000, "judge (user-1 matching, grouping)": 4000, "judge (leave-one-out, loo_llm)": 2000,
                                  "Haiku alone": 2000, "memory probes": 1500, "age_split dating": 4000}},
        "tool_constants": {k: getattr(consider, k) for k in ("SUGGESTIONS", "RESERVED_PROPOSALS", "RECENT_SLOTS", "MAX_REFERENCES", "MONTHS_BEFORE",
                                                              "MAX_PROPOSALS", "MAX_DATE_LOOKUPS", "MAX_LITERATURE_CANDIDATES", "MAX_BENCHMARKS_FOR_COUSAGE",
                                                              "CITING_PER_BENCHMARK", "PAPERS_PER_EXTRACTION_CALL", "MAX_CITED_TO_LABEL")}
                          | {"MAX_ARXIV_REF_LOOKUPS": __import__("researchreadernerd.scholar", fromlist=["x"]).MAX_ARXIV_REF_LOOKUPS, "budget_usd_per_run": 1.0},
        "version_2": {"changes": ["1. ranking call by claude-sonnet-5-5 (RRN_RANK_MODEL)",
                                  "2. candidate dating with the recency rule (RRN_DATE_CANDIDATES=1): dating call, years shown to the ranker, RECENCY_RULE appended to the ranking prompt; with 0 the ranker sees the as-run prompt and listing",
                                  "3. the paper's own arXiv bibliography in find_paper (leave-one-out runs pass the arXiv id as before; user-1 runs with a reference list get it through RRN_USER1_ARXIV_BIB=1 in benchmark_eval.py, which looks the paper up by arXiv DOI first)"],
                      "off": "second proposal call (Prompts.propose_domain empty)",
                      "decision_rule": DECISION_RULE, "decision_file": str(DECISION.relative_to(HERE))},
        "stages": {s: {"cap_usd": CAPS[s[0]], "steps": [{"cmd": " ".join(["python", *st["cmd"]]), "env": st["env"]} for st in steps_for(s, True)]}
                   for s in ("A", "B", "C1", "C2")},
        "stage_notes": {"B and C": "RRN_DATE_CANDIDATES is written by 'launch.py decide' from the stage A rows (shown here as 1)",
                        "C1": "the draw starts only after decide, once stage B is complete (or the group's decision to go on after a B that stopped early is recorded in out/launch_2026-11-14/stage_b_partial_accepted.json) and once out/launch_2026-11-14/predictions_v2.json records the version-2 predictions (its sha256 goes into heldout_draw.json); the draw runs once and C1 then pauses; C1 resumes only when out/launch_2026-11-14/heldout_pdfs_checked.json marks every sampled paper's PDF checked (swaps by the recipe's rule in papers_heldout_swaps.json), and then records the sample's hashes in heldout_sample.json; the pre-check scoring (results, summary, the judge's matches and its terminal output) and the tool's runs and Haiku-alone answers (blind, abl, replay) go to out_heldout/sealed_pre_check/, which is never overwritten, so the truth-check reviewer sees none of them; a complete C1 is never run again. The truth check follows by hand (protocol section 3, as for the development sets): heldout_truth_check.py packet writes each paper's prompt and list for Gemini in its chat window with the PDF, and review and apply turn the reviewed answers into the checked lists; no API call",
                        "C2": "starts only when C1 is complete, the sample still matches heldout_sample.json, and out/launch_2026-11-14/heldout_truth_checked.json (written by 'heldout_truth_check.py apply' after the reviewer's decisions) marks every sampled paper checked and still matches the truth lists' hash; it restores the sealed runs, the judge matches again on the checked lists, and a complete C2 is never run again"},
        "caps_usd": CAPS | {"total": TOTAL_CAP}, "stop_rules": ["stage spend >= stage cap - $0.50", "all stages >= $79 - $0.50", f"projected stage spend > cap once {PROJECT_AFTER:.0%} of its runs are done", "checked before and after every step and every 30 s during it"],
        "spend_measurement": {"anthropic": "sum of per-call estimates in the spend ledger (pricing.py prices); every Claude response that arrives, including refused, cut-off and malformed answers; a response lost to a network failure or to the runner stopping a step is not counted, which the $0.50 stop margin covers", "openalex": "rate-limit endpoint: daily_used_usd per day plus decreases of prepaid_remaining_usd; endpoint classes in the OpenAlex ledger"},
        "prices_usd_per_M_tokens": {m: pricing.PRICES[m] for m in (HAIKU, SONNET)},
        "outputs": {"stage A": "dating/, v2_rerank/, v2_offline_rows.json, v2_offline.md inside each as-run folder (new names only); out/launch_2026-11-14/report/",
                    "stage B": [v[0] for v in V2.values()], "stage C": HELDOUT[0], "ledgers, logs, status": str(LAUNCH.relative_to(HERE))},
        "heldout_recipe": json.loads(RECIPE.read_text()),
        "prompts": prompts(),
        "files": file_hashes(),
    }
    MANIFEST.write_text(json.dumps(man, indent=1, default=str))
    write_manifest_md(man)
    print(f"manifest written: {MANIFEST.relative_to(HERE)} ({MANIFEST.stat().st_size / 1e3:.0f} KB), sha256 {sha(MANIFEST)}")


def write_manifest_md(man: dict) -> None:
    """The readable companion of the manifest; the JSON is the binding record."""
    f = man["files"]
    L = ["# Launch manifest and configuration (14 November 2026)", "",
         f"Binding record: launch_manifest_2026-11-14.json, written by 'launch.py freeze' on {man['frozen']}. launch.py refuses to start a stage if any file below differs from it ('launch.py check' shows the comparison). "
         "This page summarizes it.", "",
         "## Approval and code state", "",
         f"- **Approval.** {man['approval']['by']}, {man['approval']['date']}: total ceiling ${man['approval']['total_ceiling_usd']:.0f} including retries.",
         f"- **Code.** Git commit {man['git']['commit'][:12]} plus the uncommitted working tree (diff sha256 {man['git']['diff_vs_commit_sha256'][:16]}... of the tracked files; experiments/ is untracked and hashed file by file below). Hashed below: the tool source, every script and paper list the stages run, the documents, the seed year cache, and the input folders. Not hashed: the OpenAlex and arXiv responses fetched live, the PDFs stage C downloads, and installed packages (versions recorded, not checked).",
         "- **Versions.** " + ", ".join(f"{k} {v}" for k, v in man["versions"].items()) + ".", "",
         "## Models and switches", "",
         "| Item | Value |", "| --- | --- |"]
    L += [f"| {k} | {v} |" for k, v in man["models"].items() if not isinstance(v, dict)]
    L += [f"| effort, {k} | {v} |" for k, v in man["models"]["efforts"].items()]
    L += [f"| max_tokens, {k} | {v} |" for k, v in man["models"]["max_tokens"].items()]
    L += [f"| {k} | {v} |" for k, v in man["tool_constants"].items()]
    L += ["", "Version 2, as launched:", ""] + [f"- {c}" for c in man["version_2"]["changes"]] + [f"- Off: {man['version_2']['off']}.", ""]
    r = man["version_2"]["decision_rule"]
    L += ["## Decision rule after stage A (fixed before stage A)", "",
          f"On the pooled post-cutoff sets ({', '.join(r['sets'])}), {r['test']}, {r['measure']}: compare '{r['v2']}' with '{r['base']}'. "
          f"Adopt the recency rule if the gain on benchmarks introduced in {' or '.join(map(str, r['recent_bins']))} is at least {r['recent_min_gain_points']:.0f} points and the change on those introduced {r['older_bin']} is no worse than -{r['older_max_loss_points']:.0f} points; "
          "otherwise version 2 is Sonnet ranking plus the arXiv bibliography, without dating or the rule. 'launch.py decide' computes this from stage A's rows and writes out/launch_2026-11-14/decision.json, bound to this manifest's hash; stages B and C refuse to start without it.", "",
          "## Stages, caps and stop rules", "",
          "| Stage | Cap | Steps |", "| --- | --- | --- |"]
    for s, v in man["stages"].items():
        L.append(f"| {s} | ${v['cap_usd']:.0f}{' (shared by C1 and C2)' if s.startswith('C') else ''} | " + "<br>".join(f"{i}. {st['cmd']}" + (f" [{st['env']['RRN_OUT']}]" if st.get("env", {}).get("RRN_OUT") else "") for i, st in enumerate(v["steps"], 1)) + " |")
    L += ["", "Environment of every step (ledger paths omitted): " + "; ".join(f"stage {s}: " + ", ".join(f"{k}={val}" for k, val in v["steps"][0]["env"].items() if not k.endswith("LEDGER") and k not in ("RRN_OUT", "RRN_PAPERS", "RRN_LAUNCH_REPORT")) for s, v in man["stages"].items()) + ". RRN_DATE_CANDIDATES in B and C is set from the decision. [folder] after a step is its RRN_OUT, the set it runs on. In B and C, 'benchmark_eval.py run' appears three times, each followed by 'redo', which deletes runs whose OpenAlex lookups failed so that the next pass retries them; 'leave_one_out.py run' also appears three times to retry failures. Finished runs are cached and never repeated, so a pass costs only what it retries.", "",
          "Stop rules: " + "; ".join(man["stop_rules"]) + f". Total ceiling ${man['caps_usd']['total']:.0f}. Spend: Claude from a per-call ledger at the pricing.py prices; OpenAlex from its rate-limit endpoint read every 30 s. A stopped stage keeps its results and is reported as partial.", "",
          "Notes: " + " ".join(f"{k}: {v}." for k, v in man["stage_notes"].items()), "",
          "## Held-out sample", "",
          f"Seed {man['heldout_recipe']['seed']}, window {man['heldout_recipe']['window']['from_publication_date']} to {man['heldout_recipe']['window']['to_publication_date']}, draw not before {man['heldout_recipe']['draw_not_before']}, {man['heldout_recipe']['exclusions']['count']} arXiv ids excluded. Full recipe: heldout_recipe_2026-11-14.md.", "",
          "## File hashes (sha256)", ""]
    for part in ("tool_source (src/researchreadernerd)", "scripts (experiments)", "paper lists (experiments)", "documents (report folder)"):
        L += [f"### {part}", "", "| File | sha256 |", "| --- | --- |"] + [f"| {k} | {v} |" for k, v in f[part].items()] + [""]
    L += ["### Inputs as folder hashes (files, sha256 over sorted paths and file hashes)", "", "| Set | Folder or pattern | Files | sha256 |", "| --- | --- | --- | --- |"]
    for part in ("stage B seed inputs (copied from the as-run folders)", "stage A saved pools (read, not changed)"):
        for d, items in f[part].items():
            for k, v in items.items():
                L.append(f"| {d} | {k} | {v['files'] if isinstance(v, dict) else 1} | {v['sha256'] if isinstance(v, dict) else v} |")
    L += ["", "## Prompts (sha256 of the exact text)", "", "| Prompt | Chars | In use | sha256 |", "| --- | --- | --- | --- |"]
    L += [f"| {k} | {v['chars']} | {'yes' if v['in_use'] else 'no (empty, off)'} | {v['sha256']} |" for k, v in man["prompts"].items()]
    (REPORT / "launch_manifest_2026-11-14.md").write_text("\n".join(L) + "\n")


def mismatches() -> list[str]:
    if not MANIFEST.exists():
        return ["no manifest: run 'launch.py freeze'"]
    want, now = json.loads(MANIFEST.read_text())["files"], file_hashes()
    bad = []

    def walk(a, b, path):
        if isinstance(a, dict):
            for k in a.keys() | (b or {}).keys():
                walk(a.get(k), (b or {}).get(k), f"{path}/{k}")
        elif a != b:
            bad.append(path.lstrip("/"))
    walk(want, now, "")
    return bad  # every hashed file, documents included: a changed protocol, plan, recipe or worksheet also blocks a run


# ---------- spend ----------

def openalex_state() -> dict | None:
    sys.path.insert(0, str(ROOT / "src"))
    from researchreadernerd.cli import load_dotenv
    import httpx2
    load_dotenv(ROOT / ".env")
    try:
        r = httpx2.get("https://api.openalex.org/rate-limit", params={"api_key": os.environ["OPENALEX_API_KEY"]}, timeout=20)
        return r.json()["rate_limit"]
    except Exception:  # noqa: BLE001
        return None


class Meter:
    """Spend of one stage: Claude from the ledger, OpenAlex from the rate-limit endpoint (daily use per day plus
    decreases of the prepaid balance; top-ups are not counted as negative spend)."""

    def __init__(self, stage: str):
        self.stage, self.f = stage, LAUNCH / f"meter_{stage}.json"
        self.s = json.loads(self.f.read_text()) if self.f.exists() else {"days": {}, "prepaid_spent": 0.0, "last_prepaid": None}
        self.lock = threading.Lock()

    def poll(self) -> None:
        """Read OpenAlex spend; raise when it cannot be read three times running, so that no step starts or keeps running
        on stale spend (the caller stops the stage)."""
        for attempt in range(3):
            r = openalex_state()
            if r:
                break
            time.sleep(10)
        else:
            raise RuntimeError("OpenAlex spend could not be read from the rate-limit endpoint")
        with self.lock:
            day = self.s["days"].setdefault(r["resets_at"], {"first": r["daily_used_usd"], "last": r["daily_used_usd"]})
            day["last"] = max(day["last"], r["daily_used_usd"])
            p = r.get("prepaid_remaining_usd") or 0.0
            if self.s["last_prepaid"] is not None and p < self.s["last_prepaid"]:
                self.s["prepaid_spent"] += self.s["last_prepaid"] - p
            self.s["last_prepaid"] = p
            self.s["funds_left"] = r["daily_remaining_usd"] + p
            self.f.write_text(json.dumps(self.s, indent=1))

    def openalex(self) -> float:
        return sum(d["last"] - d["first"] for d in self.s["days"].values()) + self.s["prepaid_spent"]

    def claude(self) -> float:
        f = LAUNCH / f"spend_{self.stage}.jsonl"
        # a model missing from the price table is charged at Sonnet's price, the highest in use
        return sum(x["usd"] if x.get("usd") is not None else (2.0 * x.get("input_tokens", 0) + 10.0 * x.get("output_tokens", 0)) / 1e6
                   for x in read_jsonl(f))

    def total(self) -> float:
        return self.claude() + self.openalex()


PARTS = {"A": ("A",), "B": ("B",), "C": ("C1", "C2")}  # stage C's cap is shared by its two parts; each part has its own meter


def part_spend(part: str) -> float:
    return Meter(part).total() if (LAUNCH / f"meter_{part}.json").exists() or (LAUNCH / f"spend_{part}.jsonl").exists() else 0.0


def read_jsonl(f: Path) -> list[dict]:
    """A ledger's lines; a last line still being written by a step is skipped until it is complete."""
    rows = []
    for x in (f.read_text().splitlines() if f.exists() else []):
        try:
            rows.append(json.loads(x))
        except json.JSONDecodeError:
            continue
    return rows


def all_stages_spend() -> float:
    return sum(part_spend(p) for parts in PARTS.values() for p in parts)


def _expected(expected) -> int:
    if isinstance(expected, int):
        return expected
    if expected.startswith("loo:"):  # leave_one_out.py's jobs: one per public benchmark of the current truth list of a readable paper
        _, out, papers = expected.split(":")
        n = 0
        for p in json.loads((HERE / papers).read_text()) if (HERE / papers).exists() else []:
            ex, tf = HERE / out / "extract" / f"{p['id']}.{HAIKU}.json", HERE / out / "truth" / f"{p['id']}.json"
            if ex.exists() and "error" not in json.loads(ex.read_text()) and tf.exists():
                n += sum(1 for b in json.loads(tf.read_text())["benchmarks"] if b.get("kind") == "public")
        return n
    return len(list(HERE.glob(expected)))


def progress(steps: list[dict]) -> float | None:
    """Share of the whole stage's runs done: output files of every run-producing step against the expected counts
    (each distinct step counted once; the retried attempts of a step share its count)."""
    specs = list(dict.fromkeys(st["progress"] for st in steps if st.get("progress")))
    n = sum(_expected(e) for _, e in specs)
    return sum(len(list(HERE.glob(g))) for g, _ in specs) / n if n else None


# ---------- running ----------

def log(stage: str, msg: str) -> None:
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    with open(LAUNCH / f"log_{stage}.txt", "a") as f:
        f.write(line + "\n")


def state_of(stage: str) -> str | None:
    f = LAUNCH / f"status_{stage}.json"
    return json.loads(f.read_text()).get("state") if f.exists() else None


def all_checked(path: Path, what: str) -> str | None:
    """None when `path` marks every paper of papers_heldout.json as checked: {"papers": {"<arXiv id>": {"checked": true, ...}}}."""
    if not path.exists():
        return f"REFUSED: {what} not recorded ({path.relative_to(HERE)})"
    marks = json.loads(path.read_text()).get("papers", {})
    ids = [p["arxiv"] for p in json.loads((HERE / HELDOUT[1]).read_text())]
    missing = [i for i in ids if not (marks.get(i) or {}).get("checked")]
    return f"REFUSED: {what}: {len(missing)} of {len(ids)} papers not marked checked in {path.name}: {', '.join(missing[:10])}" if missing else None


def over_cap(meter, part: str, steps: list[dict]) -> str | None:
    """Why the stage must stop, or None. Stage spend sums the stage's parts; the projection scales only this part's spend
    by this part's progress, the other parts' spend being fixed (C1's spend is not projected over C2's runs)."""
    cap = CAPS[part[0]]
    mine = meter.total()
    others = sum(part_spend(p) for p in PARTS[part[0]] if p != part)
    spent, everything = mine + others, all_stages_spend() - part_spend(part) + mine
    frac = progress(steps)
    projected = others + mine / frac if frac else None
    return (f"stage spend ${spent:.2f} reached the ${cap:.0f} cap less ${STOP_MARGIN:.2f}" if spent >= cap - STOP_MARGIN
            else f"all stages ${everything:.2f} reached ${TOTAL_CAP:.0f} less ${STOP_MARGIN:.2f}" if everything >= TOTAL_CAP - STOP_MARGIN
            else f"projected ${projected:.2f} over the ${cap:.0f} cap at {frac:.0%} of this part's runs" if frac and frac >= PROJECT_AFTER and projected > cap
            else None)


def run_stage(stage: str) -> int:
    meter_key = stage  # the part: A, B, C1 or C2
    LAUNCH.mkdir(parents=True, exist_ok=True)
    bad = mismatches()
    if bad:
        print("REFUSED: files differ from the launch manifest:\n  " + "\n  ".join(bad[:40])); return 2
    dc = None
    if stage != "A":
        if not DECISION.exists():
            print("REFUSED: no decision yet; run stage A, then 'launch.py decide'"); return 2
        d = json.loads(DECISION.read_text())
        if d.get("manifest_sha256") != sha(MANIFEST):
            print("REFUSED: the decision was recorded against another manifest"); return 2
        dc = d["date_candidates"]
    # The draw runs once: when papers_heldout.json exists (a resumed C1), the draw step is skipped and the saved sample used.
    if stage == "C1" and not (HERE / HELDOUT[1]).exists() and datetime.now().date().isoformat() < json.loads(RECIPE.read_text())["draw_not_before"]:
        print("REFUSED: before the recipe's draw_not_before date"); return 2
    if stage == "C1" and state_of("C1") == "complete":
        print("REFUSED: C1 is complete; running it again would disturb the sealed pre-check scoring"); return 2
    if stage == "C1" and not (HERE / HELDOUT[1]).exists():
        if state_of("B") != "complete" and not B_PARTIAL_OK.exists():
            print(f"REFUSED: stage B is not complete (state: {state_of('B') or 'not run'}); the draw waits for B, or for the group's decision "
                  f"to go on recorded in {B_PARTIAL_OK.relative_to(HERE)}"); return 2
        if not PREDICTIONS.exists():
            print(f"REFUSED: version-2 predictions not recorded ({PREDICTIONS.relative_to(HERE)}; protocol sections 6 and 8)"); return 2
    if stage == "C1" and (HERE / HELDOUT[1]).exists() and (why := all_checked(PDFS_CHECKED, "the PDF check after the draw")):
        print(why); return 2
    if stage == "C1" and (HERE / HELDOUT[1]).exists() and not (LAUNCH / "heldout_sample.json").exists():
        swaps = HERE / "papers_heldout_swaps.json"
        (LAUNCH / "heldout_sample.json").write_text(json.dumps({  # the sample as run, after any swap; recorded once, before any run on it
            "papers_heldout.json": sha(HERE / HELDOUT[1]), "papers_heldout_swaps.json": sha(swaps) if swaps.exists() else None,
            "heldout_pdfs_checked.json": sha(PDFS_CHECKED), "predictions_v2.json": sha(PREDICTIONS),
            "recorded": datetime.now(timezone.utc).isoformat(timespec="seconds")}, indent=1))
    if stage in ("C1", "C2") and (LAUNCH / "heldout_sample.json").exists() and \
            json.loads((LAUNCH / "heldout_sample.json").read_text())["papers_heldout.json"] != sha(HERE / HELDOUT[1]):
        print("REFUSED: papers_heldout.json differs from the sample recorded in heldout_sample.json"); return 2
    if stage == "C2" and state_of("C1") != "complete":
        print(f"REFUSED: C1 is not complete (state: {state_of('C1') or 'not run'})"); return 2
    if stage == "C2" and (why := all_checked(TRUTH_CHECKED, "the held-out truth check (protocol section 3)")):
        print(why); return 2
    if stage == "C2" and state_of("C2") == "complete":
        print("REFUSED: C2 is complete; the held-out results are never re-scored"); return 2
    if stage == "C2" and json.loads(TRUTH_CHECKED.read_text()).get("truth_sha256") != tree_sha(HERE / HELDOUT[0] / "truth")["sha256"]:
        print("REFUSED: out_heldout/truth differs from the lists recorded by 'heldout_truth_check.py apply'"); return 2
    if stage == "C2":  # the tool's runs and Haiku-alone answers, sealed in C1 so that the reviewer did not see them, come back for scoring
        sealed = HERE / HELDOUT[0] / "sealed_pre_check"
        for d in SEALED_RUNS:
            if (sealed / d).exists() and not (HERE / HELDOUT[0] / d).exists():
                shutil.copytree(sealed / d, HERE / HELDOUT[0] / d)
    if stage == "B":
        seed_stage_b()
    meter = Meter(meter_key)
    try:
        meter.poll()
    except RuntimeError as e:
        print(f"REFUSED: {e}"); return 2
    steps = steps_for(stage, dc)
    log(stage, f"stage {stage} start; manifest sha256 {sha(MANIFEST)}; cap ${CAPS[stage[0]]:.0f}; spent so far in this part ${meter.total():.2f}, all stages ${all_stages_spend():.2f}")
    for i, st in enumerate(steps, 1):
        if st.get("once") and (HERE / st["once"]).exists():
            continue
        try:
            while True:  # OpenAlex money for the step: daily budget or prepaid
                meter.poll()
                if meter.s.get("funds_left", 0) >= MIN_OPENALEX_FUNDS:
                    break
                log(stage, f"OpenAlex funds ${meter.s.get('funds_left', 0):.2f}, waiting 15 min"); time.sleep(900)
        except RuntimeError as e:
            log(stage, f"STOPPED before step {i}: {e}. Partial results kept; stage {stage} is reported as partial.")
            write_status(stage, meter, f"stopped: {e}"); return 3
        if why := over_cap(meter, meter_key, steps):
            log(stage, f"STOPPED before step {i}: {why}. Partial results kept; stage {stage} is reported as partial.")
            write_status(stage, meter, "stopped: " + why); return 3
        env = os.environ.copy() | st["env"]
        log(stage, f"step {i}/{len(steps)}: python {' '.join(st['cmd'])}  [{', '.join(f'{k}={v}' for k, v in st['env'].items() if not k.endswith('LEDGER'))}]")
        out = None
        if st.get("seal"):  # pre-check scores are written to the sealed folder, never to the terminal
            sealed = HERE / HELDOUT[0] / "sealed_pre_check"
            if (sealed / "results.json").exists():
                log(stage, f"skip step {i}: the pre-check scoring is already sealed in {sealed.relative_to(HERE)} (never overwritten)")
                continue
            sealed.mkdir(parents=True, exist_ok=True)
            out = open(sealed / f"score_stdout_step{i}.txt", "w")
        proc = subprocess.Popen([PY, *st["cmd"]], cwd=HERE, env=env, stdout=out, stderr=subprocess.STDOUT if out else None)
        reason = []

        def watch():
            while proc.poll() is None:
                time.sleep(30)
                if proc.poll() is not None:
                    return
                try:
                    meter.poll()
                    why = over_cap(meter, meter_key, steps)
                except Exception as e:  # noqa: BLE001  fail closed: a step is never left running unmetered
                    why = f"spend could not be read ({type(e).__name__}: {e})"
                if why:
                    reason.append(why); proc.terminate(); return
        t = threading.Thread(target=watch, daemon=True); t.start()
        rc = proc.wait(); t.join(timeout=1)
        try:
            meter.poll()
        except RuntimeError as e:
            reason.append(str(e))
        if out:
            out.close()
        if not reason and (why := over_cap(meter, meter_key, steps)):
            reason.append(why)
        log(stage, f"  exit {rc}; stage spend ${meter.total():.2f} (Claude ${meter.claude():.2f}, OpenAlex ${meter.openalex():.2f})")
        if st.get("once"):
            resp = HERE / HELDOUT[1].replace("papers", "draw_response")
            if resp.exists():
                (LAUNCH / "heldout_draw.json").write_text(json.dumps({"file": resp.name, "sha256": sha(resp), "drawn": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                                                      "papers_heldout_sha256": sha(HERE / HELDOUT[1]) if (HERE / HELDOUT[1]).exists() else None,
                                                                      "predictions_sha256": sha(PREDICTIONS), "manifest_sha256": sha(MANIFEST)}, indent=1))
            if rc == 0 and not reason:
                log(stage, f"PAUSED after the draw: check every PDF, swap any unreadable one by the recipe's rule (papers_heldout_swaps.json), "
                           f"record {PDFS_CHECKED.relative_to(HERE)}, then run C1 again")
                write_status(stage, meter, "paused after the draw for the PDF check"); return 0
        if st.get("seal") and rc == 0:
            sealed = HERE / HELDOUT[0] / "sealed_pre_check"; sealed.mkdir(exist_ok=True)
            for f in ("results.json", "summary.md", "match", *SEALED_RUNS):  # match/: the judge's pre-check matches, re-judged in C2; the runs come back in C2
                if (HERE / HELDOUT[0] / f).exists():
                    shutil.move(str(HERE / HELDOUT[0] / f), sealed / f)
        if reason:
            log(stage, f"STOPPED: {reason[0]}. Partial results kept; stage {stage} is reported as partial.")
            write_status(stage, meter, "stopped: " + reason[0]); return 3
        if rc not in st.get("ok_codes", (0,)):
            log(stage, f"STOPPED: step exited {rc}"); write_status(stage, meter, f"step {i} exited {rc}"); return rc
    write_status(stage, meter, "complete")
    log(stage, f"stage {stage} complete; spend ${meter.total():.2f}")
    return 0


def write_status(stage: str, meter: Meter, state: str) -> None:
    oa = Counter_from(LAUNCH / f"openalex_{stage}.jsonl", "class")
    cl = Counter_from(LAUNCH / f"spend_{stage}.jsonl", "model")
    (LAUNCH / f"status_{stage}.json").write_text(json.dumps({"stage": stage, "state": state, "claude_usd": round(meter.claude(), 4),
                                                              "openalex_usd": round(meter.openalex(), 4), "cap_usd": CAPS[stage[0]], "stage_spend_usd": round(sum(part_spend(p) for p in PARTS[stage[0]]), 4),
                                                              "claude_calls_by_model": cl, "openalex_calls_by_class": oa,
                                                              "manifest_sha256": sha(MANIFEST)}, indent=1))


def Counter_from(f: Path, key: str) -> dict:
    from collections import Counter
    return dict(Counter(x.get(key) for x in read_jsonl(f)))


def seed_stage_b() -> None:
    """Copy the tool-independent inputs into the stage B folders. Folders are copied over at every start of B (the hashed
    as-run copy replaces a same-named file); the pattern files and single files are copied only where missing."""
    for label, (new, papers) in V2.items():
        src, dst = HERE / AS_RUN[label][0], HERE / new
        for sub in SEED_DIRS:
            if (src / sub).exists():
                shutil.copytree(src / sub, dst / sub, dirs_exist_ok=True)
        for sub, pat in SEED_GLOBS:
            (dst / sub).mkdir(parents=True, exist_ok=True)
            for f in (src / sub).glob(pat):
                if not (dst / sub / f.name).exists():
                    shutil.copy2(f, dst / sub / f.name)
        for f in SEED_FILES:
            if (src / f).exists() and not (dst / f).exists():
                shutil.copy2(src / f, dst / f)


def cmd_decide() -> int:
    """Apply DECISION_RULE to stage A's rows and record the version-2 configuration for stages B and C."""
    bad = mismatches()
    if bad:
        print("REFUSED: files differ from the launch manifest:\n  " + "\n  ".join(bad[:40])); return 2
    if state_of("A") != "complete":
        print(f"REFUSED: stage A is not complete (state: {state_of('A') or 'not run'})"); return 2
    rows = [r for s in DECISION_RULE["sets"] for r in json.loads((HERE / s / "v2_offline_rows.json").read_text()) if r["test"] == DECISION_RULE["test"]]
    rec = [r for r in rows if r["year"] in DECISION_RULE["recent_bins"]]; old = [r for r in rows if r["year"] and r["year"] <= 2024]
    pct = lambda sel, k: 100 * sum(1 for r in sel if r[k]) / len(sel)  # noqa: E731
    gain_recent = pct(rec, DECISION_RULE["v2"]) - pct(rec, DECISION_RULE["base"])
    change_old = pct(old, DECISION_RULE["v2"]) - pct(old, DECISION_RULE["base"])
    adopt = gain_recent >= DECISION_RULE["recent_min_gain_points"] and change_old >= -DECISION_RULE["older_max_loss_points"]
    d = {"decided": datetime.now().isoformat(timespec="seconds"), "rule": DECISION_RULE, "n_recent": len(rec), "n_older": len(old),
         "gain_recent_points": round(gain_recent, 1), "change_older_points": round(change_old, 1), "adopt_recency_rule": adopt,
         "date_candidates": adopt, "version_2": "Sonnet ranking + dating with the recency rule + arXiv bibliography" if adopt
         else "Sonnet ranking + arXiv bibliography, no dating or recency rule", "manifest_sha256": sha(MANIFEST)}
    DECISION.parent.mkdir(parents=True, exist_ok=True); DECISION.write_text(json.dumps(d, indent=1))
    print(json.dumps(d, indent=1)); return 0


def cmd_check(stage: str | None) -> int:
    bad = mismatches()
    print("manifest:", "all files match" if not bad else f"{len(bad)} differ:\n  " + "\n  ".join(bad[:40]))
    r = openalex_state()
    print("OpenAlex:", f"daily ${r['daily_remaining_usd']:.2f} left of ${r['daily_budget_usd']}, prepaid ${r.get('prepaid_remaining_usd') or 0:.2f}" if r else "rate-limit endpoint not readable")
    print("Anthropic key:", "set" if (ROOT / ".env").exists() and "ANTHROPIC_API_KEY" in (ROOT / ".env").read_text() else "missing")
    print("decision:", json.loads(DECISION.read_text())["version_2"] if DECISION.exists() else "not yet (after stage A)")
    for s, parts in PARTS.items():
        if any((LAUNCH / f"meter_{p}.json").exists() or (LAUNCH / f"spend_{p}.jsonl").exists() for p in parts):
            print(f"stage {s}: ${sum(part_spend(p) for p in parts):.2f} of ${CAPS[s]:.0f}")
    if stage:
        for i, st in enumerate(steps_for(stage, json.loads(DECISION.read_text())["date_candidates"] if DECISION.exists() else True), 1):
            print(f"  {i}. python {' '.join(st['cmd'])}")
    return 1 if bad else 0


def main() -> int:
    a = sys.argv[1:]
    if not a:
        print(__doc__); return 0
    if a[0] == "freeze":
        cmd_freeze(replace="--replace" in a); return 0
    if a[0] == "check":
        return cmd_check(a[1] if len(a) > 1 else None)
    if a[0] == "decide":
        return cmd_decide()
    if a[0] == "run" and len(a) > 1 and a[1] in ("A", "B", "C1", "C2"):
        return run_stage(a[1])
    print(__doc__); return 2


if __name__ == "__main__":
    sys.exit(main())
