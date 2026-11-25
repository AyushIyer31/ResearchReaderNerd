"""Bundle the version-2 launch package (stages A and B, hybrid sources) for Harsh and Prof. Shasha, with the hashed code
and its diff against commit 0ab6d66 (the code Harsh reviewed in PR #1). No provider calls.

  python launch_bundle.py -> out/report_2026-11-08/launch_2026-11-24/ and launch_2026-11-24.zip
"""
from __future__ import annotations

import difflib
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import launch  # noqa: E402

REPORT = HERE / "out" / "report_2026-11-08"
DEST = REPORT / "launch_2026-11-24"
CACHE_DEST = REPORT / "launch_2026-11-24_oa_cache"
BASE = "0ab6d66"  # last commit before version 2; experiments/ was untracked then, so its scripts appear whole


def unified(old: str, new: Path, a: str, b: str) -> list[str]:
    return list(difflib.unified_diff(old.splitlines(keepends=True), new.read_text().splitlines(keepends=True), a, b))


def at_base(path: str) -> str:
    p = subprocess.run(["git", "show", f"{BASE}:{path}"], cwd=HERE.parent, capture_output=True, text=True)
    return p.stdout if p.returncode == 0 else ""


def copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> None:
    bad = launch.mismatches()
    if bad:
        sys.exit("files differ from the launch manifest; run launch.py freeze first:\n  " + "\n  ".join(bad))
    for d in (DEST, CACHE_DEST):
        if d.exists():
            shutil.rmtree(d)
    T = DEST / "ResearchReaderNerd"  # the repository's own layout, so that 'python experiments/launch.py check' runs inside it
    root, src = HERE.parent, HERE.parent / "src" / "researchreadernerd"
    inventory = []  # (manifest item, where it is)
    for f in launch.TOOL_FILES:
        copy(src / f, T / "src" / "researchreadernerd" / f); inventory.append((f"src/researchreadernerd/{f}", "shipped"))
    for f in sorted((root / "src" / "researchreadernerd").rglob("*")):  # non-Python package files (static/) for a working install
        if f.is_file() and f.suffix != ".py" and "__pycache__" not in f.parts:
            copy(f, T / f.relative_to(root))
    copy(root / "pyproject.toml", T / "pyproject.toml"); inventory.append(("pyproject.toml", "shipped"))
    for f in sorted((root / "tests").glob("*.py")):
        copy(f, T / "tests" / f.name)
    for f in launch.SCRIPTS + launch.INPUT_FILES + ["launch_bundle.py"]:
        copy(HERE / f, T / "experiments" / f); inventory.append((f"experiments/{f}", "shipped"))
    for f in launch.DOCS + ["launch_manifest_2026-11-24.json", "launch_manifest_2026-11-24.md"]:
        copy(REPORT / f, T / "experiments" / "out" / "report_2026-11-08" / f)
        inventory.append((f"experiments/out/report_2026-11-08/{f}", "shipped"))
    copy(REPORT / "hand_check" / "decisions_batch6.json", DEST / "decisions" / "truth_decisions_2026-11-14.json")
    import fnmatch
    for d, _ in launch.AS_RUN.values():
        base = HERE / d
        items = [(sub, "*") for sub in launch.SEED_DIRS] + list(launch.SEED_GLOBS) + [("blind", f"*.{launch.HAIKU}.*+model.json"), ("loo", "*"),
                                                                               (f"rerank_{launch.SONNET}", "*")]
        for sub, pat in items:
            dest_root = (CACHE_DEST / "ResearchReaderNerd" if sub == "oa_cache" else T) / "experiments" / d / sub
            for f in sorted((base / sub).rglob("*")):
                if f.is_file() and fnmatch.fnmatch(f.relative_to(base / sub).as_posix(), pat):
                    copy(f, dest_root / f.relative_to(base / sub))
            where = f"not in this archive: {CACHE_DEST.name}.zip (unzip next to it)" if sub == "oa_cache" else "shipped"
            inventory.append((f"experiments/{d}/{sub}" + ("" if pat == "*" else f"/{pat}"), where))
        for f in launch.SEED_FILES + ["results.json", "loo_results.json"]:
            if (base / f).exists():
                copy(base / f, T / "experiments" / d / f); inventory.append((f"experiments/{d}/{f}", "shipped"))
    diff = []
    for f in launch.TOOL_FILES:
        diff += unified(at_base(f"src/researchreadernerd/{f}"), src / f, f"{BASE}/src/researchreadernerd/{f}", f"launch/src/researchreadernerd/{f}")
    (DEST / f"changes_since_{BASE}.diff").write_text("".join(diff))
    (DEST / "INVENTORY.md").write_text("# Every item the manifest hashes, and where it is\n\n" + "\n".join(f"- {a}: {b}" for a, b in inventory) + "\n")
    man = json.loads((REPORT / "launch_manifest_2026-11-24.json").read_text())
    cost = json.loads((REPORT / "launch_cost_summary_2026-11-24.json").read_text())
    fb = f"{cost['fallback_rate']:.0%}" + (" (measured in the live check)" if cost["fallback_rate_measured"] else " (an upper bound)")
    (DEST / "README.md").write_text(f"""# Version-2 launch package, stages A and B (24 Nov 2026; revised after Harsh's review)

1. **Launch manifest and configuration with hashes:** ResearchReaderNerd/experiments/out/report_2026-11-08/launch_manifest_2026-11-24.json (binding; frozen {man['frozen']}; sha256 in SHA256SUMS) and its .md. It approves stages A and B only; stage C is frozen separately before its draw.
2. **Protocol, version 2:** protocol_2026-11-24.md (same folder): section 5a, the stage C criterion agreed on 17 Nov; section 8, change 4, the hybrid sources; section 7, free scholarly data.
3. **Cost and time worksheet:** launch_cost_worksheet_2026-11-24.md and .csv (same folder). Paid money is Claude only: ${cost['claude_with_retries']['A']:.2f} for stage A and ${cost['claude_with_retries']['B']:.2f} for stage B with retries, under the caps (A $21, B $43). OpenAlex fallback share of title lookups: {fb}.

Nothing has run. Every stage runs only through experiments/launch.py, and only after your written go-ahead.

## Verifying the package

- ResearchReaderNerd/ has the repository's layout. Its code, tests, scripts, paper lists and protocol are those of public commit {man['git']['commit'][:7]} ({'a clean working tree' if not man['git']['working_tree_uncommitted'] else 'with uncommitted changes recorded in the manifest'}); the run data and input folders are not in git and are identified by their hashes in the manifest, as is every file here.
- INVENTORY.md lists every item the manifest hashes and where it is. All are in this archive except the three OpenAlex response caches (oa_cache, about 350 MB), which are in {CACHE_DEST.name}.zip: unzip both into the same folder.
- To check every hash: `cd ResearchReaderNerd && python experiments/launch.py check` prints "all files match" (with the caches unzipped; without them, only the three oa_cache entries differ). It needs no key; the OpenAlex line says the endpoint is not readable without one.
- To run the tests: `pip install -e . && pytest` in ResearchReaderNerd/ (no network, no key).

## Changes after Harsh's review (24 Nov)

1. **OpenAlex cannot be billed beyond its free budget.** The launch key holds no prepaid credit, so OpenAlex refuses rather than charges once the free $1 a day is spent; the runner refuses to start with a key that holds any. Each priced request is guarded in scholar.py (RRN_OPENALEX_FREE_ONLY): it is sent only if the free budget, re-read every 20 s or after 2 cents, less this process's spend since, covers it with a 2-cent margin. After every step and at the end of a stage the runner checks the prepaid balance; a stage that saw any prepaid charge is stopped, never complete. Tests: tests/test_launch_guards.py and the dry run.
2. **The Claude cap is enforced before calls.** Before each call, llm.py counts its input tokens (Anthropic's free counting endpoint, plus an allowance for the output schema and 10%) and reserves its worst case, that input plus a full max_tokens of output at the model's card, in the stage's ledger; it refuses the call if spend plus open reservations plus this reserve would pass the cap the runner passes to the step (the stage cap and the $79 total, each less $0.50). A reservation counts in full until its response arrives, so a call in flight or a lost response is counted at its worst case; the SDK's own retries are off, so every attempt is reserved. Bound: spend at most equals the cap, since every billed call was reserved at an amount no lower than its cost before it was sent. If the input cannot be counted, the call is not sent.
3. **The bundle is complete and verifiable** (above), with the tests, and the repository commit identifies it.
4. **Stage A's two rankings use separate prompts with recorded provenance.** The baseline ('Sonnet, no years') uses BASE_RANK_PROMPT, the as-run prompt without the recency sentence (sha256 c6d4434f...); the v2 rerank uses it with the sentence. The 342 cached baseline reranks were made by sonnet_rerank.py, whose manifest records that exact prompt; stage A refuses to run if it differs. Every new rerank records its prompt's sha256, and a cached rerank from the other prompt is refused.
5. **Funding:** Claude calls are covered by the existing Anthropic account; no other funding is needed for stages A and B.

## Sources (protocol section 8, change 4)

- Semantic Scholar (free, keyed) for every search: title match, topic search, a paper's reference list.
- OpenAlex for records by id (free), works in batches and co-usage (list queries), and the title-search fallback when an id resolves to a record with another title (10% of lookups in the live check). Free daily budget only.
- Every candidate is still an OpenAlex record, so co-usage, the caches and the leave-one-out withholding work as before.

## How the runner runs and stops

- Order: stage A; `decide` (needs a complete A; records whether version 2 includes dating and the recency rule, bound to the manifest's hash); stage B. Stage C is refused by this manifest.
- Every run and decide refuses if any hashed file differs from the manifest; freeze never overwrites a manifest; every status file, log and decision carries the manifest's sha256.
- Besides the pre-call cap, a stage stops when its spend projected from the share of its runs done exceeds the cap after 10%, or when OpenAlex's rate-limit endpoint cannot be read. A stopped stage keeps its results and is reported as partial.
- Retries: a step that stops at a failed call runs up to three times, each pass skipping what is cached.

## Truth lists

Settled on 14 Nov by the 12 Nov availability rule (decisions/truth_decisions_2026-11-14.json): EmulatorBench is own; PMPP-Hard, MazeBench and the Mental Health Multimodal tabular set stay public. Stage B has 275 leave-one-out runs.

## Known limits

- Semantic Scholar currently answers about one call in 13 s even with the key (live check of 24 Nov): stage A about 7 hours, stage B 1-2 days. The runner is unaffected by the rate; only the duration changes.
- A failed arXiv fetch leaves that run without the paper's own bibliography; failures are counted in the ledger and reported.
- Version 1's development results came from OpenAlex-only retrieval; stage B measures version 2 with the hybrid sources on the same papers.
""")
    import hashlib
    sums = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(DEST)}" for p in sorted(DEST.rglob("*")) if p.is_file()]
    (DEST / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    z = shutil.make_archive(str(DEST), "zip", root_dir=REPORT, base_dir=DEST.name)
    zc = shutil.make_archive(str(CACHE_DEST), "zip", root_dir=REPORT, base_dir=CACHE_DEST.name)
    print(z, f"{Path(z).stat().st_size / 1e6:.1f} MB", f"{len(sums)} files")
    print(zc, f"{Path(zc).stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
