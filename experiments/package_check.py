"""Reviewer-style consistency check of a built launch bundle (24 Nov 2026): the shipped documents must agree with the
manifest and with each other, and must not carry stale claims from earlier packages. Run after launch_bundle.py;
exit 1 lists every problem. No provider calls.

  python package_check.py [path to the unzipped launch_2026-11-24 folder]   (default: the built folder)
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEST = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "out" / "report_2026-11-08" / "launch_2026-11-24"
REP = DEST / "ResearchReaderNerd" / "experiments" / "out" / "report_2026-11-08"

# Phrases that were true of an earlier package and must not appear in this one's prose.
STALE = [
    ("uncommitted working tree", "the manifest records a clean tree"),
    ("is untracked", "experiments/ is committed"),
    ("experiments/ was untracked", None),  # allowed only in the diff note (checked below)
    ("is not counted, which the $0.50", "lost responses are now counted at their worst case"),
    ("which the $0.50 margin covers", "lost responses are now counted at their worst case"),
    ("read every 30 s. A stopped", "spend text predates the free-only guard"),
    ("for_harsh_v7", "an archived package"),
    ("launch_2026-11-14", "the superseded launch folder"),
    ("Launch manifest and configuration (14 November 2026)", "old title"),
    ("about $22 of OpenAlex credit", "paid OpenAlex is no longer used"),
    ("If OpenAlex bills title lookups at its search price", "paid OpenAlex is no longer used"),
]


def main() -> int:
    problems: list[str] = []
    man = json.loads((REP / "launch_manifest_2026-11-24.json").read_text())
    cost = json.loads((REP / "launch_cost_summary_2026-11-24.json").read_text())
    docs = {"README.md": DEST / "README.md", "INVENTORY.md": DEST / "INVENTORY.md",
            "launch_manifest_2026-11-24.md": REP / "launch_manifest_2026-11-24.md", "protocol_2026-11-24.md": REP / "protocol_2026-11-24.md",
            "launch_cost_worksheet_2026-11-24.md": REP / "launch_cost_worksheet_2026-11-24.md"}
    text = {k: p.read_text() for k, p in docs.items()}

    for name, t in text.items():
        for phrase, why in STALE:
            if why and phrase in t:
                problems.append(f"{name}: stale phrase '{phrase}' ({why})")
    if "experiments/ was untracked" in text["README.md"] and "the experiment scripts were untracked then" not in text["README.md"]:
        problems.append("README.md: 'untracked' outside the diff note")

    g = man["git"]
    if g["working_tree_uncommitted"] is not False:
        problems.append(f"manifest: working tree not recorded as clean ({g['working_tree_uncommitted']})")
    if g["commit"][:7] not in text["README.md"]:
        problems.append(f"README.md does not name the manifest's commit {g['commit'][:7]}")
    if "clean working tree" not in text["launch_manifest_2026-11-24.md"]:
        problems.append("manifest .md does not state the clean working tree")
    if man.get("package_stages") != ["A", "B"] or "approves stages A, B only" not in text["launch_manifest_2026-11-24.md"]:
        problems.append("manifest does not state that it approves stages A and B only")
    if not any(c.startswith("4. hybrid") for c in man["version_2"]["changes"]):
        problems.append("manifest version_2 changes omit the hybrid sources")
    if "lost" in man["spend_measurement"]["anthropic"] and "not counted" in man["spend_measurement"]["anthropic"]:
        problems.append("manifest spend_measurement still says lost responses are not counted")
    for st in ("A", "B"):
        figure = f"${cost['claude_with_retries'][st]:.2f}"
        if figure not in text["README.md"]:
            problems.append(f"README.md lacks stage {st}'s Claude figure {figure} from the cost summary")
    if not cost.get("fallback_rate_measured"):
        problems.append("cost summary: the fallback rate is a bound, not the measured rate")
    if "protocol_2026-11-24" not in text["README.md"]:
        problems.append("README.md does not point to protocol version 2")

    # protocol 5a must be the amendment Harsh agreed, word for word
    prot, amend = text["protocol_2026-11-24.md"], (REP / "protocol_amendment_stageC_2026-11-16.md").read_text()
    a5 = re.sub(r"\s+", " ", amend[amend.index("## 5a."):].replace("## Justification", "### Justification of 5a")).strip()
    p5 = re.sub(r"\s+", " ", prot[prot.index("## 5a."):prot.index("## 6. Held-out rule")]).strip()
    if a5 != p5:
        problems.append("protocol section 5a differs from the agreed amendment")
    for phrase in ("Ranking call by claude-sonnet-5-5", "Every search goes to Semantic Scholar"):
        if phrase not in prot:
            problems.append(f"protocol lacks '{phrase}'")
    # the older OpenAlex sentences carry the version-2 note
    for line in prot.splitlines():
        if re.search(r"reference list (from|fetched from) OpenAlex|OpenAlex confirms it|looks up by title, in OpenAlex", line) and "change 4" not in line:
            problems.append(f"protocol: an OpenAlex sentence lacks the version-2 note: {line[:90]}")

    # every file the README names is in the bundle
    for f in sorted(set(re.findall(r"[\w.-]+\.(?:json|md|csv|diff)", text["README.md"]))):
        if f in ("README.md",) or f.endswith("_oa_cache.zip"):
            continue
        if not list(DEST.rglob(f)):
            problems.append(f"README.md names {f}, which is not in the bundle")

    print("\n".join(problems) if problems else f"package check: no problems ({len(text)} documents, manifest commit {g['commit'][:7]})")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
