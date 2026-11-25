"""Truth-list check for the held-out sample (protocol section 3), by the procedure used for the development sets on
11 November: a checking model, Gemini in its chat window, reads each paper's PDF against the list, and a reviewer applies
only verdicts with a quoted location. No API call; the runner does not run this script.

  python heldout_truth_check.py packet   after C1: writes out_heldout/truth_check_packet/, one file per sampled paper
                                         with the prompt, the required JSON answer format and the paper's pre-check
                                         list, to paste into Gemini with the paper's PDF (the PDFs are in pdfs/).
                                         Save each answer, the JSON only, as out_heldout/truth_check_answers/<arXiv id>.json.
  python heldout_truth_check.py review   validates every answer against the format, then writes
                                         out_heldout/truth_check_review.md and the decisions file
                                         out_heldout/truth_check_decisions.json for the reviewer to fill in.
  python heldout_truth_check.py apply    applies the reviewed decisions to out_heldout/truth/ (the pre-check lists are
                                         kept in out_heldout/truth_pre_check/), gives every entry a hand_check record,
                                         and writes out/launch_2026-11-14/heldout_truth_checked.json, which C2 requires,
                                         with the hash of the checked lists. Refused once C2 has started.

Rules (protocol section 3):
- Only verdicts with a quoted location can be applied, and the reviewer decides each non-keep verdict against the PDF; a
  verdict without a location is recorded and not applied; 'unclear' leaves the entry as it is.
- Availability (12 Nov rule): a public entry needs a source paper that releases the benchmark or a dated release before
  the paper. Where the PDF shows neither, the reviewer resolves it: public (with the evidence found), own, or
  unresolved. Unresolved entries stay public, are flagged, and age_split.py reports the headline with and without them.
- Years: the benchmark's own arXiv id in the quoted citation gives the year first; else the year printed in the quoted
  citation (taken only if that year appears in the quote); never the checking model's memory. Every adopted year is
  listed in the review sheet. Entries without one are dated later by age_split.py (tool evidence, years_manual.json,
  then Haiku's memory last).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

# run by hand: always the held-out sample, whatever the shell has set (benchmark_eval reads these at import)
os.environ["RRN_OUT"], os.environ["RRN_PAPERS"] = "out_heldout", "papers_heldout.json"
from benchmark_eval import OUT, PAPERS, write_atomic  # noqa: E402
from launch import LAUNCH, tree_sha  # noqa: E402

CHECKER = "Gemini (chat, PDF attached)"
PACKET = OUT / "truth_check_packet"
ANSWERS = OUT / "truth_check_answers"
REVIEW = OUT / "truth_check_review.md"
DECISIONS = OUT / "truth_check_decisions.json"
PRE = OUT / "truth_pre_check"
TRUTH_CHECKED = LAUNCH / "heldout_truth_checked.json"

PROMPT = """You are checking the list of evaluation benchmarks that a reading tool extracted from one research paper. The paper's PDF is attached and the list follows it. Verify the list against the paper's text, tables and appendices, entry by entry, and find what the tool missed. Do not rely on memory of the paper or of any benchmark: every judgment must point to a place in the attached PDF.

Kinds.
- public: the paper reports a result on it (a table, figure or numbered result) and it was publicly available before this paper, released by others or earlier by the same authors.
- own: the paper introduces or builds it, including filtered or relabelled versions it names as its own, and in-house or preview sets released with the paper.
- metric: a measure such as accuracy, FID, BLEU or pass@k.
- composite: an umbrella name for several benchmarks reported separately.
- not_a_benchmark: a method, model, tool, simulator, data source or vague phrase.
- not_evaluated: named in the paper, but no result is reported on it (related work, training only, a method's source).

For each listed entry return:
- verdict: keep (right as listed), rekind (right name, wrong kind; give kind), rename (the paper's canonical name or version differs; give new_name), remove (the paper reports no result on it), split (a composite; give components), or unclear (the PDF does not let you decide).
- where: the section, table or figure, with a short quote under 25 words. Leave it empty if you cannot point to a place.
- citation: the paper's bibliography entry for this benchmark, quoted as printed (authors, title, venue, year, arXiv id if printed), or empty if the paper cites none.
- citation_year: the year printed in that bibliography entry, or 0.
- availability: how the PDF shows the benchmark was public before this paper: source_paper (it cites a published or preprint paper that releases the benchmark), release (it points to a dated repository, download page or leaderboard), own (the paper releases it itself), or none (no citation and no release given).
- availability_note: the release or source as the PDF gives it, or empty.
- note: one sentence, only if something is unusual (a subset, a version, a re-split).

Then list missed public benchmarks: every public benchmark the paper reports a result on that is not in the list, with name, where, citation, citation_year, availability, availability_note and note. Check the experiments section, every results table and the appendix. Treat subsets and versions as the parent benchmark unless the paper names the subset as its evaluation. Do not add baselines that are methods.

Answer with JSON only, in exactly this format (one object in "entries" per listed entry, by its number; kind, new_name and components empty unless the verdict needs them):
{"entries": [{"entry": 0, "verdict": "keep", "kind": "", "new_name": "", "components": [], "where": "Table 2: '...'", "citation": "Author et al. Title. Venue, 2024.", "citation_year": 2024, "availability": "source_paper", "availability_note": "", "note": ""}],
 "missed": [{"name": "...", "where": "...", "citation": "...", "citation_year": 0, "availability": "none", "availability_note": "", "note": ""}]}"""

AVAIL = Literal["source_paper", "release", "own", "none"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Verdict(_Strict):
    entry: int = Field(description="The entry's number in the list.")
    verdict: Literal["keep", "rekind", "rename", "remove", "split", "unclear"]
    kind: Literal["public", "own", "metric", "composite", "not_a_benchmark", "not_evaluated", ""]
    new_name: str
    components: list[str]
    where: str
    citation: str
    citation_year: int
    availability: AVAIL
    availability_note: str
    note: str


class Missed(_Strict):
    name: str
    where: str
    citation: str
    citation_year: int
    availability: AVAIL
    availability_note: str
    note: str


class Check(_Strict):
    entries: list[Verdict]
    missed: list[Missed]


def papers() -> list[dict]:
    return json.loads(PAPERS.read_text())


def truth_of(p: dict) -> dict | None:
    f = OUT / "truth" / f"{p['id']}.json"
    return json.loads(f.read_text()) if f.exists() else None


def listing(t: dict) -> str:
    return "\n".join(f"[{i}] {b['name']} (kind: {b['kind']}; also: {', '.join(b['aliases'])})" for i, b in enumerate(t["benchmarks"])) or "(the list is empty)"


def year_from(citation: str, citation_year: int) -> tuple[int | None, str | None]:
    """Section 3's order, from the quoted citation only: the benchmark's arXiv id, else the printed year if it is in the quote."""
    m = re.search(r"(?<!\d)(\d{2})(\d{2})\.\d{4,5}(?!\d)", citation or "")
    if m and 1 <= int(m.group(2)) <= 12:
        return 2000 + int(m.group(1)), "arxiv_id_in_citation"
    if citation_year and str(citation_year) in (citation or ""):
        return citation_year, "paper_citation"
    return None, None


def needs_resolution(availability: str) -> str | None:
    """Pre-set resolution of a public entry: shown by the PDF as a source paper or a dated release is public; anything
    else (the paper's own release, or nothing) is left for the reviewer."""
    return "public" if availability in ("source_paper", "release") else None


def cmd_packet() -> None:
    PACKET.mkdir(parents=True, exist_ok=True); ANSWERS.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in papers():
        t = truth_of(p)
        if t is None:
            continue
        (PACKET / f"{p['arxiv']}.md").write_text(f"{PROMPT}\n\n----\nPaper: arXiv {p['arxiv']}, {p['title']}\n\nThe extracted list:\n{listing(t)}\n")
        (PACKET / f"{p['arxiv']}.listed.json").write_text(json.dumps([b["name"] for b in t["benchmarks"]], ensure_ascii=False))
        n += 1
    print(f"{n} packets in {PACKET}; save each Gemini answer as {ANSWERS}/<arXiv id>.json")


def answer_of(p: dict) -> dict:
    f, lf = ANSWERS / f"{p['arxiv']}.json", PACKET / f"{p['arxiv']}.listed.json"
    if not lf.exists():
        sys.exit(f"no packet for {p['arxiv']}; run 'packet' first")
    if not f.exists():
        sys.exit(f"no Gemini answer for {p['arxiv']} ({f})")
    try:
        check = Check.model_validate_json(f.read_text()).model_dump()
    except Exception as e:  # noqa: BLE001
        sys.exit(f"{f.name} does not match the answer format: {e}")
    return {"check": check, "listed": json.loads(lf.read_text())}


def cmd_review() -> None:
    lines = ["# Held-out truth check: decisions for the reviewer", "",
             "In truth_check_decisions.json, set every null: apply (true or false) for each non-keep verdict and each missed benchmark, after opening the "
             "PDF at the quoted place; and resolution (public, own or unresolved) for each entry that would be public but whose availability the PDF "
             "does not show as a source paper or dated release, with resolution_evidence for public. Verdicts without a location are pre-set not to "
             "apply; 'unclear' leaves the entry as it is. The years to be adopted are listed per paper; each comes from the quoted citation only.", ""]
    dec = {"reviewer": "", "date": "", "papers": {}}
    for p in papers():
        t = truth_of(p)
        if t is None:
            continue
        c = answer_of(p)
        if c["listed"] != [b["name"] for b in t["benchmarks"]]:
            sys.exit(f"{p['arxiv']}: the truth list changed after the packet was written; write the packet and check again")
        pd = {"entries": {}, "missed": []}
        lines += [f"## {p['arxiv']} {p['title'][:80]}", ""]
        years = []
        for v in c["check"]["entries"]:
            if not 0 <= v["entry"] < len(t["benchmarks"]):
                continue
            b = t["benchmarks"][v["entry"]]
            item = {k: x for k, x in v.items() if k != "entry"}
            if v["verdict"] == "keep":
                item["apply"] = True
            elif v["verdict"] == "unclear" or not v["where"].strip():
                item["apply"] = False
                item["reason"] = "unclear: entry left as listed" if v["verdict"] == "unclear" else "no location: not applied"
            else:
                item["apply"] = None
            # public if the verdict is applied (rekind to public) or if it stays as listed (public, verdict not applied or keep)
            maybe_public = (v["verdict"] == "rekind" and v["kind"] == "public") or (b["kind"] == "public" and v["verdict"] in ("keep", "rename", "unclear", "rekind", "remove"))
            if maybe_public and not (v["verdict"] in ("rekind", "remove") and b["kind"] == "public" and item["apply"] is True):
                item["resolution"] = needs_resolution(v["availability"])
            elif v["verdict"] in ("rekind", "remove") and b["kind"] == "public":
                item["resolution"] = needs_resolution(v["availability"])  # used only if the reviewer declines the verdict
            y, src = year_from(v["citation"], v["citation_year"])
            if y and "resolution" in item:
                years.append(f"{b['name']}: {y} ({src})")
            pd["entries"][b["name"]] = item
            if v["verdict"] != "keep" or item.get("resolution", "public") is None:
                lines.append(f"- **{b['name']}** ({b['kind']}): {v['verdict']}"
                             + (f" -> {v['kind'] or v['new_name'] or ', '.join(v['components'])}" if v['verdict'] in ('rekind', 'rename', 'split') else "")
                             + f". Where: {v['where'] or '(none)'}. Availability: {v['availability']} {v['availability_note']}".rstrip()
                             + f". Tool's snippet: {b.get('snippet') or '(none)'}" + (f". Note: {v['note']}" if v["note"] else ""))
        for b in t["benchmarks"]:
            pd["entries"].setdefault(b["name"], {"verdict": "unchecked", "apply": False, "reason": "the checking model returned no verdict",
                                                 **({"resolution": None} if b["kind"] == "public" else {})})
        for m in c["check"]["missed"]:
            it = {**m, "apply": None if m["where"].strip() else False, "resolution": needs_resolution(m["availability"])}
            if not m["where"].strip():
                it["reason"] = "no location: not applied"
            y, src = year_from(m["citation"], m["citation_year"])
            if y:
                years.append(f"{m['name']} (missed): {y} ({src})")
            pd["missed"].append(it)
            lines.append(f"- missed: **{m['name']}**. Where: {m['where'] or '(none)'}. Availability: {m['availability']} {m['availability_note']}".rstrip()
                         + f". Citation: {m['citation'] or '(none)'}")
        lines += [f"- years to be adopted: {'; '.join(years) if years else 'none'}", ""]
        dec["papers"][p["arxiv"]] = pd
    REVIEW.write_text("\n".join(lines) + "\n")
    if DECISIONS.exists():
        sys.exit(f"{DECISIONS.name} exists and was not overwritten; delete it to start the review again ({REVIEW.name} rewritten)")
    DECISIONS.write_text(json.dumps(dec, indent=1, ensure_ascii=False))
    print(f"{REVIEW}\n{DECISIONS}")


def resolve(b: dict, it: dict) -> None:
    """Apply the availability resolution to an entry that is public after the verdict: own, or public flagged unresolved."""
    r = it.get("resolution")
    if b["kind"] != "public" or r is None:
        return
    if r == "own":
        b["kind"] = "own"
    elif r == "unresolved":
        b["unresolved"] = True


def cmd_apply() -> None:
    if (LAUNCH / "status_C2.json").exists():
        sys.exit("C2 has started; the checked truth lists are final")
    if not DECISIONS.exists():
        sys.exit("no decisions file; run 'review' and fill it in first")
    dec = json.loads(DECISIONS.read_text())
    if not dec.get("reviewer") or not dec.get("date"):
        sys.exit("set reviewer and date in the decisions file")
    items = [(ax, n, it) for ax, pd in dec["papers"].items() for n, it in pd["entries"].items()] + \
            [(ax, f"missed {m['name']}", m) for ax, pd in dec["papers"].items() for m in pd["missed"]]
    open_items = [f"{ax}: {n} ({k})" for ax, n, it in items for k in ("apply", "resolution")
                  if k in it and it[k] is None and not (k == "resolution" and n.startswith("missed ") and it.get("apply") is False)]
    bad = [f"{ax}: {n}" for ax, n, it in items if it.get("resolution") not in (None, "public", "own", "unresolved")]
    if open_items or bad:
        sys.exit(f"{len(open_items)} items have no decision, {len(bad)} have an invalid resolution: " + "; ".join((open_items + bad)[:10]))
    if not PRE.exists():
        shutil.copytree(OUT / "truth", PRE)
    checker = f"{CHECKER} reviewed by {dec['reviewer']}"
    marks = {}
    for p in papers():
        pd = dec["papers"].get(p["arxiv"])
        tf = PRE / f"{p['id']}.json"
        if pd is None or not tf.exists():
            continue
        t = json.loads(tf.read_text())  # always from the pre-check copy, so apply can be re-run after a corrected decision
        new, changes = [], 0
        for b in t["benchmarks"]:
            it = pd["entries"].get(b["name"], {"verdict": "unchecked", "apply": False})
            rec = {"date": dec["date"], "checker": checker, "verdict": it["verdict"], "applied": bool(it.get("apply")),
                   "where": it.get("where", ""), "citation": it.get("citation", ""), "availability": it.get("availability", ""),
                   "availability_note": it.get("availability_note", ""), "resolution": it.get("resolution"),
                   "resolution_evidence": it.get("resolution_evidence", "")} | ({"reason": it["reason"]} if it.get("reason") else {})
            b = dict(b)
            if it.get("apply"):
                v = it["verdict"]
                if v == "rekind" and it["kind"]:
                    b["kind"] = it["kind"]
                elif v == "rename" and it["new_name"]:
                    b["aliases"] = sorted(set(b["aliases"]) | {b["name"], it["new_name"]})
                    b["name"] = it["new_name"]
                elif v == "remove":
                    b["kind"] = "not_evaluated"
                elif v == "split" and it["components"]:
                    b["kind"] = "composite"
                    new += [{"name": c, "aliases": [c], "kind": "public", "found_by": ["truth check"], "in_text": True, "snippet": None,
                             "unresolved": True, "hand_check": {**rec, "verdict": f"component of {b['name']}", "resolution": "unresolved"}}
                            for c in it["components"]]  # a component's availability is not shown per component: listed as unresolved
                changes += v != "keep"
            resolve(b, it)
            if b["kind"] == "public":
                y, src = year_from(it.get("citation", ""), it.get("citation_year", 0))
                if y:
                    b["year"], b["year_source"] = y, src
            b["hand_check"] = rec
            new.append(b)
        for m in pd["missed"]:
            if m.get("apply"):
                rec = {"date": dec["date"], "checker": checker, "verdict": "missed", "applied": True, "where": m["where"], "citation": m["citation"],
                       "availability": m["availability"], "availability_note": m["availability_note"], "resolution": m.get("resolution"),
                       "resolution_evidence": m.get("resolution_evidence", "")}
                e = {"name": m["name"], "aliases": [m["name"]], "kind": "public", "found_by": ["truth check"], "in_text": True, "snippet": None, "hand_check": rec}
                resolve(e, m)
                y, src = year_from(m["citation"], m["citation_year"])
                if y and e["kind"] == "public":
                    e["year"], e["year_source"] = y, src
                new.append(e); changes += 1
        t["benchmarks"] = new
        t["checked_by_author"] = True
        t["hand_check"] = {"date": dec["date"], "checker": checker, "status": "complete"}
        write_atomic(OUT / "truth" / f"{p['id']}.json", json.dumps(t, indent=1, ensure_ascii=False))
        marks[p["arxiv"]] = {"checked": True, "entries": len(new), "changes": changes, "public": sum(b["kind"] == "public" for b in new),
                             "unresolved": sum(bool(b.get("unresolved")) for b in new)}
    for p in papers():  # a paper with no truth list (unreadable PDF, swapped by the recipe) has nothing to check
        if p["arxiv"] not in marks and not (OUT / "truth" / f"{p['id']}.json").exists():
            marks[p["arxiv"]] = {"checked": True, "entries": 0, "changes": 0, "public": 0, "unresolved": 0, "note": "no truth list"}
    TRUTH_CHECKED.parent.mkdir(parents=True, exist_ok=True)
    TRUTH_CHECKED.write_text(json.dumps({"date": dec["date"], "reviewer": dec["reviewer"], "checker": checker, "applied_on": date.today().isoformat(),
                                         "truth_sha256": tree_sha(OUT / "truth")["sha256"], "papers": marks}, indent=1))
    print(f"{len(marks)} papers marked checked -> {TRUTH_CHECKED}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    {"packet": cmd_packet, "review": cmd_review, "apply": cmd_apply}.get(cmd, lambda: sys.exit(__doc__))()
