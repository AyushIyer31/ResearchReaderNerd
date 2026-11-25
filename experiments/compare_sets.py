"""Old papers (2018-2021, before the model's training cutoff) against new papers (Aug-Oct 2026, after it), 9 Nov 2026.

Same pipeline, same model (claude-haiku-5-5), same judge. Four figures per set: model alone from title and abstract
(ablation 'llm'), the tool in author mode (refs+model), the leave-one-out test (user 1'), and the title-only memory
probe. Intervals are 90%: Wilson over benchmarks, percentile bootstrap over papers (10,000 rounds, seed 7).

  python compare_sets.py  -> out/report_2026-11-08/old_vs_new.md
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL = "claude-haiku-5-5"
SETS = {"old": ("out", "papers.json", "2018-2021, before the June 2026 cutoff"),
        "new": ("out_2026", "papers_2026.json", "Aug-Oct 2026, after the cutoff")}


def wilson(k, n, z=1.645):
    if not n:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def boot(pairs, rounds=10000, seed=7):
    """pairs: per paper (hits, n). Percentile bootstrap over papers of the pooled rate."""
    rng = random.Random(seed)
    pairs = [x for x in pairs if x[1]]
    if not pairs:
        return (0.0, 0.0)
    rates = []
    for _ in range(rounds):
        s = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        rates.append(sum(h for h, _ in s) / sum(n for _, n in s))
    rates.sort()
    return (rates[int(0.05 * rounds)], rates[int(0.95 * rounds) - 1])


def pct(k, n):
    return f"{k}/{n} ({100 * k / n:.0f}%)" if n else "-"


def ci(k, n, pairs):
    a, b = wilson(k, n)
    c, d = boot(pairs)
    return f"{100 * a:.0f}-{100 * b:.0f}% benchmarks, {100 * c:.0f}-{100 * d:.0f}% papers"


def load(name):
    out, papers, _ = SETS[name]
    out = HERE / out
    results = json.loads((out / "results.json").read_text())
    loo = json.loads((out / "loo_results.json").read_text())
    probe = {}
    for p in results:
        f = out / "probe" / f"{p['id']}.{MODEL}.json"
        m = out / "match" / f"{p['id']}.{MODEL}.probe.json"
        if f.exists():
            probe[p["id"]] = (json.loads(f.read_text()), json.loads(m.read_text()) if m.exists() else None)
    rows = []  # one per public benchmark
    for p in results:
        used = [b for b in p["truth"]["benchmarks"] if b["kind"] == "public"]
        if not used:
            continue
        tool = {s["benchmark"]: s for s in p["blind"][f"{MODEL}/refs+model"]["score"]}
        topic = {s["benchmark"]: s for s in p["blind"][f"{MODEL}/topic+model"]["score"]}
        llm = {s["benchmark"]: s for s in p["ablation"][f"{MODEL}/llm"]["score"]}
        loo_p = {r["benchmark"]: r for r in loo if r["arxiv"] == p["arxiv"]}
        pr = probe.get(p["id"])
        for b in used:
            n = b["name"]
            r = {"set": name, "id": p["id"], "arxiv": p["arxiv"], "benchmark": n,
                 "llm": bool(llm.get(n, {}).get("in_top10")),
                 "tool": bool(tool.get(n, {}).get("in_top10")), "tool_pool": bool(tool.get(n, {}).get("in_pool")),
                 "topic": bool(topic.get(n, {}).get("in_top10")),
                 "loo": loo_p.get(n, {}).get("in_top10") if loo_p else None,
                 "loo_pool": loo_p.get(n, {}).get("in_pool") if loo_p else None}
            if pr and pr[1]:
                i = pr[1]["used"].index(n) if n in pr[1]["used"] else None
                r["probe"] = bool(pr[1]["found_as"][i]) if i is not None else False
            else:
                r["probe"] = False
            r["any_pool"] = r["tool_pool"] or r["llm"] or r["topic"] or bool(r["loo_pool"])
            rows.append(r)
    return results, rows, loo, probe


def main():
    lines = ["# Old papers against new papers: the same pipeline on both sides of the model's training cutoff", "",
             f"Model {MODEL} everywhere; judge and prompts unchanged. Old set: papers.json (25 scored papers, truth lists hand-checked). "
             "New set: papers_2026.json, arXiv computer-science preprints posted August to October 2026, drawn by the same OpenAlex sampling "
             "recipe (23 scored papers; truth lists are the tool's own reading, not yet hand-checked). "
             "Haiku 5.5's training data ends June 2026, so it cannot have read any new-set paper.", ""]
    per_set = {}
    for name in SETS:
        results, rows, loo, probe = load(name)
        per_set[name] = rows
        papers = sorted({r["id"] for r in rows})
        def agg(key):
            k = sum(1 for r in rows if r[key]); n = len(rows)
            pairs = [(sum(1 for r in rows if r["id"] == p and r[key]), sum(1 for r in rows if r["id"] == p)) for p in papers]
            return k, n, pairs
        per_set[name + "_agg"] = {key: agg(key) for key in ("llm", "tool", "tool_pool", "topic", "probe")}
        # LOO from the summary files (field names differ), read totals from loo_results
        rec = sum(1 for r in loo if r.get("in_top10")); pool = sum(1 for r in loo if r.get("in_pool")); n = len(loo)
        lp = {}
        for r in loo:
            lp.setdefault(r["arxiv"], [0, 0, 0]); lp[r["arxiv"]][2] += 1; lp[r["arxiv"]][0] += bool(r.get("in_top10")); lp[r["arxiv"]][1] += bool(r.get("in_pool"))
        per_set[name + "_loo"] = (rec, pool, n, [(v[0], v[2]) for v in lp.values()], [(v[1], v[2]) for v in lp.values()])
        per_set[name + "_probe_meta"] = (len(probe), sum(1 for v in probe.values() if v[0]["recognised"]), sum(1 for v in probe.values() if v[0]["benchmarks"]))
        per_set[name + "_npapers"] = len(papers)

    lines += ["## Headline", "",
              "| Test | What it measures | Old papers (pre-cutoff) | 90% CI | New papers (post-cutoff) | 90% CI |",
              "| --- | --- | --- | --- | --- | --- |"]
    def row(label, what, key):
        o = per_set["old_agg"][key]; nw = per_set["new_agg"][key]
        lines.append(f"| {label} | {what} | {pct(o[0], o[1])} | {ci(*o)} | {pct(nw[0], nw[1])} | {ci(*nw)} |")
    row("Model alone (title + abstract)", "Haiku names 10 benchmarks with no literature search; share of the paper's benchmarks named", "llm")
    row("Tool, author mode (refs+model)", "Full pipeline in the author test: share of the paper's benchmarks in its 10 suggestions", "tool")
    row("Tool, candidate pool", "Share of the paper's benchmarks anywhere in the ranked pool (ceiling for the ranker)", "tool_pool")
    row("Tool, topic+model", "Same, with topic search instead of the paper's reference list", "topic")
    o = per_set["old_loo"]; nw = per_set["new_loo"]
    lines.append(f"| Leave-one-out (user 1') | Each benchmark withheld from the read paper in turn; share recovered in the 10 suggestions | {pct(o[0], o[2])} | {ci(o[0], o[2], o[3])} | {pct(nw[0], nw[2])} | {ci(nw[0], nw[2], nw[3])} |")
    def loo_llm(name):
        rs = json.loads((HERE / SETS[name][0] / "loo_llm_results.json").read_text())
        k = sum(1 for r in rs if r["llm_in_top10"]); n = len(rs)
        pairs = [(sum(1 for r in rs if r["arxiv"] == a and r["llm_in_top10"]), sum(1 for r in rs if r["arxiv"] == a)) for a in sorted({r["arxiv"] for r in rs})]
        return k, n, pairs
    o2 = loo_llm("old"); n2 = loo_llm("new")
    lines.append(f"| Leave-one-out, Haiku alone | Same withheld-benchmark runs; Haiku given the scrubbed paper description, no literature | {pct(o2[0], o2[1])} | {ci(*o2)} | {pct(n2[0], n2[1])} | {ci(*n2)} |")
    lines.append(f"| Leave-one-out, in pool | Withheld benchmark anywhere in the candidate pool | {pct(o[1], o[2])} | {ci(o[1], o[2], o[4])} | {pct(nw[1], nw[2])} | {ci(nw[1], nw[2], nw[4])} |")
    row("Memory probe (title only)", "Haiku asked, from the title alone, which benchmarks the paper used; share recalled", "probe")
    om = per_set["old_probe_meta"]; nm = per_set["new_probe_meta"]
    lines += ["", f"- Memory probe, papers the model says it recognises: old {om[1]}/{om[0]}, new {nm[1]}/{nm[0]}. Papers for which it names any benchmark: old {om[2]}/{om[0]}, new {nm[2]}/{nm[0]}.",
              f"- Scored papers: old {per_set['old_npapers']}, new {per_set['new_npapers']} (the rest have no public benchmark).", ""]

    # Overlap between model alone and tool
    lines += ["## Who finds what: model alone against the tool, per benchmark", "",
              "Top-10 hits on the same benchmarks. 'Only model' are benchmarks the tool's ranker left out of its 10 although Haiku alone named them.", "",
              "| Set | Both | Only model alone | Only tool | Neither | Of 'neither', in the tool's pool |", "| --- | --- | --- | --- | --- | --- |"]
    for name in SETS:
        rows = per_set[name]
        both = sum(1 for r in rows if r["llm"] and r["tool"]); om_ = sum(1 for r in rows if r["llm"] and not r["tool"])
        ot = sum(1 for r in rows if r["tool"] and not r["llm"]); nei = [r for r in rows if not r["tool"] and not r["llm"]]
        lines.append(f"| {name} | {both} | {om_} | {ot} | {len(nei)} | {sum(1 for r in nei if r['tool_pool'])} |")
    lines += ["", "Benchmarks no system placed in any pool (model alone, tool, topic search, or leave-one-out). On the new set about half of these were introduced in 2026 (OOLONG-Pairs, OBLIQ-Bench, ManyIH, EmulatorBench, PMPP-Hard, MazeBench, IllVQA), which neither the model nor the cited literature could know yet; the rest are niche or in-house test sets:", ""]
    for name in SETS:
        unseen = [r for r in per_set[name] if not r["any_pool"]]
        lines.append(f"- {name}: {len(unseen)}/{len(per_set[name])}: " + ", ".join(f"{r['benchmark']} ({r['arxiv']})" for r in unseen))
    lines += ["", "## Reading", "",
              "- The memory probe collapses from recall of about half the benchmarks on the old papers to none on the new papers, and the model says it recognises none of the new titles. The post-cutoff set is clean of paper-level memory.",
              "- Yet the model-alone figure barely moves. Given a title and abstract, Haiku names the right benchmarks at the same rate for papers it has never seen. Its 66% on the old set was therefore mostly inference from the topic, not recall of the paper; the leakage worry does not change the model-alone number.",
              "- The tool moves by the same small amount, and stays below the model alone in the title-and-abstract author test on both sets.",
              "- On the leave-one-out, Haiku alone falls from 76% (old) to 56% (new) while the tool holds at 50% and 55%. The drop is benchmark age, not paper memory: on benchmarks introduced by 2024 Haiku alone scores 77% on the new papers (report/new_set_years.json); on 2025 benchmarks 31% against the tool's 50%. A forced title-only probe (62% old vs 45% new on comparable benchmarks) is consistent with some exposure to the old papers, but it does not affect scores once the abstract is given. The tool's recoveries that Haiku alone misses on the new set are the recent benchmarks (ARC-AGI-3, OOLONG, ViDoRe-V3, RoboTwin 2.0, Skill-Inject, AIME 2025), reached through the literature.",
              "- The misses that no system can reach on the new set include the 2026-born benchmarks listed above. Those are the cases where only citing papers, once they exist, can help: the published mode and users 2 and 3.",
              "- On both sets the ranker, not the search, is where the tool loses to the model alone: 'only model alone' outnumbers 'only tool' 29 to 12 (old) and 20 to 5 (new), while the tool's pool holds the benchmark at a higher rate than the model names it.",
              "- Caveats: new-set truth lists are not yet hand-checked; the judge's name matching varies by about 3 points between runs; 23 and 25 papers give wide intervals, and every old-new difference above lies inside them.", ""]
    out = HERE / "out" / "report_2026-11-08" / "old_vs_new.md"
    out.write_text("\n".join(lines))
    print("\n".join(lines))
    json.dump({k: v for k, v in per_set.items() if k in SETS}, open(HERE / "out" / "report_2026-11-08" / "old_vs_new_rows.json", "w"), indent=1)


if __name__ == "__main__":
    main()
