# Free sources vs OpenAlex (24 November 2026)

**Setup**
- **Papers.** 10 development papers (post-cutoff sets a and b), drawn with seed 7 from the 17 whose own arXiv bibliography names at least two of their public truth benchmarks.
- **Gold.** For each benchmark, the gold is the bibliography entry whose title names it: the benchmark's source paper, as the authors cite it. This gold doesn't depend on OpenAlex.
- **Excluded.** No Claude calls, and the held-out windows were not touched.
- **Cost.** OpenAlex used $0.17 of the free $1 daily budget, with nothing paid. Semantic Scholar ran without a key.
- **Script and raw output.** experiments/free_source_test.py and free_source_test.json.
- **Rules, fixed before running.**
  - A free source wins a function if it is within 5 points of OpenAlex or better, at a workable call count.
  - Any function it loses is a gap that still needs OpenAlex.

## Results

| Function | OpenAlex | Semantic Scholar | Crossref / OpenCitations | Verdict |
| --- | --- | --- | --- | --- |
| 1. Title lookup: correct paper found (150 titles from the papers' bibliographies) | 111/150 (74%); gold 29/36 | 123/148 (83%); gold 32/36 | Crossref 72/150 (48%), correct at rank 1 for 66; gold 21/36 | **Semantic Scholar wins.** Crossref loses |
| 2. Reference list: gold source papers present | 29/36 (81%); 357 references | 33/36 (92%); 455 references (the papers' bibliographies have 478 entries) | OpenCitations returned 0 references for all 10; Crossref has no record of any of the 10 (arXiv DOIs are not in Crossref) | **Semantic Scholar wins.** Crossref and OpenCitations can't be used for arXiv papers |
| 3. Co-usage: top 15 most-cited papers citing a benchmark's source paper (16 benchmarks) | 1 call each, sorted on the server | No server-side sort: we pulled all citations, 1-10 calls each. Stops at about 9,000 citations (HTTP 400 past that), so the list was truncated for ImageNet, Cityscapes and nuScenes. Overlap with OpenAlex's top 15: 0 to 12, median about 6 | Not tested (OpenCitations has no citations for arXiv DOIs) | **Unresolved.** It works for most benchmarks, but gives a different top 15 and truncates very popular ones. Agreement isn't correctness: settling this needs a pipeline-level check of which source surfaces more truth benchmarks |
| 4. Topic search with a date limit (30 of the tool's real queries) | 300 results; 5 gold hits | 230 results (some queries returned fewer than 10; 7 refused with 429); 179 overlap with OpenAlex's top 10; 4 gold hits | Not tested | **About a tie.** Too few gold hits to separate them |
| 5. Random sample (past window, 19-23 Oct 2026) | 2,265 eligible under the recipe's filters, in one call | Bulk search returned 2. My venue filter was probably wrong, so this is inconclusive | Crossref sample of 20: 0 arXiv papers (bioRxiv, ACS, Springer). The arXiv API lists 3,790 CS papers in the window, but without the recipe's reference-count and abstract filters | **Crossref loses.** arXiv API plus Semantic Scholar could rebuild the sample, but the recipe would change |

**Speed and reliability.**
- Semantic Scholar without a key refused 239 of 477 calls (HTTP 429). A served call took 0.7 s on average. The test took about 40 minutes for 10 papers.
- OpenAlex took 0.4 s per call, with no refusals.
- Crossref took 0.8 s per call.

## Reading

- **Semantic Scholar beats OpenAlex on the two functions that make most of the calls:** title lookup and reference lists. These are also the costly ones on OpenAlex, at $0.001 per title search.
- **Crossref and OpenCitations don't fit this project:** they don't hold arXiv preprints.
- **Two gaps remain.**
  - *Co-usage:* Semantic Scholar's top 15 differs from OpenAlex's and truncates very popular benchmarks.
  - *The seeded sample:* it would have to be rebuilt from the arXiv API.
- **A Semantic Scholar key is required.** Without one, half the calls are refused.

**A hybrid option.**
- Semantic Scholar would handle title lookups and references.
- OpenAlex would handle co-usage and the sample. These are list queries at $0.0001 each, roughly $1 for the whole run, within the free daily budget.
- That would cost about nothing, but any change of source is a new tool version, so the setup must be re-frozen and the development runs redone in stage B.
