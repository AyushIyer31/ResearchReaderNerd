# Stage C held-out sample: recipe, seed and exclusions (14 November 2026)

Fixed before any draw, under protocol sections 2 and 6. The machine-readable copy is experiments/heldout_recipe_2026-11-14.json; its sha256 is in launch_manifest_2026-11-14.json, and launch.py runs the command below from that file.

## Recipe

| Item | Value |
| --- | --- |
| Script | experiments/sample_2026.py (sha256 in the launch manifest) |
| Seed | 23 (development draws used 7 and 11) |
| Publication window | 2026-11-04 to 2026-11-08 |
| Draw date | not before 2026-11-14, and only once the stage A decision is recorded, stage B is complete (or the group's decision to go on from a B that stopped early is recorded) and the version-2 predictions are recorded (launch.py refuses otherwise) |
| Size | 30 papers, at most 3 per OpenAlex primary topic |
| Source | arXiv preprints in OpenAlex (source S4306400194), primary field Computer Science (fields/17) |
| Other filters | more than 15 referenced works; has an abstract; abstract at least 300 characters |
| Sample | OpenAlex seeded random sample of 150 works, scanned in the order returned |

Command, run once by launch.py stage C1:

```
python sample_2026.py --n 30 --seed 23 --per-topic 3 --start 2026-11-04 --end 2026-11-08 --out papers_heldout.json --exclude papers.json,papers_2026.json,papers_2026b.json
```

OpenAlex query:

```
GET https://api.openalex.org/works?filter=from_publication_date:2026-11-04,to_publication_date:2026-11-08,type:preprint,primary_topic.field.id:fields/17,referenced_works_count:>15,primary_location.source.id:S4306400194,has_abstract:true&sample=150&seed=23&per_page=150&select=id,title,publication_year,publication_date,doi,cited_by_count,referenced_works_count,primary_topic,abstract_inverted_index,ids
```

## Selection

1. Scan the 150 works in the order OpenAlex returns them.
2. Skip a work with no arXiv id in its DOI or ids, or whose arXiv id is excluded.
3. Skip a work whose abstract is under 300 characters.
4. Take it if its primary topic has fewer than 3 papers taken. Stop at 30.

## Exclusions

91 arXiv ids: the 90 papers of the three development sets (old 2018-2021, post-cutoff a, post-cutoff b), and 2609.38721, which was swapped out of set a for an oversized PDF and is skipped by the script itself. No development paper falls in the window: the latest publication date among the 60 post-cutoff development papers is 2026-11-02 (the old set is 2018 to 2021), and among the 144 works of set b's 150-work sample that passed the arXiv-id, exclusion and abstract checks it is 2026-11-03; the window starts on 2026-11-04. The exclusion by id applies regardless.

```
1801.02062 1801.09847 1802.04799 1806.02817 1807.00652 1808.07233 1810.00184 1901.08933 1901.11448 1903.12136 1906.06032 1907.00953 1908.09791 1909.00025 1910.04855 1910.13598 1910.14425 1911.02744 2002.07948 2002.08988 2002.09434 2002.11569 2006.12567 2007.02835 2007.04439 2009.14471 2103.12051 2106.06984 2107.00135 2108.07732 2608.03461 2608.03632 2608.04887 2608.05108 2608.07639 2608.09999 2608.13026 2608.17023 2608.18008 2608.18937 2608.23552 2608.24848 2608.25148 2608.27168 2608.28062 2608.28272 2608.30714 2609.02272 2609.06783 2609.08690 2609.09212 2609.09495 2609.09643 2609.11249 2609.12317 2609.17210 2609.20464 2609.29015 2609.31654 2609.32618 2609.35189 2609.36235 2609.37105 2609.38145 2609.38721 2609.39982 2610.00095 2610.00949 2610.01942 2610.02201 2610.03713 2610.05534 2610.07835 2610.08232 2610.09240 2610.11123 2610.11459 2610.13465 2610.15721 2610.24348 2610.24640 2610.28871 2610.29523 2610.31656 2610.35193 2610.39175 2610.39377 2610.41992 2610.42082 2611.00408 2611.02779
```

## Window choice

The window had already closed when this recipe was fixed on 14 November, so the sample can be drawn as soon as the stage A decision is recorded. It starts the day after the latest publication date among the checked works of set b's sample (2026-11-03), so no development paper falls in it, and it is after Haiku 5.5's June 2026 cutoff. The six days between the window's end and the earliest draw let OpenAlex index the papers' reference lists, which the reference-count filter needs. The papers in the window were not looked at before the recipe was fixed, and the sample is fixed by the seed, so the draw remains held out.

## Replacement and shortfall

- A paper whose PDF cannot be read (over 24 MB, or withdrawn on arXiv) is replaced before any run (launch.py pauses C1 after the draw for this check, and resumes only when heldout_pdfs_checked.json marks every PDF checked) by the next eligible work in the same scan (same filters and topic limit), and the swap is recorded in papers_heldout_swaps.json. No other replacement: a paper with no public benchmark is read, counted and not scored.
- If fewer than 30 are taken from the 150, the sample is what was taken; no second draw and no other seed.

## Records kept

- papers_heldout.json (the 30)
- sample_raw_heldout.json (every scanned work that passed the arXiv-id, exclusion and abstract checks, in order)
- draw_response_heldout.json (OpenAlex's full response to the draw; launch.py records its sha256 in out/launch_2026-11-14/)
- papers_heldout_swaps.json if any

OpenAlex's seeded sample is reproducible only while its index is unchanged. The saved draw response and its hash are therefore the record of which papers were drawn.
