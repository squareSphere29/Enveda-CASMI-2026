# CASMI26 pipeline status

Sources: this repo's scripts, GitHub `squareSphere29/Enveda-CASMI-2026` commit `d527b20`, Kaggle evaluation snippet (MRR@25 on tautomer-canonical InChIKey14), [m0-ar RESEARCH.md](https://m0-ar.github.io/casmi-2026/RESEARCH.md), [shiferaxa/casmi26 docs/plan.md](https://github.com/shiferaxa/casmi26/blob/main/docs/plan.md).

## Metric

Guess and truth are tautomer-canonicalized with RDKit 2026.03.3, then compared on InChIKey14. Score is MRR@25. Notebook submissions, internet off, 9 hour limit.

## What was wrong

Train queries used the molecule's own concatenated library spectrum, so cosine of the truth was a self-match. `adduct_offsets.json` used a 3 Da cap because molecule-level residuals mixed adducts. `collect_records` then cosine-scored thousands of candidates. Validation never finished. Empty windows were dropped from MRR. The np panel was in-sample. Public rankers are linear analogs, not forked notebooks. There is no PyTorch FPNet.

## What this pass changed

Query is one held-out train spectrum. Truth cosine uses the other rows of that molecule. Mass window is per-adduct Da shift with 0.005–0.05 Da tolerance. Extra hypotheses are same-mode monomers. `mass_ppm` is centered on the shift. MRR includes misses as 0. Default validation is 200 non-np keys plus a held-out np panel. `score_test.py` loads `np_keys.json`. Paths are relative to the repo root.

## Still open

Gluzdov identity-disjoint folds are not loaded. Public haideptry / prvsiyan engines are not forked. Candidate pool is still train-only, so novel structures cannot hit. A 5% tautomer mismatch vs the `inchikey14` column remains.
