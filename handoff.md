# Handoff — CASMI26 Enveda competition solution

Goal: predict up to 25 SMILES guesses per molecule from `data/test.parquet` spectra;
the metric is mean reciprocal rank @ 25, keyed on the metric key.

## Metric key confirmation

The metric key is `inchokey14` in `data/train.parquet` — the first 14 characters of
the InChIKey of the tautomer-canonicalized molecule.
Cross-check: `scripts/train_explore.py` computed
`Chem.MolToInchiKey(rdMolStandardize.TautomerEnumerator().Canonicalize(mol))[:14]`
on 4,000 molecules: 3,804 matched the column, 196 did not (~95%). We dedup the
library (and any structure we generate) on the `inchokey14` column.

## Data

- `data/train.parquet` — 2,539,608 rows (truth: `inchokey14`, `normalized_smiles`,
  spectra: `ms2_mzs`, `ms2_normalized_intensities`, `precursor_mz`, `adduct`,
  `ionization_mode`, `instrument_type`, ...).
- `data/test.parquet` — 1,213 spectra rows, 400 molecules (`molecule_id`, `spectrum_id`,
  `ms2_mzs`, `ms2_normalized_intensities`, `precursor_mz`, `adduct`, ...).
- `data/sample_submission.csv` — format: `molecule_id,smiles`; up to 25 guesses
  semicolon-separated per molecule.
- Built library (cache): `data/pool.parquet` (275,810 unique molecules,
  keyed by `inchokey14`), `data/pool_spec.npz` (per-molecule library spectra,
  402,806,233 points), `data/adduct_offsets.json` (per-adduct offsets/tolerances).
- `data/model.pkl` — final LightGBM ranker (saved by validation).

## Pipeline (`scripts/pipeline.py`)

- `mass_candidates`: per-adduct observed offset/tolerance from `adduct_offsets.json`;
  window on the neutral mass centered on the observed offset, 4 adduct hypotheses.
- Evidence channels per candidate: `spec_cos` (query vs library spectra),
  `fp_match` (FP Tanimoto vs spectral winner), `fp_mean_top5`, `in_silico`
  (fraction of theoretical fragments observed), `popularity`, `doc_prior`
  (np-examples panel: 4.0; ingest-lib weights otherwise), NP features.
- Public rankers as features: v8 (mathleteutd analog), anchored (flexonafft analog,
  POP/SPEC/NP=0.12, LIB_HIT_GATE 0.78), haideptry (4-channel analog).
- Blend: `lgb.LGBMRanker` (250 trees, 0.05 lr, 31 leaves) over the 29 features.
- `train_queries` / `test_queries`: query spectrum = library segment of the molecule;
  `prec` = median precursor of the molecule's rows; adduct = median-precursor row's adduct.

## Validation (`scripts/validate.py`)

Identity-disjoint 5-fold CV (folds by `inchokey14`; 24,000-molecule subsample of the
275,810) + natural-product panel benchmark (enveda-np-examples molecules).
Per fold: fit the blend on the other folds, evaluate with MRR on the fold
(reciprocal rank of the truth in the top 25). Final model trained on all data.

Note: the np panel lives in `np_set` = train rows with `ingest_lib == enveda-np-examples`
(the pool's `ingest_lib` is the first row's value, so it misses np molecules that are
recorded under another lib first).

## Fixes applied this session

- `inchokey14` spelling: the real column is `inchokey14`; fixed `inchokey14` literals in scripts.
- Mass window: `M in [prec/(1+(off+tol)/1e6), prec/(1+(off-tol)/1e6)]` (was off by 1e6).
- `adduct_offsets.json`: recomputed — `ppm = (prec - mass)/mass*1e6`, `offset = median`,
  `tol = quantile(0.999, |ppm-off|) + 1` capped at 12. (The build's version was buggy.)
- `train_queries`/`test_queries`: adduct = the row nearest the median precursor
  (the truth was missing from candidates when the molecule's first adduct differed
  from the median-precursor row's adduct).
- `doc_prior`: 4.0 for np-examples molecules (via `np_set`), else pool `doc_prior`.
- validate.py: new protocol (all validation records; per-fold fit/eval); panel step guarded.

## Results so far

- Smoke (`scripts/smoke_test.py`): truth in candidates + top-1 for 5 molecules.
- Dev tools: `scripts/check_mass_def.py` (precursor vs neutral mass per adduct),
  `scripts/debug_collect.py`, `scripts/debug_rank.py` (truth rank by model prediction),
  `scripts/fix_adduct_stats.py`, `scripts/fix_inch_spelling.py`, `scripts/check_panel_literal.py`.

## Known state / open items

- Validation re-run (2nd attempt) is in flight: PID 42204 (`python scripts/validate.py`).
  Expect fold MRRs + panel MRR in the log `logs/validate.log`. The last (buggy) run
  had MRR ~0.027 because the truth was absent from candidates; the fixes should raise it.
- After validation: read `logs/validate.log` results; update README.md "Results" +
  `make_notebook.py` RESULT_MD; run `scripts/score_test.py` (writes `submission.csv`);
  run `scripts/make_notebook.py` (writes `CASMI26_solution.ipynb`); run
  `scripts/model_insights.py` (feature importances).
- `scripts/build_library.py` — only if `data/pool.parquet` is missing (cache).

## Repo

- `git init` done; no remote yet.
- Remotes: do not push the `data/` folder.
- Commit (next step): `.gitignore`, `handoff.md`, `README.md`, `scripts/`, notebook.

## Commands to continue

```
python scripts/validate.py           # re-running
python scripts/score_test.py         # -> submission.csv
python scripts/make_notebook.py      # -> CASMI26_solution.ipynb
python scripts/model_insights.py     # feature importances
```
