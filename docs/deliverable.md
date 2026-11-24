# What the competition scores

Meanings of MRR and the three classes are in [About MRR, classes, and the local driver](mrr-and-classes.md). Where MRR moves is [Score levers in the local driver](score-levers.md).

This page describes the scored artifact. Local plots and `run_fork_val.py` numbers are not the artifact.

## File

The competition is a Kaggle code notebook. The notebook writes `submission.csv` under `/kaggle/working`. The session has no internet. The time limit is 9 hours. Datasets are attached when the kernel starts.

Kaggle scores that notebook output. Kaggle does not score `scripts/score_test.py` or `scripts/pipeline.py`.

Kernel sources that write `submission.csv` live under `vendor/haideptry-casmi26/kaggle/fork_v*`. `vendor/haideptry-casmi26/kaggle/fork_v4/notebook.py` ranks every `molecule_id` in `test.parquet` and writes the file.

## Rows and columns

The file uses the same columns and row order as `data/sample_submission.csv`.

- One row per test molecule. There are 400 molecules. `test.parquet` has 1,213 spectra.
- Column 1 is `molecule_id`.
- Column 2 is `smiles`. The cell holds up to 25 SMILES strings, best first, joined by `;`.
- An empty cell is invalid. The public kernel pads short lists with `CCO`. `casmi.submission.write_submission` writes `C` when the candidate list is empty.

Stereo in a SMILES string does not change the score. The metric drops stereo.

## Match rule

Kaggle parses each predicted SMILES with RDKit 2026.03.3, tautomer-canonicalizes the molecule, computes InChIKey, and keeps the first 14 characters. That block is connectivity. A hit is equality with the truth key. `casmi.metric.reciprocal_rank` skips duplicate keys so they do not consume a rank slot.

The score for one molecule is `1 / rank` of the first hit in the top 25, or 0 if none hit. The leaderboard is the mean over the 400 molecules. The name of that mean is MRR@25.

The public leaderboard is about one third of test. The quota is five submissions per day and two final picks. Team merge closes on 7 Dec 2026. The final close is 14 Dec 2026 23:59 UTC.

## Test mix

The host describes three classes.

1. The structure is in the spectral library and a matching spectrum is visible.
2. The structure is in the library, and the query spectrum is new. The local driver hides that structure's own spectra with `hide_key`.
3. The structure is absent from the library. The local driver also drops that key from the candidate pool with `drop_key`.

Public notebooks treat rows with `ingest_lib == enveda-np-examples` as the split that tracks the board. That set is 250 structures measured on the test instrument. Class 1 MRR near 0.99 and class 2 MRR near 0.76 still map to a public score near 0.339, because class 3 and domain shift dominate the 400.

## Local measurement

`vendor/haideptry-casmi26/scripts/run_fork_val.py` scores the 250 np-examples structures under class 1, class 2, and class 3. A kernel that copies the winning flags is what Kaggle scores.
