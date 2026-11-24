# About MRR, classes, and the local driver

The scored file is [What the competition scores](deliverable.md). Where points move is [Score levers in the local driver](score-levers.md).

This page is the meaning of the overnight table, not a runbook.

## MRR@25

Kaggle parses each predicted SMILES with RDKit, tautomer-canonicalizes the molecule, and compares the first 14 characters of the InChIKey to the truth. That block is connectivity. Stereo does not count.

Rank 1 scores 1. Rank 2 scores 0.5. Rank 25 scores 0.04. A miss in the top 25 scores 0. The leaderboard is the mean of those scores over 400 test molecules.

The overnight column `MRR@25` is the same mean over 250 np-examples structures, not over the test set. `r1` is how many of the 250 were correct in slot 1. `miss` is how many had no hit in 25. `top25` is 250 minus `miss`.

## np-examples

Rows with `ingest_lib == enveda-np-examples` in `train.parquet`. There are 250 structures the host measured on the test instrument. `run_fork_val.py --split npx` uses only those keys. That split predicted the public board. Random held-out train structures did not.

## Class 1

Every training spectrum of the structure stays in the library. The query is still the np-examples spectra of that structure. A high score means the pipeline can recognize a molecule it has already seen. Public v2 checkpoints plus ranker v4 sit near 0.994 here. The board is not class 1.

## Class 2

`hide_key` removes that structure's spectra from library search and analog search. The structure stays in the candidate pool, so mass matching and FPNet can still pick it. This is a known structure with a new spectrum. Public protocol sits near 0.758. Overnight success is this number.

## Class 3

The same hide as class 2, plus `drop_key` removes that structure key from the pool. The only route to the truth is another copy, usually COCONUT under a different string. Public protocol sits near 0.083. The board is closer to a mix of class 2 and class 3 than to class 1.

## What `harness.py` does

`vendor/haideptry-casmi26/src/casmi/fork/harness.py` is the local driver for the public notebook. It is not a second ranker.

`init` loads `train.parquet`, the COCONUT-plus-train pool, the two `fp_*.pt` files, and fits eight gradient boosting models on `rank_train.npz`. For each of the 250 keys it runs three masks:

1. `channels` computes library entropy similarity, analog Tanimoto, and FPNet logits.
2. `evidence` takes the 8.5 ppm mass window, prunes, and scores fragmentation.
3. `rank` builds 31 features, runs the ensemble, and returns 25 SMILES after `dedupe_fill`.

Overnight used `--prunes 400:lib --classes 1,2,3 --sim-powers 4`. The log prints one MRR row per class for that prune. A class 2 of 0.0166 from analog `scripts/validate.py` is a different engine.
