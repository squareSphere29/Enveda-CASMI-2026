# Score levers in the local driver

The scored file is defined in [What the competition scores](deliverable.md).

The public pipeline lives in `vendor/haideptry-casmi26/src/casmi/fork/core.py`. `harness.py` stages the same work so you can change one stage and keep the rest. Dated measurements are in `vendor/haideptry-casmi26/docs/plan.md`. This page argues where new points come from. It is not a runbook.

The analog stack in `scripts/pipeline.py` is a separate, weaker engine. Fold MRR 0.0166 on that stack is not a class 2 score of `harness.py`.

## What `harness.py` computes

`init` loads `train.parquet`, COCONUT fingerprints, `fp_*.pt` checkpoints, and `rank_train.npz`. It fingerprints every training structure that is not already in COCONUT and caches `data/cache/train_fp.npz`.

For each query molecule:

1. `channels` computes library entropy similarity, analog hits, and FPNet logits. `hide_key` removes that structure from library and analog search. The `hide_key` mask is class 2.
2. `evidence` takes the 8.5 ppm mass window, optional 30 ppm fallback, optional `drop_key` for class 3, then prunes, then fragmentation scores.
3. `rank` builds 31 features, runs the eight HistGradientBoosting models, and returns 25 SMILES after `dedupe_fill`.

`run_fork_val.py` calls those three stages for 250 np-examples keys. Public v2 checkpoints plus ranker v4 and prune `400:lib` gave class 1 MRR 0.994, class 2 MRR 0.758, and class 3 MRR 0.083 on 8 Oct 2026. The public kernel of that family scored about 0.339 on the board.

## Where the 250 lose points

Sep 18 2026 `diag_fork_val.py` on np-examples, class 2, prune 80 by library score:

- 248 of 250 truths sit in the 8.5 ppm window. Mass and adducts are not the miss.
- 50 of 250 truths leave the list at prune, because a hidden structure has library similarity 0 and the prune keeps closest mass among the rest.
- Of the 199 that remain, 126 are rank 1 and 2 fall outside the top 25.

Class 3 MRR stays near 0.08 because the training copy is gone. COCONUT often stores the same molecule under a different `inchikey14` string. `dedupe_fill` canonicalizes at rank time. The pool index still keys on the stored string, so `drop_key` does not always remove every copy.

The board sits far below class 2 np-examples. Plan.md treats the gap as class 3 plus timsTOF shift, about a third of the 400.

## Levers, in the order they hit the score

### Candidate recall

If the truth never enters `evidence.cand`, the ranker cannot recover it.

Keep more than 80 candidates, or prune with FPNet score instead of library score. `prune_mode="model"` keeps library hits and otherwise ranks the window by fingerprint dot logits. Kernel `fork_v4` sets prune so high that every window member is ranked. Ranking the full window costs runtime and still needs a ranker that can bury decoys.

Widen the pool after the ranker is strong. COCONUT plus train is about 711k structures. ChEBI and LIPID MAPS are `use_bio` in `init`. `CFG.USE_BIO_DB` is false in the public kernel because extra decoys hurt a weak ranker. LOTUS, NPAtlas, and a mass-sliced PubChem table are the next pools. The full Gluzdov PubChem prior dataset is about 40.8 GB and is not required for the first overnight val.

Canonical keys on the pool, via `scripts/pool_canon_keys.py`, merge tautomer copies so class 3 `drop_key` and COCONUT recall mean the same molecule.

### Channel 4, FPNet

Every public notebook near 0.33 shares `prvsiyan/casmi26-fp-models-v2`. Plan.md calls a better spectrum-to-fingerprint model the largest remaining gain.

`load_neural_models` takes only the folder passed as `models_dir`. A recursive glob of `data/` would load every `fp_*.pt` file and change logits in silence.

A new checkpoint changes 12 of the 31 ranker features. Ranker v4 was fit on v2 logits. Own-three models with ranker v4 lost 0.044 class 2 MRR against 0.758. Dump rows with `--models` pointed at the new files, refit, then measure. A new FPNet under ranker v4 is not a test of the model.

Bernoulli log-likelihood of bits given logits beats raw dot product. Raw dot product prefers dense fingerprints. `eval_channel4.py` compares those two without the ranker.

### Ranker rows and objective

`rank_train.npz` is 142,762 rows by 31 features from `prvsiyan/casmi26-ranker-features`. The dump method is unpublished.

`fit_rankers` trains two class-weight priors times four seeds of `HistGradientBoostingClassifier`. Inference truncates features to `NFEAT` from that file.

Rows from random timsTOF library compounds ranked rankers in the wrong order versus the board. np-examples class 2 ordered them correctly. Dump new rows from structures that look like np-examples or from COCONUT overlap, with the same prune and the same checkpoints the kernel will load.

A listwise objective such as LambdaRank on MRR may beat pointwise classification. Measure on np-examples class 2 and class 3 before a submit.

### Library and analog channels

Class 1 is already near 0.99 when library spectra stay visible. `lib_hard` 0.80 in fork v2 forced strong library hits to the top and cost 0.034 class 2 MRR, because an analog of a hidden structure often scores above 0.80. Leave `lib_hard` off for class 2 work. `lib_gate` that zeros weak library evidence changed nothing on Sep 18.

Analog `SIM_POWER` 4 beat 1 by about 0.027 class 2 in the public config comments. Extra reference spectra from GNPS or MassBank would help class 2 if the rules still allow the attached datasets.

Library search ignores polarity, so a negative-mode reference can match a positive query through the mass-shift analog path. Fixing that is a small, unmeasured gain.

### Fragmentation

Channel 3 enumerates one-bond and two-bond breaks. Plan.md reports that molecules with more than 34 bonds return only the molecular mass. Most of the test mass range hits that cap. Fragmentation features stay near-dead until the cap is gone.

### Ranking glue that already works

8.5 ppm beat 10 ppm by 0.003 class 2. The window is not the first knob to turn.

Median neutral mass across adducts of one molecule already fuses the 108 test molecules with mixed adducts. A hard "every adduct must fit" rule is cleaner and unmeasured.

`dedupe_fill` writes 25 distinct canonical keys and fills from nearest mass. Empty windows in weaker notebooks submitted `C` and left rank slots unused.

Public leaderboard noise is about 0.006 on 133 molecules. Pick kernels by np-examples class 2 and class 3, not by a 0.003 public bump.

## What already failed as a shortcut

Retraining the ranker on own-three FPNet features did not beat 0.758 class 2. The bottleneck was the row source, not the GBM.

Mixing public v4 rows with rows from a different FPNet taught one averaged trust in channel 4 that fit neither model.

Turning on ChEBI and LIPID MAPS without a stronger ranker adds decoys.

The analog LGBM in `scripts/pipeline.py` never reproduced public class 2.

## After the overnight val

If class 2 lands near 0.758 with prune `400:lib` and ranker v4, the local copy matches the public engine. The next submit is `kaggle/fork_v*` with those flags, not a rewrite of `pipeline.py`.

If class 2 is far below 0.758, the log from `init` is the first check: ranker path, `fp_*.pt` folder, and RDKit. Then `diag_fork_val.py` for window, prune, and rank.

After a match, the next point of work is a new FPNet plus a ranker dump from those logits, then a pool with canonical keys. Prune and PubChem come after that ranker can rank a larger window.
