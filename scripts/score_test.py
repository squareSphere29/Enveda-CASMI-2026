"""Score the test set: run the pipeline on test.parquet, write submission.csv.

Checks the submission rules: 400 molecule_ids, unique, <=25 guesses each, no nulls.
"""

import pickle
import time

import numpy as np
import pandas as pd

import pipeline as pl

DATA = "d:/Dev/KaggleEnvedaCASMI"


def score_test(model_pkl=f"{DATA}/data/model.pkl", out_csv=f"{DATA}/submission.csv", verbose=True):
    t0 = time.time()
    ctx = pl.build_context()
    test = pd.read_parquet(f"{DATA}/data/test.parquet")
    queries = pl.test_queries(test)
    pl.log(f"{len(queries)} test molecules")
    recs = pl.collect_records(ctx, queries)
    pl.log(f"{len(recs)} candidate records")
    with open(model_pkl, "rb") as fh:
        model = pickle.load(fh)
    preds = model.predict(pl.feats_array(recs))
    rows = pl.submission_rows(ctx, recs, preds, max_len=25)
    sub = pd.DataFrame(rows, columns=["molecule_id", "smiles"])
    sub.to_csv(out_csv, index=False)

    # ---- submission rules check
    assert len(sub) == len(test["molecule_id"].unique()), (len(sub), test["molecule_id"].nunique())
    assert sub["molecule_id"].is_unique
    assert sub["smiles"].notna().all() and (sub["smiles"] != "").all()
    for s in sub["smiles"]:
        assert len([x for x in s.split(";") if x]) <= 25
    pl.log(f"submission written to {out_csv} ({len(sub)} rows) in {time.time() - t0:.0f}s")

    # ---- info: overlap with train pool (truth-side sanity on the visible set)
    train_keys = set(ctx["pool"].index)
    by_q = {}
    for r, p in zip(recs, preds):
        by_q.setdefault(r["qid"], []).append((p, r["key"]))
    n_top1_in_train = sum(1 for qid in by_q if by_q[qid][0][1] in train_keys)
    pl.log(f"top-1 candidates: {n_top1_in_train}/{len(by_q)} (all candidates come from the train pool by construction)")
    return sub


if __name__ == "__main__":
    score_test()
