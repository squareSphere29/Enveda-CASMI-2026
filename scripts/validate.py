"""Local validation of the CASMI26 pipeline.

Protocols:
  - Identity-disjoint 5-fold CV over train: collect candidate records for a
    deterministic subsample of train queries (default 24k of the 275,810
    molecules), then per fold: fit the blend (LightGBM ranker over our
    channels + the three public rankers) on the records of the other folds and
    score the fold with MRR (reciprocal rank of the truth in the top 25).
  - The natural-product panel benchmark: MRR over the enveda-np-examples
    molecules (the closest library to the test set), using the final model.

Outputs: fold MRRs, panel MRR, saves the final model (trained on all data) to
data/model.pkl.
"""

import pickle
import time

import numpy as np
import pandas as pd

import pipeline as pl

log = pl.log

TRAIN = "d:/Dev/KaggleEnvedaCASMI/data/train.parquet"
MODEL_PKL = "d:/Dev/KaggleEnvedaCASMI/data/model.pkl"


def run_validation(n_folds=5, max_queries=24000, seed=42, save_model=True, theo_cap=40000, verbose=True):
    t0 = time.time()
    ctx = pl.build_context()
    ctx["theo_cap"] = theo_cap
    log("read train.parquet for query table")
    tr = pd.read_parquet(TRAIN)
    log("build train query table")
    queries = pl.train_queries(tr, ctx)
    key_col = next(c for c in tr.columns if "key14" in c)
    np_set = set(tr.loc[tr["ingest_lib"] == "enveda-np-examples", key_col].unique())
    ctx["np_set"] = np_set
    log(f"panel (enveda-np-examples): {len(np_set)} molecules")
    del tr
    log(f"{len(queries)} molecules with queries")

    keys = sorted(queries)
    if max_queries is not None and len(keys) > max_queries:
        rng = np.random.default_rng(seed)
        keys = sorted(rng.choice(np.array(keys, dtype=object), size=max_queries, replace=False))
        log(f"subsampled {len(keys)} molecules")
    queries = {k: queries[k] for k in keys}
    folds = pl.make_folds(keys, n_folds=n_folds, seed=seed)
    folds = [set(f) for f in folds]

    log("collect records for validation queries")
    recs = pl.collect_records(ctx, queries)
    log(f"{len(recs)} candidate records")
    X = pl.feats_array(recs)
    y = np.array([r["label"] for r in recs])
    qids = np.array([r["qid"] for r in recs])

    fold_results = []
    for f in range(n_folds):
        te = np.array([q in folds[f] for q in qids])
        tr_mask = ~te
        if not tr_mask.any():
            fold_results.append((f, 0, float("nan")))
            continue
        pred_te = pl.fit_predict_fold(X[tr_mask], y[tr_mask], qids[tr_mask], X[te])
        m = pl.mrr([r for r, t in zip(recs, te) if t], pred_te)
        n_q = len({q for q in qids[te]})
        fold_results.append((f, n_q, m))
        log(f"fold {f}: {n_q} queries, {te.sum()} records, MRR = {m:.4f}")

    # ---- final model on all data
    log("train final model on all validation data")
    model = pl.train_final(X, y, qids)
    if save_model:
        with open(MODEL_PKL, "wb") as fh:
            pickle.dump(model, fh)
        log(f"model saved to {MODEL_PKL}")

    # ---- in-fold MRR of the final model (in-sample; optimistic)
    preds_all = model.predict(X)
    mrr_all = pl.mrr(recs, preds_all)

    # ---- panel benchmark: np-examples molecules
    np_keys = sorted(k for k in queries if k in np_set)
    recs_np = pl.collect_records(ctx, {k: queries[k] for k in np_keys})
    if recs_np:
        preds_np = model.predict(pl.feats_array(recs_np))
        mrr_np = pl.mrr(recs_np, preds_np)
    else:
        mrr_np = 0.0

    fold_mrr = [m for _, _, m in fold_results]
    log(f"fold MRRs: {['%.4f' % m for m in fold_mrr]} mean={np.nanmean(fold_mrr):.4f}")
    log(f"panel (np-examples) MRR: {mrr_np:.4f}")
    log(f"all-records MRR (in-fold): {mrr_all:.4f}")
    log(f"validation wall time: {time.time() - t0:.0f}s")
    return {"fold_mrrs": fold_mrr, "panel_mrr": mrr_np, "in_fold_mrr": mrr_all,
            "n_records": len(recs)}


if __name__ == "__main__":
    import sys
    run_validation()
