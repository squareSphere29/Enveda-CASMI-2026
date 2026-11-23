"""Leak-free local validation: held-out train spectra, tight mass windows,
identity-disjoint folds, held-out np panel. Default 200 queries so it finishes."""

import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

import pipeline as pl

log = pl.log
ROOT = Path(__file__).resolve().parent.parent
TRAIN = str(ROOT / "data" / "train.parquet")
MODEL_PKL = str(ROOT / "data" / "model.pkl")
NP_KEYS = ROOT / "data" / "np_keys.json"


def _read_keys(path, key_col, extra_cols):
    cols = [key_col] + extra_cols
    try:
        return pd.read_parquet(path, columns=cols)
    except Exception:
        return pd.read_parquet(path)[cols]


def _read_rows(path, key_col, keys):
    keys = list(keys)
    try:
        return pd.read_parquet(path, filters=[(key_col, "in", keys)])
    except Exception:
        df = pd.read_parquet(path)
        return df[df[key_col].isin(keys)]


def run_validation(n_folds=5, max_queries=200, seed=42, save_model=True, theo_cap=40000, verbose=True):
    t0 = time.time()
    ctx = pl.build_context()
    ctx["theo_cap"] = theo_cap

    log("read train key column")
    meta = _read_keys(TRAIN, "inchikey14", ["ingest_lib"])
    key_col = "inchikey14"
    np_set = set(meta.loc[meta["ingest_lib"] == "enveda-np-examples", key_col].astype(str).unique())
    ctx["np_set"] = np_set
    NP_KEYS.write_text(json.dumps(sorted(np_set)))
    log(f"panel (enveda-np-examples): {len(np_set)} molecules")

    pool_keys = [str(k) for k in ctx["pool"].index]
    non_np = [k for k in pool_keys if k not in np_set]
    rng = np.random.default_rng(seed)
    n_cv = min(max_queries, len(non_np))
    cv_keys = sorted(rng.choice(np.array(non_np, dtype=object), size=n_cv, replace=False).tolist())
    load_keys = set(cv_keys) | np_set
    log(f"load {len(load_keys)} train molecules (cv={len(cv_keys)} + panel)")
    tr = _read_rows(TRAIN, key_col, load_keys)
    log("build held-out-spectrum query table")
    queries = pl.train_queries(tr, ctx)
    cv_queries = {k: queries[k] for k in cv_keys if k in queries}
    missing_q = [k for k in cv_keys if k not in cv_queries]
    if missing_q:
        log(f"{len(missing_q)} sampled keys had no train rows")
    folds = [set(f) for f in pl.make_folds(list(cv_queries), n_folds=n_folds, seed=seed)]

    log("collect records for cv queries")
    recs, report = pl.collect_records(ctx, cv_queries)
    log(f"{len(recs)} candidate records; mass miss={report['n_miss_mass']} "
        f"recall@{100}={report['recall_at_max']:.3f}")
    expected = list(cv_queries)
    if not recs:
        log("no records")
        return {"fold_mrrs": [], "panel_mrr": 0.0, "in_fold_mrr": 0.0, "n_records": 0,
                "recall": report}

    pos_q = {r["qid"] for r in recs if r["label"] == 1}
    fit_recs = [r for r in recs if r["qid"] in pos_q]
    log(f"{len(pos_q)}/{len(expected)} queries have the truth in the cosine shortlist")
    X = pl.feats_array(fit_recs)
    y = np.array([r["label"] for r in fit_recs])
    qids = np.array([r["qid"] for r in fit_recs])

    fold_results = []
    for f in range(n_folds):
        te = np.array([q in folds[f] for q in qids])
        tr_mask = ~te
        expected_te = [q for q in expected if q in folds[f]]
        if not tr_mask.any() or not te.any():
            fold_results.append((f, 0, float("nan")))
            continue
        pred_te = pl.fit_predict_fold(X[tr_mask], y[tr_mask], qids[tr_mask], X[te])
        m = pl.mrr([r for r, t in zip(recs, te) if t], pred_te, expected_qids=expected_te)
        n_q = len(expected_te)
        fold_results.append((f, n_q, m))
        log(f"fold {f}: {n_q} queries, {te.sum()} records, MRR = {m:.4f}")

    log("train final model on cv data")
    model = pl.train_final(X, y, qids)
    if save_model:
        with open(MODEL_PKL, "wb") as fh:
            pickle.dump(model, fh)
        log(f"model saved to {MODEL_PKL}")

    preds_all = model.predict(X)
    mrr_all = pl.mrr(recs, preds_all, expected_qids=expected)

    panel_keys = sorted(k for k in np_set if k in queries and k not in cv_queries)
    recs_np, panel_report = pl.collect_records(ctx, {k: queries[k] for k in panel_keys})
    if recs_np:
        preds_np = model.predict(pl.feats_array(recs_np))
        mrr_np = pl.mrr(recs_np, preds_np, expected_qids=panel_keys)
    else:
        mrr_np = 0.0

    fold_mrr = [m for _, _, m in fold_results]
    log(f"fold MRRs: {['%.4f' % m for m in fold_mrr]} mean={np.nanmean(fold_mrr):.4f}")
    log(f"panel (np held-out) MRR: {mrr_np:.4f} n={len(panel_keys)} "
        f"mass_miss={panel_report['n_miss_mass']} recall={panel_report['recall_at_max']:.3f}")
    log(f"cv in-sample MRR: {mrr_all:.4f}")
    log(f"validation wall time: {time.time() - t0:.0f}s")
    return {"fold_mrrs": fold_mrr, "panel_mrr": mrr_np, "in_fold_mrr": mrr_all,
            "n_records": len(recs), "recall": report, "panel_recall": panel_report}


if __name__ == "__main__":
    run_validation()
