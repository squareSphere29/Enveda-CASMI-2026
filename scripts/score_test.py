"""Score the test set: run the pipeline on test.parquet, write submission.csv."""

import json
import pickle
from pathlib import Path

import pandas as pd

import pipeline as pl

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def score_test(model_pkl=None, out_csv=None, verbose=True):
    model_pkl = model_pkl or str(DATA / "model.pkl")
    out_csv = out_csv or str(ROOT / "submission.csv")
    ctx = pl.build_context()
    np_path = DATA / "np_keys.json"
    if np_path.exists():
        ctx["np_set"] = set(json.loads(np_path.read_text()))
    else:
        ctx["np_set"] = set(
            str(k) for k, lib in zip(ctx["pool"].index, ctx["col"]["ingest_lib"])
            if lib == "enveda-np-examples"
        )
    test = pd.read_parquet(DATA / "test.parquet")
    queries = pl.test_queries(test)
    pl.log(f"{len(queries)} test molecules")
    recs, report = pl.collect_records(ctx, queries)
    pl.log(f"{len(recs)} candidate records; mass miss={report['n_miss_mass']}")
    with open(model_pkl, "rb") as fh:
        model = pickle.load(fh)
    preds = model.predict(pl.feats_array(recs))
    rows = pl.submission_rows(ctx, recs, preds, max_len=25)
    sub = pd.DataFrame(rows, columns=["molecule_id", "smiles"])
    # molecules with no candidates still need a row
    have = set(sub["molecule_id"].astype(str))
    for mid in queries:
        if str(mid) not in have:
            rows.append((str(mid), ctx["smiles"][0]))
    sub = pd.DataFrame(rows, columns=["molecule_id", "smiles"]).drop_duplicates("molecule_id")
    sub.to_csv(out_csv, index=False)

    assert sub["molecule_id"].is_unique
    assert sub["smiles"].notna().all() and (sub["smiles"] != "").all()
    for s in sub["smiles"]:
        assert len([x for x in s.split(";") if x]) <= 25
    pl.log(f"submission written to {out_csv} ({len(sub)} rows)")
    return sub


if __name__ == "__main__":
    score_test()
