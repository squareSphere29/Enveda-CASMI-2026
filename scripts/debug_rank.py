"""Check: ingest_lib values in pool vs the validate.py literal; print truth-rank sanity on 6 queries."""
import pickle

import numpy as np
import pandas as pd

import pipeline as pl

ctx = pl.build_context()
vals = sorted(set(ctx["col"]["ingest_lib"]))
print("pool ingest_lib values (first 10):", vals[:10])
print("count of enveda-np-examples in pool:", int(np.sum(np.array(ctx["col"]["ingest_lib"]) == "enveda-np-examples")))

model = pickle.load(open("d:/Dev/KaggleEnvedaCASMI/data/model.pkl", "rb"))
tr = pd.read_parquet("d:/Dev/KaggleEnvedaCASMI/data/train.parquet")
queries = pl.train_queries(tr, ctx)
del tr

recs = pl.collect_records(ctx, {k: queries[k] for k in list(queries)[:400]}, verbose=False)
X = pl.feats_array(recs)
preds = model.predict(X)
from collections import defaultdict
by_q = defaultdict(list)
for r, p in zip(recs, preds):
    by_q[r["qid"]].append((p, r["label"], r["key"], r["feats"]["spec_cos"]))
n_truth = 0
ranks = []
for qid, lst in by_q.items():
    truth = [t for t in lst if t[1] == 1]
    if not truth:
        continue
    n_truth += 1
    ranked = sorted(lst, key=lambda t: -t[0])
    for pos, t in enumerate(ranked):
        if t[1] == 1:
            ranks.append(pos + 1)
            break
    print(f"{qid}: truth_in_cands=1, rank={ranks[-1]}, top spec_cos={max(t[3] for t in lst):.3f}, top_key==truth: {ranked[0][2] == truth[0][2]}")
print("queries with truth in cands:", n_truth, "of", len(by_q), "| rank histogram:", pd.Series(ranks).value_counts().head(10).to_dict())
