"""Patch pipeline.py + validate.py: np-set doc prior, median-precursor adduct, panel fixes."""
from pathlib import Path

p = Path("d:/Dev/KaggleEnvedaCASMI/scripts/pipeline.py")
src = p.read_text(encoding="utf-8")

# 1) doc_prior helper before ranker_anchored
anchor = "def ranker_anchored(ctx, cand_list):"
helper = '''def doc_prior(ctx, key):
    """Documentation / natural-product prior: the np-examples panel gets the top
    weight; other ingested libraries their own weights."""
    if key in ctx.get("np_set", ()):
        return 4.0
    return float(ctx["col"]["doc_prior"][ctx["pos_of_key"][key]])


'''
if "def doc_prior(" not in src:
    assert anchor in src, "anchor ranker_anchored missing"
    src = src.replace(anchor, helper + anchor, 1)

# 2) ranker_anchored: add the np doc prior term
old = '        sc = 0.12 * np.log10(r["n_rows"] + 1) + 0.12 * c["spec_cos"] + 0.12 * np_likeness(r)\n'
new = old + '        sc += 0.12 * doc_prior(ctx, c["key"])\n'
if old in src:
    assert src.count(old) == 1
    src = src.replace(old, new, 1)
else:
    assert 'sc += 0.12 * doc_prior(ctx, c["key"])' in src

# 3) cand_features doc_prior (tolerant)
old = '        "doc_prior": float(r["doc_prior"]),\n'
if old in src:
    assert src.count(old) == 1
    src = src.replace(old, '        "doc_prior": doc_prior(ctx, cand["key"]),\n', 1)
else:
    assert '"doc_prior": doc_prior(ctx, cand["key"])' in src

# 4) train_queries: adduct of the median-precursor row
new_train = '''def train_queries(train_df, ctx):
    """Per molecule key: query spectrum (merged train rows), median precursor,
    adduct of the median-precursor row, first ionization mode."""
    key_col = next(c for c in train_df.columns if "key14" in c)
    g = train_df.groupby(key_col)
    prec_map = g["precursor_mz"].median()
    mode_map = g["ionization_mode"].first()

    def med_adduct(d):
        i = d["precursor_mz"].sub(np.median(d["precursor_mz"])).abs().idxmin()
        return str(d.at[i, "adduct"])

    adduct_map = g.apply(med_adduct, include_groups=False)
    queries = {}
    for key in ctx["pool"].index:
        if key not in prec_map.index:
            continue
        lm, li = spec_segment(ctx, key)
        queries[str(key)] = {
            "q_mz": lm, "q_int": li, "prec": float(prec_map[key]),
            "adduct": str(adduct_map[key]), "ionization_mode": str(mode_map[key]),
        }
    return queries
'''
start = src.index("def train_queries")
end = src.index("def test_queries")
src = src[:start] + new_train + "\n\n" + src[end:]

# 5) test_queries: adduct of the median-precursor row
new_test = '''def test_queries(test_df):
    q = {}
    for mid, sub in test_df.groupby("molecule_id"):
        q_mz = np.concatenate([np.asarray(a, np.float64) for a in sub.ms2_mzs])
        q_int = np.concatenate([np.asarray(a, np.float64) for a in sub.ms2_normalized_intensities])
        o = np.argsort(q_mz, kind="stable")
        q_mz, q_int = q_mz[o], q_int[o]
        med = float(np.median(sub["precursor_mz"]))
        i = sub["precursor_mz"].sub(med).abs().idxmin()
        q[str(mid)] = {
            "q_mz": q_mz, "q_int": q_int, "prec": med,
            "adduct": str(sub.at[i, "adduct"]),
            "ionization_mode": str(sub.ionization_mode.iloc[0]),
        }
    return q
'''
start = src.index("def test_queries")
end = src.index("def submission_rows")
src = src[:start] + new_test + "\n\n" + src[end:]

p.write_text(src, encoding="utf-8")
print("pipeline.py patched")

v = Path("d:/Dev/KaggleEnvedaCASMI/scripts/validate.py")
src = v.read_text(encoding="utf-8")
old = '''    queries = pl.train_queries(tr, ctx)
    del tr
'''
new = '''    queries = pl.train_queries(tr, ctx)
    key_col = next(c for c in tr.columns if "key14" in c)
    np_set = set(tr.loc[tr["ingest_lib"] == "enveda-np-examples", key_col].unique())
    ctx["np_set"] = np_set
    log(f"panel (enveda-np-examples): {len(np_set)} molecules")
    del tr
'''
if old in src:
    assert src.count(old) == 1
    src = src.replace(old, new, 1)
else:
    assert "np_set" in src

old = '''    np_keys = sorted({k for k in queries
                      if ctx["col"]["ingest_lib"][ctx["pos_of_key"][k]] == "enveda-np-examples"})
    log(f"panel (enveda-np-examples): {len(np_keys)} molecules")
    recs_np = pl.collect_records(ctx, {k: queries[k] for k in np_keys})
    preds_np = model.predict(pl.feats_array(recs_np))
    mrr_np = pl.mrr(recs_np, preds_np)
'''
new = '''    np_keys = sorted(k for k in queries if k in np_set)
    recs_np = pl.collect_records(ctx, {k: queries[k] for k in np_keys})
    if recs_np:
        preds_np = model.predict(pl.feats_array(recs_np))
        mrr_np = pl.mrr(recs_np, preds_np)
    else:
        mrr_np = 0.0
'''
if old in src:
    assert src.count(old) == 1, f"panel block: {src.count(old)}"
    src = src.replace(old, new, 1)
else:
    assert 'np_keys = sorted(k for k in queries if k in np_set)' in src
v.write_text(src, encoding="utf-8")
print("validate.py patched")

import ast
for f in ("d:/Dev/KaggleEnvedaCASMI/scripts/pipeline.py", "d:/Dev/KaggleEnvedaCASMI/scripts/validate.py"):
    ast.parse(open(f).read())
print("both parse OK")
