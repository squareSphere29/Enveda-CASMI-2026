"""Focused debug: is the true molecule in mass_candidates for real pool molecules?"""
import numpy as np
import pandas as pd

import pipeline as pl

ctx = pl.build_context()
tr = pd.read_parquet("d:/Dev/KaggleEnvedaCASMI/data/train.parquet")
key_col = next(c for c in tr.columns if "key14" in c)
stats = ctx["adduct_stats"]

n_in = 0
n_tot = 0
for key in list(ctx["pool"].index)[:60]:
    pos = ctx["pos_of_key"][key]
    m = ctx["col"]["mass"][pos]
    sub = tr[tr[key_col] == key]
    med = float(np.median(sub["precursor_mz"]))
    # query adduct like med_adduct
    i = sub["precursor_mz"].sub(med).abs().idxmin()
    adduct = str(sub.at[i, "adduct"])
    cands = pl.mass_candidates(ctx, med, adduct, str(sub["ionization_mode"].iloc[0]))
    ppm = (med - m) / m * 1e6
    st = stats.get(adduct, stats.get("__default__", {}))
    off = st.get("offset_ppm", 0.0)
    tol = st.get("tol_ppm", 0.0)
    in_c = key in cands
    n_tot += 1
    n_in += int(in_c)
    if not in_c and n_in < 5:
        print(f"MISS key={key[:12]} m={m:.3f} prec={med:.4f} adduct={adduct} ppm={ppm:.2f} off={off:.2f} tol={tol:.2f} lo={med/(1+(off+tol)/1e6):.4f} hi={med/(1+(off-tol)/1e6):.4f} ncands={len(cands)}")
print(f"truth in mass_candidates: {n_in}/{n_tot}")
